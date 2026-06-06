from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from xgboost import XGBClassifier

from engagement_daisee.common.config import CHECKPOINT_DIR, RANDOM_SEED
from engagement_daisee.mediapipe.extract_features import FEATURE_COLUMNS


LOGGER = logging.getLogger("mediapipe_train_product")
DEFAULT_OUTPUT_DIR = CHECKPOINT_DIR / "runs" / "mediapipe_product"


class TinyTCN(nn.Module):
    def __init__(self, input_size: int, channels: int = 32, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(channels, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = sequence.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.classifier(x).view(-1)


class TinyTCNInference(nn.Module):
    def __init__(self, model: TinyTCN, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean.view(1, 1, -1).float())
        self.register_buffer("std", std.view(1, 1, -1).float())

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = (sequence - self.mean) / self.std
        return torch.sigmoid(self.model(x))


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b <= 0 else float(a / b)


def _compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    return {
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": float((recall[0] + recall[1]) / 2.0),
        "precision_neg": float(precision[0]),
        "recall_neg": float(recall[0]),
        "f1_neg": float(f1[0]),
        "precision_pos": float(precision[1]),
        "recall_pos": float(recall[1]),
        "f1_pos": float(f1[1]),
        "f1_macro": float((f1[0] + f1[1]) / 2.0),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray, objective: str) -> tuple[float, dict[str, float]]:
    best_t = 0.5
    best_m = _compute_metrics(labels, probabilities, best_t)
    for t in np.arange(0.05, 0.96, 0.01):
        m = _compute_metrics(labels, probabilities, float(t))
        score = m[objective]
        best_score = best_m[objective]
        if score > best_score + 1e-8 or (abs(score - best_score) <= 1e-8 and m["recall_neg"] > best_m["recall_neg"]):
            best_t = float(t)
            best_m = m
    return best_t, best_m


def _load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"feature_path", "video_id", "split", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def _pad_or_sample(sequence: np.ndarray, seq_len: int) -> np.ndarray:
    sequence = np.nan_to_num(sequence.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(sequence) == seq_len:
        return sequence
    if len(sequence) > seq_len:
        idx = np.linspace(0, len(sequence) - 1, seq_len).round().astype(int)
        return sequence[idx]
    if len(sequence) == 0:
        return np.zeros((seq_len, len(FEATURE_COLUMNS)), dtype=np.float32)
    pad = np.repeat(sequence[-1:], seq_len - len(sequence), axis=0)
    return np.concatenate([sequence, pad], axis=0)


def _sequence_stats(sequence: np.ndarray) -> np.ndarray:
    seq = np.asarray(sequence, dtype=np.float32)
    t = np.arange(len(seq), dtype=np.float32)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc) + 1e-6)
    centered = seq - seq.mean(axis=0, keepdims=True)
    slope = (tc[:, None] * centered).sum(axis=0) / denom
    diff = np.diff(seq, axis=0)
    mean_abs_diff = np.mean(np.abs(diff), axis=0) if len(diff) else np.zeros(seq.shape[1], dtype=np.float32)
    p10 = np.percentile(seq, 10, axis=0)
    p90 = np.percentile(seq, 90, axis=0)
    return np.concatenate(
        [
            seq.mean(axis=0),
            seq.std(axis=0),
            seq.min(axis=0),
            seq.max(axis=0),
            p10,
            p90,
            slope.astype(np.float32),
            mean_abs_diff.astype(np.float32),
        ]
    ).astype(np.float32)


def _load_arrays(manifest: pd.DataFrame, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = []
    tabular = []
    labels = []
    for row in manifest.itertuples(index=False):
        seq = np.load(Path(str(row.feature_path)))
        seq = _pad_or_sample(seq, seq_len)
        sequences.append(seq)
        tabular.append(_sequence_stats(seq))
        labels.append(int(row.label))
    return (
        np.stack(sequences).astype(np.float32),
        np.stack(tabular).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def _split_arrays(manifest: pd.DataFrame, seq_len: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for split in ("train", "validation", "test"):
        part = manifest[manifest["split"] == split].copy()
        if part.empty:
            raise ValueError(f"Empty split: {split}")
        out[split] = _load_arrays(part, seq_len)
    return out


def _benchmark_callable(fn, sample, iters: int = 500) -> float:
    for _ in range(30):
        fn(sample)
    start = time.perf_counter()
    for _ in range(iters):
        fn(sample)
    return (time.perf_counter() - start) * 1000.0 / iters


def _train_logistic(data, output_dir: Path, objective: str, seed: int) -> dict:
    x_train, y_train = data["train"][1], data["train"][2]
    x_val, y_val = data["validation"][1], data["validation"][2]
    x_test, y_test = data["test"][1], data["test"][2]
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=1)),
        ]
    )
    started = time.time()
    model.fit(x_train, y_train)
    val_probs = model.predict_proba(x_val)[:, 1]
    threshold, val_metrics = _select_threshold(y_val, val_probs, objective)
    test_probs = model.predict_proba(x_test)[:, 1]
    test_metrics = _compute_metrics(y_test, test_probs, threshold)
    latency = _benchmark_callable(lambda x: model.predict_proba(x.reshape(1, -1))[:, 1], x_test[0])
    out = output_dir / "logistic_learned_heuristic"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    result = {
        "model": "logistic_learned_heuristic",
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "latency_ms": latency,
        "elapsed_sec": time.time() - started,
        "artifact": str(out / "model.joblib"),
        "top_coefficients": _top_logistic_coefficients(model),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _top_logistic_coefficients(model: Pipeline, top_k: int = 20) -> list[dict[str, object]]:
    clf = model.named_steps["clf"]
    coef = clf.coef_.reshape(-1)
    stats = ["mean", "std", "min", "max", "p10", "p90", "slope", "mean_abs_diff"]
    names = [f"{stat}:{name}" for stat in stats for name in FEATURE_COLUMNS]
    order = np.argsort(np.abs(coef))[::-1][:top_k]
    return [{"feature": names[i], "coefficient": float(coef[i])} for i in order]


def _train_xgb(data, output_dir: Path, objective: str, seed: int) -> dict:
    x_train, y_train = data["train"][1], data["train"][2]
    x_val, y_val = data["validation"][1], data["validation"][2]
    x_test, y_test = data["test"][1], data["test"][2]
    counts = np.bincount(y_train, minlength=2)
    scale_pos_weight = _safe_div(counts[0], counts[1]) if counts[1] else 1.0
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=180,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    started = time.time()
    model.fit(x_train, y_train, verbose=False)
    val_probs = model.predict_proba(x_val)[:, 1]
    threshold, val_metrics = _select_threshold(y_val, val_probs, objective)
    test_probs = model.predict_proba(x_test)[:, 1]
    test_metrics = _compute_metrics(y_test, test_probs, threshold)
    latency = _benchmark_callable(lambda x: model.predict_proba(x.reshape(1, -1))[:, 1], x_test[0])
    out = output_dir / "xgboost_shallow"
    out.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(out / "model.json"))
    result = {
        "model": "xgboost_shallow",
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "latency_ms": latency,
        "elapsed_sec": time.time() - started,
        "artifact": str(out / "model.json"),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _normalize_sequences(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0).astype(np.float32)
    std = train.reshape(-1, train.shape[-1]).std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, [(x - mean) / std for x in others], mean, std


def _tcn_probs(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=batch_size, shuffle=False)
    probs = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def _train_tiny_tcn(data, output_dir: Path, objective: str, seed: int, device_name: str, epochs: int) -> dict:
    _set_seed(seed)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    x_train, y_train = data["train"][0], data["train"][2]
    x_val, y_val = data["validation"][0], data["validation"][2]
    x_test, y_test = data["test"][0], data["test"][2]
    x_train_n, (x_val_n, x_test_n), mean, std = _normalize_sequences(x_train, x_val, x_test)

    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = np.asarray([1.0 / max(1.0, counts[y]) for y in y_train], dtype=np.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_n).float(), torch.from_numpy(y_train).float()),
        batch_size=128,
        sampler=sampler,
        num_workers=0,
    )
    model = TinyTCN(input_size=x_train.shape[-1]).to(device)
    pos_weight = torch.tensor([_safe_div(counts[0], counts[1]) if counts[1] else 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = -1.0
    best_state = None
    bad = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_probs = _tcn_probs(model, x_val_n, device)
        _, val_metrics = _select_threshold(y_val, val_probs, objective)
        score = val_metrics[objective]
        if score > best_score + 1e-5:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0:
            LOGGER.info("tiny_tcn epoch=%d/%d val_%s=%.4f bal=%.4f", epoch, epochs, objective, score, val_metrics["balanced_accuracy"])
        if epoch >= 10 and bad >= 8:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    val_probs = _tcn_probs(model, x_val_n, device)
    threshold, val_metrics = _select_threshold(y_val, val_probs, objective)
    test_probs = _tcn_probs(model, x_test_n, device)
    test_metrics = _compute_metrics(y_test, test_probs, threshold)

    out = output_dir / "tiny_tcn"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "mean": mean, "std": std, "feature_columns": FEATURE_COLUMNS, "threshold": threshold},
        out / "model.pt",
    )
    wrapper = TinyTCNInference(TinyTCN(input_size=x_train.shape[-1]), torch.from_numpy(mean), torch.from_numpy(std)).eval()
    wrapper.model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    scripted = torch.jit.optimize_for_inference(torch.jit.script(wrapper))
    scripted.save(str(out / "model.ts"))
    cpu_model = torch.jit.load(str(out / "model.ts"), map_location="cpu").eval()
    torch.set_num_threads(2)
    sample = torch.from_numpy(x_test[:1]).float()
    latency = _benchmark_callable(lambda x: cpu_model(x), sample, iters=300)
    result = {
        "model": "tiny_tcn",
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "latency_ms": latency,
        "elapsed_sec": time.time() - started,
        "artifact": str(out / "model.ts"),
        "device": str(device),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_product_models(
    manifest_path: Path,
    output_dir: Path,
    seq_len: int,
    objective: str,
    seed: int,
    device: str,
    tcn_epochs: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    LOGGER.info("Loading arrays | manifest=%s rows=%d seq_len=%d", manifest_path, len(manifest), seq_len)
    data = _split_arrays(manifest, seq_len=seq_len)
    split_counts = {
        split: np.bincount(values[2], minlength=2).astype(int).tolist()
        for split, values in data.items()
    }
    LOGGER.info("Split counts: %s", split_counts)

    results = [
        _train_logistic(data, output_dir, objective, seed),
        _train_xgb(data, output_dir, objective, seed),
        _train_tiny_tcn(data, output_dir, objective, seed, device, tcn_epochs),
    ]
    results = sorted(results, key=lambda r: (r["test_metrics"]["balanced_accuracy"], r["test_metrics"]["f1_macro"]), reverse=True)
    summary = {
        "protocol": "mediapipe_product_cpu_models",
        "manifest": str(manifest_path),
        "seq_len": seq_len,
        "objective": objective,
        "split_counts": split_counts,
        "feature_columns": FEATURE_COLUMNS,
        "leaderboard": results,
    }
    (output_dir / "leaderboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPU-first product models on MediaPipe temporal features.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--objective", choices=["balanced_accuracy", "f1_macro", "recall_neg"], default="balanced_accuracy")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tcn-epochs", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    summary = train_product_models(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        objective=args.objective,
        seed=args.seed,
        device=args.device,
        tcn_epochs=args.tcn_epochs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
