from __future__ import annotations

import argparse
import json
import logging
import math
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from engagement_daisee.common.manifest import normalize_manifest_columns
from engagement_daisee.ml.train import _apply_feature_preprocessor, _build_feature_matrix, _load_feature_preprocessor
from engagement_daisee.multiclass.models import NUM_CLASSES, build_multiclass_model
from engagement_daisee.rnn.dataset import FeatureSequenceDataset


LOGGER = logging.getLogger("multiclass_late_fusion")
DEFAULT_BATCH_SIZE = 128
DEFAULT_FEATURE_MODE = "tsfresh"
DEFAULT_WEIGHT_STEP = 0.05
DEFAULT_POLL_SECONDS = 120
DEFAULT_LATENCY_THREADS = 2
DEFAULT_LATENCY_WARMUP = 30
DEFAULT_LATENCY_ITERS = 200


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _aggregate_by_video(manifest_subset: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "video_id" not in manifest_subset.columns:
        return labels, probabilities

    frame = manifest_subset[["video_id"]].copy().reset_index(drop=True)
    frame["label"] = labels.astype(np.int64)
    for idx in range(probabilities.shape[1]):
        frame[f"p_{idx}"] = probabilities[:, idx].astype(np.float32)
    grouped = frame.groupby("video_id", sort=False)
    video_labels = grouped["label"].first().to_numpy(dtype=np.int64)
    video_probabilities = grouped[[f"p_{idx}" for idx in range(probabilities.shape[1])]].mean().to_numpy(dtype=np.float32)
    return video_labels, video_probabilities


def _compute_multiclass_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, object]:
    labels = labels.astype(np.int64)
    predictions = predictions.astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "precision_per_class": [float(value) for value in precision],
        "recall_per_class": [float(value) for value in recall],
        "f1_per_class": [float(value) for value in f1],
        "support_per_class": [int(value) for value in support],
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES))).tolist(),
    }
    if probabilities is not None:
        eps = 1e-8
        clipped = np.clip(probabilities.astype(np.float64), eps, 1.0)
        metrics["cross_entropy"] = float(-np.mean(np.log(clipped[np.arange(len(labels)), labels])))
    return metrics


def _split_indices(manifest: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    split_series = manifest["split"].astype(str).str.strip().str.lower()
    train_indices = split_series[split_series == "train"].index.tolist()
    val_indices = split_series[split_series == "validation"].index.tolist()
    test_indices = split_series[split_series == "test"].index.tolist()
    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            "Official split requires non-empty train/validation/test rows in manifest. "
            f"Got train={len(train_indices)}, validation={len(val_indices)}, test={len(test_indices)}"
        )
    return train_indices, val_indices, test_indices


def _weight_grid(n_models: int, step: float) -> list[np.ndarray]:
    if n_models <= 0:
        raise ValueError("n_models must be positive")
    ticks = int(round(1.0 / step))
    if not math.isclose(ticks * step, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"weight_step={step} must divide 1.0 evenly")
    weights = []
    for raw in product(range(ticks + 1), repeat=n_models):
        if sum(raw) == ticks:
            weights.append(np.array(raw, dtype=np.float32) / ticks)
    return weights


def _stack_probs(probs_by_model: dict[str, np.ndarray], names: list[str], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack([probs_by_model[name] for name in names], axis=0)
    return np.sum(stacked * weights[:, None, None], axis=0)


def _is_better(candidate: dict[str, object], incumbent: dict[str, object] | None, objective: str) -> bool:
    if incumbent is None:
        return True
    cand_val = float(candidate["validation_metrics"][objective])
    inc_val = float(incumbent["validation_metrics"][objective])
    if cand_val > inc_val + 1e-8:
        return True
    if abs(cand_val - inc_val) <= 1e-8:
        cand_bal = float(candidate["validation_metrics"]["balanced_accuracy"])
        inc_bal = float(incumbent["validation_metrics"]["balanced_accuracy"])
        if cand_bal > inc_bal + 1e-8:
            return True
        if abs(cand_bal - inc_bal) <= 1e-8:
            cand_f1 = float(candidate["validation_metrics"]["f1_macro"])
            inc_f1 = float(incumbent["validation_metrics"]["f1_macro"])
            return cand_f1 > inc_f1 + 1e-8
    return False


def _build_component_checkpoint(model_dir: Path, model_name: str) -> Path:
    return model_dir / f"{model_name}.pt"


def _load_neural_component(
    manifest_df: pd.DataFrame,
    dataset: FeatureSequenceDataset,
    indices: list[int],
    checkpoint_path: Path,
    model_name: str,
    batch_size: int,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = dict(checkpoint.get("model_kwargs") or {})
    model = build_multiclass_model(model_name=model_name, **model_kwargs).to("cpu").eval()
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Missing model_state_dict in checkpoint: {checkpoint_path}")
    model.load_state_dict(state_dict)

    feature_mean = feature_std = None
    if checkpoint.get("feature_mean") is not None and checkpoint.get("feature_std") is not None:
        feature_mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32).view(1, 1, -1)
        feature_std = torch.tensor(checkpoint["feature_std"], dtype=torch.float32).view(1, 1, -1)

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to("cpu")
            labels = labels.to("cpu").long()
            if feature_mean is not None and feature_std is not None:
                features = (features - feature_mean) / feature_std
            logits = model(features)
            probs = torch.softmax(logits, dim=-1)
            labels_all.append(labels.cpu().numpy().astype(np.int64))
            probs_all.append(probs.cpu().numpy().astype(np.float32))

    labels_np = np.concatenate(labels_all)
    probs_np = np.concatenate(probs_all)
    manifest_subset = manifest_df.iloc[indices].reset_index(drop=True)
    video_labels, video_probs = _aggregate_by_video(manifest_subset, labels_np, probs_np)
    video_predictions = np.argmax(video_probs, axis=1)
    return {
        "kind": "neural",
        "checkpoint_path": str(checkpoint_path),
        "model_name": model_name,
        "model": model,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "labels": labels_np,
        "probs": probs_np,
        "video_labels": video_labels,
        "video_probs": video_probs,
        "row_metrics": _compute_multiclass_metrics(labels_np, np.argmax(probs_np, axis=1), probs_np),
        "video_metrics": _compute_multiclass_metrics(video_labels, video_predictions, video_probs),
        "sample_feature_path": str(manifest_subset.iloc[0]["feature_path"]),
    }


def _predict_xgb(model: XGBClassifier, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return model.predict_proba(x, iteration_range=(0, int(best_iteration) + 1)).astype(np.float32)
    return model.predict_proba(x).astype(np.float32)


def _load_xgb_component(
    manifest_df: pd.DataFrame,
    indices: list[int],
    run_root: Path,
    feature_mode: str,
) -> dict[str, object]:
    model_path = run_root / "xgboost" / "model.json"
    summary_path = run_root / "xgboost" / "summary.json"
    summary = json.loads(summary_path.read_text())
    resolved_feature_mode = str((summary.get("model_kwargs") or {}).get("feature_mode") or feature_mode)

    xgb = XGBClassifier()
    xgb.load_model(str(model_path))

    split_df = manifest_df.iloc[indices].reset_index(drop=True)
    x_eval, y_eval = _build_feature_matrix(split_df, feature_mode=resolved_feature_mode)
    preprocessor_path = None
    feature_norm = summary.get("feature_normalization") or {}
    if feature_norm.get("preprocessor_path"):
        preprocessor_path = Path(feature_norm["preprocessor_path"])
    elif (run_root / "xgboost" / "preprocessor.npz").exists():
        preprocessor_path = run_root / "xgboost" / "preprocessor.npz"

    if preprocessor_path is not None and preprocessor_path.exists():
        x_eval = _apply_feature_preprocessor(x_eval, _load_feature_preprocessor(preprocessor_path))

    probs = _predict_xgb(xgb, x_eval)
    video_labels, video_probs = _aggregate_by_video(split_df, y_eval, probs)
    video_predictions = np.argmax(video_probs, axis=1)
    return {
        "kind": "xgboost",
        "checkpoint_path": str(model_path),
        "summary_path": str(summary_path),
        "feature_mode": resolved_feature_mode,
        "model": xgb,
        "preprocessor_path": str(preprocessor_path) if preprocessor_path is not None else None,
        "labels": y_eval,
        "probs": probs,
        "video_labels": video_labels,
        "video_probs": video_probs,
        "row_metrics": _compute_multiclass_metrics(y_eval, np.argmax(probs, axis=1), probs),
        "video_metrics": _compute_multiclass_metrics(video_labels, video_predictions, video_probs),
        "sample_feature_path": str(split_df.iloc[0]["feature_path"]),
    }


def _align_scores(scores_by_model: dict[str, dict[str, object]]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    common_ids = None
    for score in scores_by_model.values():
        ids = set(score["video_ids"])
        common_ids = ids if common_ids is None else common_ids.intersection(ids)
    if not common_ids:
        raise ValueError("No common videos available across ensemble components")
    ordered_ids = [vid for vid in scores_by_model[next(iter(scores_by_model))]["video_ids"] if vid in common_ids]
    labels = np.array(
        [int(scores_by_model[next(iter(scores_by_model))]["video_label_map"][vid]) for vid in ordered_ids],
        dtype=np.int64,
    )
    probs = {
        name: np.array([score["video_prob_map"][vid] for vid in ordered_ids], dtype=np.float32)
        for name, score in scores_by_model.items()
    }
    return ordered_ids, labels, probs


def _load_split_component_scores(
    manifest_path: Path,
    run_root: Path,
    split: str,
    *,
    batch_size: int,
    feature_mode: str,
) -> dict[str, dict[str, object]]:
    manifest_df = normalize_manifest_columns(pd.read_csv(manifest_path))
    manifest_df["split"] = manifest_df["split"].astype(str).str.strip().str.lower()
    manifest_df["label"] = manifest_df["label"].astype(int)
    train_indices, val_indices, test_indices = _split_indices(manifest_df)
    indices = {"train": train_indices, "validation": val_indices, "test": test_indices}[split]
    dataset = FeatureSequenceDataset(manifest_path)
    component_specs = {
        "gru": ("neural", _build_component_checkpoint(run_root / "gru", "gru")),
        "tcn": ("neural", _build_component_checkpoint(run_root / "tcn", "tcn")),
        "xgboost": ("xgboost", run_root / "xgboost" / "model.json"),
    }

    results: dict[str, dict[str, object]] = {}
    for name, (kind, checkpoint_path) in component_specs.items():
        if kind == "neural":
            component = _load_neural_component(manifest_df, dataset, indices, checkpoint_path, name, batch_size)
        else:
            component = _load_xgb_component(manifest_df, indices, run_root, feature_mode)
        video_ids_ordered = (
            manifest_df.iloc[indices].reset_index(drop=True)["video_id"].astype(str).drop_duplicates().tolist()
        )
        if len(video_ids_ordered) != len(component["video_labels"]):
            raise ValueError(
                f"Video aggregation mismatch for {name}: ids={len(video_ids_ordered)} labels={len(component['video_labels'])}"
            )
        video_label_map = {
            vid: int(label)
            for vid, label in zip(video_ids_ordered, component["video_labels"].astype(np.int64).tolist(), strict=True)
        }
        video_prob_map = {
            vid: component["video_probs"][idx]
            for idx, vid in enumerate(video_ids_ordered)
        }
        results[name] = {
            **component,
            "video_ids": video_ids_ordered,
            "video_label_map": video_label_map,
            "video_prob_map": video_prob_map,
        }
    return results


def _build_latency_payload(
    manifest_path: Path,
    run_root: Path,
    feature_mode: str,
    batch_size: int,
    latency_threads: int,
    latency_warmup: int,
    latency_iters: int,
) -> dict[str, object]:
    manifest_df = normalize_manifest_columns(pd.read_csv(manifest_path))
    manifest_df["split"] = manifest_df["split"].astype(str).str.strip().str.lower()
    manifest_df["label"] = manifest_df["label"].astype(int)
    _, _, test_indices = _split_indices(manifest_df)
    test_df = manifest_df.iloc[test_indices].reset_index(drop=True)
    sample_row = test_df.iloc[0]
    sample_feature_path = Path(str(sample_row["feature_path"]))
    sample_sequence = np.load(sample_feature_path).astype(np.float32)
    sample_tensor = torch.from_numpy(sample_sequence[None, ...]).to("cpu")

    gru_component = _load_neural_component(manifest_df, FeatureSequenceDataset(manifest_path), test_indices, _build_component_checkpoint(run_root / "gru", "gru"), "gru", batch_size)
    tcn_component = _load_neural_component(manifest_df, FeatureSequenceDataset(manifest_path), test_indices, _build_component_checkpoint(run_root / "tcn", "tcn"), "tcn", batch_size)
    xgb_component = _load_xgb_component(manifest_df, test_indices, run_root, feature_mode)

    gru_model = gru_component["model"].to("cpu").eval()
    tcn_model = tcn_component["model"].to("cpu").eval()
    xgb = xgb_component["model"]
    gru_mean = gru_component["feature_mean"]
    gru_std = gru_component["feature_std"]
    tcn_mean = tcn_component["feature_mean"]
    tcn_std = tcn_component["feature_std"]
    xgb_preprocessor = None
    xgb_feature_mode = str(xgb_component["feature_mode"])
    if xgb_component["preprocessor_path"]:
        xgb_preprocessor = _load_feature_preprocessor(Path(str(xgb_component["preprocessor_path"])))
    sample_tabular = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode=xgb_feature_mode)[0]
    if xgb_preprocessor is not None:
        sample_tabular = _apply_feature_preprocessor(sample_tabular, xgb_preprocessor)

    torch.set_num_threads(max(1, latency_threads))

    def _fuse_preloaded() -> np.ndarray:
        with torch.no_grad():
            seq = sample_tensor
            if gru_mean is not None and gru_std is not None:
                seq_gru = (seq - gru_mean) / gru_std
            else:
                seq_gru = seq
            if tcn_mean is not None and tcn_std is not None:
                seq_tcn = (seq - tcn_mean) / tcn_std
            else:
                seq_tcn = seq
            gru_probs = torch.softmax(gru_model(seq_gru), dim=-1).cpu().numpy()[0]
            tcn_probs = torch.softmax(tcn_model(seq_tcn), dim=-1).cpu().numpy()[0]
            xgb_probs = _predict_xgb(xgb, sample_tabular[None, ...])[0]
            return (0.3 * gru_probs + 0.3 * tcn_probs + 0.4 * xgb_probs).astype(np.float32)

    def _fuse_from_disk() -> np.ndarray:
        seq = np.load(sample_feature_path).astype(np.float32)
        seq_tensor = torch.from_numpy(seq[None, ...])
        with torch.no_grad():
            if gru_mean is not None and gru_std is not None:
                seq_gru = (seq_tensor - gru_mean) / gru_std
            else:
                seq_gru = seq_tensor
            if tcn_mean is not None and tcn_std is not None:
                seq_tcn = (seq_tensor - tcn_mean) / tcn_std
            else:
                seq_tcn = seq_tensor
            gru_probs = torch.softmax(gru_model(seq_gru), dim=-1).cpu().numpy()[0]
            tcn_probs = torch.softmax(tcn_model(seq_tcn), dim=-1).cpu().numpy()[0]
        tabular = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode=xgb_feature_mode)[0]
        if xgb_preprocessor is not None:
            tabular = _apply_feature_preprocessor(tabular, xgb_preprocessor)
        xgb_probs = _predict_xgb(xgb, tabular[None, ...])[0]
        return (0.3 * gru_probs + 0.3 * tcn_probs + 0.4 * xgb_probs).astype(np.float32)

    def _timer_ms(fn, warmup: int, iters: int) -> dict[str, float]:
        for _ in range(max(0, warmup)):
            fn()
        samples: list[float] = []
        for _ in range(max(1, iters)):
            started = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - started) * 1000.0)
        ordered = sorted(samples)
        p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return {
            "latency_ms_mean": float(np.mean(samples)),
            "latency_ms_median": float(np.median(samples)),
            "latency_ms_p95": float(ordered[p95_index]),
            "latency_ms_min": float(min(samples)),
            "latency_ms_max": float(max(samples)),
        }

    model_side = _timer_ms(_fuse_preloaded, latency_warmup, latency_iters)
    end_to_end = _timer_ms(_fuse_from_disk, latency_warmup, latency_iters)
    return {
        "latency_kind": "processed_feature_sequence",
        "sample_feature_path": str(sample_feature_path),
        "model_side": {
            "variant": "gru+tcn+xgboost",
            **model_side,
        },
        "end_to_end": {
            "variant": "gru+tcn+xgboost",
            **end_to_end,
        },
    }


def run_late_fusion(
    manifest_path: Path,
    run_root: Path,
    output_json: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    weight_step: float = DEFAULT_WEIGHT_STEP,
    objective: str = "balanced_accuracy",
    wait_for_summary: bool = False,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    latency_threads: int = DEFAULT_LATENCY_THREADS,
    latency_warmup: int = DEFAULT_LATENCY_WARMUP,
    latency_iters: int = DEFAULT_LATENCY_ITERS,
) -> dict[str, object]:
    summary_marker = run_root / "train_all_summary.json"
    required_paths = [
        run_root / "gru" / "gru.pt",
        run_root / "tcn" / "tcn.pt",
        run_root / "xgboost" / "model.json",
        run_root / "xgboost" / "summary.json",
    ]

    if wait_for_summary:
        LOGGER.info("Waiting for base train_all summary: %s", summary_marker)
    while wait_for_summary and not summary_marker.exists():
        missing = [str(path) for path in required_paths if not path.exists()]
        LOGGER.info("Late-fusion stage waiting. Missing: %s", ", ".join(missing) if missing else "summary marker")
        time.sleep(max(1, poll_seconds))

    missing_after_wait = [path for path in required_paths if not path.exists()]
    if missing_after_wait:
        raise FileNotFoundError(
            "Late-fusion prerequisites are missing: " + ", ".join(str(path) for path in missing_after_wait)
        )

    manifest_df = normalize_manifest_columns(pd.read_csv(manifest_path))
    manifest_df["split"] = manifest_df["split"].astype(str).str.strip().str.lower()
    manifest_df["label"] = manifest_df["label"].astype(int)

    val_scores = _load_split_component_scores(manifest_path, run_root, "validation", batch_size=batch_size, feature_mode=feature_mode)
    test_scores = _load_split_component_scores(manifest_path, run_root, "test", batch_size=batch_size, feature_mode=feature_mode)

    val_ids, val_labels, val_probs = _align_scores(val_scores)
    test_ids, test_labels, test_probs = _align_scores(test_scores)

    names = ["gru", "tcn", "xgboost"]
    best: dict[str, object] | None = None
    for weights in _weight_grid(len(names), weight_step):
        fused = _stack_probs(val_probs, names, weights)
        val_predictions = np.argmax(fused, axis=1)
        val_metrics = _compute_multiclass_metrics(val_labels, val_predictions, fused)
        candidate = {
            "weights": {name: float(weight) for name, weight in zip(names, weights, strict=True)},
            "validation_metrics": val_metrics,
        }
        if _is_better(candidate, best, objective):
            best = candidate

    assert best is not None
    selected_weights = np.array([best["weights"][name] for name in names], dtype=np.float32)
    fused_test = _stack_probs(test_probs, names, selected_weights)
    test_predictions = np.argmax(fused_test, axis=1)
    test_metrics = _compute_multiclass_metrics(test_labels, test_predictions, fused_test)

    component_reports = {}
    for name in names:
        component_reports[name] = {
            "validation": _compute_multiclass_metrics(val_scores[name]["video_labels"], np.argmax(val_scores[name]["video_probs"], axis=1), val_scores[name]["video_probs"]),
            "test": _compute_multiclass_metrics(test_scores[name]["video_labels"], np.argmax(test_scores[name]["video_probs"], axis=1), test_scores[name]["video_probs"]),
        }

    latency = _build_latency_payload(
        manifest_path=manifest_path,
        run_root=run_root,
        feature_mode=feature_mode,
        batch_size=batch_size,
        latency_threads=latency_threads,
        latency_warmup=latency_warmup,
        latency_iters=latency_iters,
    )

    report = {
        "manifest": str(manifest_path),
        "run_root": str(run_root),
        "models": {
            "gru": str(run_root / "gru" / "gru.pt"),
            "tcn": str(run_root / "tcn" / "tcn.pt"),
            "xgboost": str(run_root / "xgboost" / "model.json"),
        },
        "video_counts": {
            "validation": int(len(val_ids)),
            "test": int(len(test_ids)),
        },
        "selected": {
            "weights": best["weights"],
            "validation_metrics": best["validation_metrics"],
            "validation_loss": float(best["validation_metrics"]["cross_entropy"]),
        },
        "test_metrics": test_metrics,
        "test_loss": float(test_metrics["cross_entropy"]),
        "component_reports": component_reports,
        "latency": latency,
        "note": "Weights were selected on validation only; test is held out for final reporting.",
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info(
        "[late_fusion] test video metrics acc=%.4f bal=%.4f f1=%.4f | model latency=%.3f ms | e2e=%.3f ms",
        float(test_metrics["accuracy"]),
        float(test_metrics["balanced_accuracy"]),
        float(test_metrics["f1_macro"]),
        float(latency["model_side"]["latency_ms_mean"]),
        float(latency["end_to_end"]["latency_ms_mean"]),
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train late-fusion GRU + TCN + XGBoost on the 4-class manifest outputs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--feature-mode", type=str, default=DEFAULT_FEATURE_MODE)
    parser.add_argument("--weight-step", type=float, default=DEFAULT_WEIGHT_STEP)
    parser.add_argument("--objective", type=str, default="balanced_accuracy", choices=["accuracy", "balanced_accuracy", "f1_macro"])
    parser.add_argument("--wait-for-summary", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--latency-threads", type=int, default=DEFAULT_LATENCY_THREADS)
    parser.add_argument("--latency-warmup", type=int, default=DEFAULT_LATENCY_WARMUP)
    parser.add_argument("--latency-iters", type=int, default=DEFAULT_LATENCY_ITERS)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = _parse_args()
    run_late_fusion(
        manifest_path=args.manifest,
        run_root=args.run_root,
        output_json=args.output_json,
        batch_size=args.batch_size,
        feature_mode=args.feature_mode,
        weight_step=args.weight_step,
        objective=args.objective,
        wait_for_summary=args.wait_for_summary,
        poll_seconds=args.poll_seconds,
        latency_threads=args.latency_threads,
        latency_warmup=args.latency_warmup,
        latency_iters=args.latency_iters,
    )


if __name__ == "__main__":
    main()
