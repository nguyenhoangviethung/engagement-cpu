from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from engagement_daisee.common.config import CHECKPOINT_DIR, PROCESSED_LABELS_CSV, RANDOM_SEED
from engagement_daisee.openface.extract_openface709 import META_COLUMNS


LOGGER = logging.getLogger("reproduce_paper_cnn")
DEFAULT_MANIFEST = Path("data/processed/runs/openface709_binary_pca_features/openface709/feature_manifest.csv")
DEFAULT_OUTPUT_DIR = CHECKPOINT_DIR / "runs" / "paper_cnn_reproduction"
PAPER_FRAME_COUNT = 300
PAPER_FEATURE_DIM = 709
PAPER_REDUCED_DIM = 300
PAPER_NUM_CLASSES = 4


@dataclass(frozen=True)
class PaperCNNConfig:
    feature: str
    epochs: int
    batch_size: int
    learning_rate: float
    repeat: int = 0

    @property
    def run_name(self) -> str:
        lr_name = f"{self.learning_rate:.0e}".replace("-", "m")
        return f"{self.feature}_e{self.epochs}_b{self.batch_size}_lr{lr_name}_r{self.repeat:02d}"


class PaperCNN(nn.Module):
    def __init__(self, num_classes: int = PAPER_NUM_CLASSES):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in (32, 64, 128, 256):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 18 * 18, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    frame.columns = [str(column).strip() for column in frame.columns]
    columns = [column for column in frame.columns if column not in META_COLUMNS]
    if len(columns) != PAPER_FEATURE_DIM:
        raise ValueError(f"Expected {PAPER_FEATURE_DIM} OpenFace features, got {len(columns)}")
    return columns


def _select_highest_confidence_face(frame: pd.DataFrame) -> pd.DataFrame:
    if {"frame", "confidence"}.issubset(frame.columns):
        confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(-1.0)
        best_indices = confidence.groupby(frame["frame"]).idxmax()
        return frame.loc[best_indices].sort_values("frame")
    return frame


def _load_video_openface_matrix(rows: pd.DataFrame, use_csv: bool) -> np.ndarray:
    if use_csv and "openface_csv_path" in rows.columns:
        csv_path = Path(str(rows.iloc[0]["openface_csv_path"]))
        if csv_path.exists():
            frame = pd.read_csv(csv_path, skipinitialspace=True)
            frame.columns = [str(column).strip() for column in frame.columns]
            frame = _select_highest_confidence_face(frame)
            columns = _feature_columns(frame)
            matrix = (
                frame[columns]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
            return matrix

    segments: list[np.ndarray] = []
    for row in rows.sort_values("segment_index").itertuples(index=False):
        path = Path(str(getattr(row, "feature_path")))
        if path.exists():
            segments.append(np.load(path).astype(np.float32))
    if not segments:
        raise FileNotFoundError(f"No feature segments found for video_id={rows.iloc[0]['video_id']}")
    return np.concatenate(segments, axis=0)


def _sample_or_pad_frames(matrix: np.ndarray, frame_count: int) -> np.ndarray:
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if len(matrix) == 0:
        return np.zeros((frame_count, PAPER_FEATURE_DIM), dtype=np.float32)
    if len(matrix) > frame_count:
        indices = np.linspace(0, len(matrix) - 1, frame_count).round().astype(int)
        return matrix[indices]
    if len(matrix) < frame_count:
        padding = np.repeat(matrix[-1:], frame_count - len(matrix), axis=0)
        return np.concatenate([matrix, padding], axis=0)
    return matrix


def _load_multiclass_labels(labels_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    required = {"video_id", "engagement_raw"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Labels CSV is missing paper multiclass columns: {sorted(missing)}")
    labels = labels[["video_id", "engagement_raw"]].drop_duplicates("video_id").copy()
    labels["video_id"] = labels["video_id"].astype(str)
    labels["label"] = pd.to_numeric(labels["engagement_raw"], errors="raise").astype(int)
    labels = labels[labels["label"].between(0, PAPER_NUM_CLASSES - 1)]
    return labels[["video_id", "label"]]


def _select_paper_like_videos(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    per_class_limits: dict[int, int] | None,
    seed: int,
) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["video_id"] = manifest["video_id"].astype(str)
    merged = manifest.merge(labels, on="video_id", how="inner", suffixes=("", "_paper"))
    merged["label"] = merged["label_paper"].astype(int)
    if per_class_limits is None:
        return merged

    rng = np.random.default_rng(seed)
    selected_ids: set[str] = set()
    video_labels = merged[["video_id", "label"]].drop_duplicates("video_id")
    for label, limit in sorted(per_class_limits.items()):
        candidates = video_labels[video_labels["label"] == label]["video_id"].to_numpy()
        if len(candidates) > limit:
            candidates = rng.choice(candidates, size=limit, replace=False)
        selected_ids.update(str(value) for value in candidates)
    return merged[merged["video_id"].isin(selected_ids)].reset_index(drop=True)


def _load_video_dataset(
    manifest_path: Path,
    labels_csv: Path,
    use_csv: bool,
    paper_selection: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    manifest = pd.read_csv(manifest_path)
    labels = _load_multiclass_labels(labels_csv)
    limits = {0: 59, 1: 56, 2: 64, 3: 73} if paper_selection else None
    manifest = _select_paper_like_videos(manifest, labels, limits, seed=seed)
    if manifest.empty:
        raise RuntimeError("No manifest rows matched multiclass engagement labels.")

    matrices: list[np.ndarray] = []
    y: list[int] = []
    video_ids: list[str] = []
    grouped = manifest.groupby("video_id", sort=False)
    for index, (video_id, rows) in enumerate(grouped, start=1):
        matrix = _load_video_openface_matrix(rows, use_csv=use_csv)
        matrices.append(_sample_or_pad_frames(matrix, PAPER_FRAME_COUNT))
        y.append(int(rows.iloc[0]["label"]))
        video_ids.append(str(video_id))
        if index % 100 == 0:
            LOGGER.info("Loaded %d/%d videos", index, grouped.ngroups)

    x = np.stack(matrices, axis=0).astype(np.float32)
    labels_array = np.asarray(y, dtype=np.int64)
    return x, labels_array, video_ids


def _fit_reduce_normalize(x: np.ndarray, feature: str, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    flat_frames = x.reshape(-1, x.shape[-1])
    if feature == "pca":
        reducer = PCA(n_components=PAPER_REDUCED_DIM, svd_solver="randomized", random_state=seed)
    elif feature == "svd":
        reducer = TruncatedSVD(n_components=PAPER_REDUCED_DIM, random_state=seed)
    else:
        raise ValueError(f"Unsupported paper feature: {feature}")

    reduced = reducer.fit_transform(flat_frames).astype(np.float32)
    min_values = reduced.min(axis=0, keepdims=True)
    max_values = reduced.max(axis=0, keepdims=True)
    scale = np.maximum(max_values - min_values, 1e-6)
    normalized = ((reduced - min_values) / scale).astype(np.float32)
    images = normalized.reshape(x.shape[0], PAPER_FRAME_COUNT, PAPER_REDUCED_DIM)
    metadata = {
        "feature": feature,
        "components": PAPER_REDUCED_DIM,
        "explained_variance_sum": float(np.sum(getattr(reducer, "explained_variance_ratio_", np.array([])))),
        "normalization": "minmax_after_dimensional_reduction",
    }
    return images, metadata


def _smote_multiclass_flat(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    target_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    flat = x.reshape(x.shape[0], -1)
    counts = np.bincount(y, minlength=PAPER_NUM_CLASSES)
    target = int(target_count or counts.max())
    out_x = [flat]
    out_y = [y]
    for label, count in enumerate(counts):
        if count == 0 or count >= target:
            continue
        indices = np.flatnonzero(y == label)
        needed = target - int(count)
        anchors = rng.choice(indices, size=needed, replace=True)
        neighbors = rng.choice(indices, size=needed, replace=True)
        lam = rng.random((needed, 1), dtype=np.float32)
        synthetic = flat[anchors] + lam * (flat[neighbors] - flat[anchors])
        out_x.append(synthetic.astype(np.float32))
        out_y.append(np.full(needed, label, dtype=np.int64))
    balanced_x = np.concatenate(out_x, axis=0)
    balanced_y = np.concatenate(out_y, axis=0)
    order = rng.permutation(len(balanced_y))
    return balanced_x[order].reshape(-1, PAPER_FRAME_COUNT, PAPER_REDUCED_DIM), balanced_y[order]


def _paper_grid(features: list[str], repeats: int, best_only: bool, max_epochs: int | None) -> list[PaperCNNConfig]:
    configs: list[PaperCNNConfig] = []
    for feature in features:
        if best_only:
            batch_size = 4 if feature == "pca" else 8
            for repeat in range(repeats):
                epochs = min(1600, max_epochs) if max_epochs else 1600
                configs.append(PaperCNNConfig(feature, epochs, batch_size, 1e-4, repeat))
            continue
        for epochs in (800, 1600):
            epochs = min(epochs, max_epochs) if max_epochs else epochs
            for learning_rate in (1e-5, 1e-4):
                for batch_size in (32, 16, 8, 4, 2):
                    for repeat in range(repeats):
                        configs.append(PaperCNNConfig(feature, epochs, batch_size, learning_rate, repeat))
    return configs


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(PAPER_NUM_CLASSES))).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(PAPER_NUM_CLASSES)),
            output_dict=True,
            zero_division=0,
        ),
    }


def _train_one(
    x: np.ndarray,
    y: np.ndarray,
    config: PaperCNNConfig,
    output_dir: Path,
    device: torch.device,
    seed: int,
    early_stop: bool,
) -> dict[str, object]:
    _set_seed(seed + config.repeat)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed + config.repeat,
        stratify=y,
    )
    train_tensor = torch.from_numpy(x_train[:, None, :, :]).float()
    test_tensor = torch.from_numpy(x_test[:, None, :, :]).float()
    train_labels = torch.from_numpy(y_train).long()
    test_labels = torch.from_numpy(y_test).long()
    train_loader = DataLoader(
        TensorDataset(train_tensor, train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(TensorDataset(test_tensor, test_labels), batch_size=max(1, config.batch_size), shuffle=False)

    model = PaperCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    patience = max(1, config.epochs // 2)
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch_y)
        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        test_loss = 0.0
        predictions: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                test_loss += float(loss.item()) * len(batch_y)
                predictions.append(logits.argmax(dim=1).cpu().numpy())
                labels.append(batch_y.cpu().numpy())
        test_loss /= max(1, len(test_loader.dataset))
        y_pred_epoch = np.concatenate(predictions)
        y_true_epoch = np.concatenate(labels)
        accuracy = float(accuracy_score(y_true_epoch, y_pred_epoch))
        history.append({"epoch": float(epoch), "train_loss": train_loss, "test_loss": test_loss, "accuracy": accuracy})

        if test_loss < best_loss - 1e-8:
            best_loss = test_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        elif early_stop and epoch - best_epoch >= patience:
            LOGGER.info("Early stopping %s at epoch=%d best_epoch=%d", config.run_name, epoch, best_epoch)
            break

        if epoch == 1 or epoch % 50 == 0 or epoch == config.epochs:
            LOGGER.info(
                "%s epoch=%d/%d train_loss=%.4f test_loss=%.4f acc=%.4f",
                config.run_name,
                epoch,
                config.epochs,
                train_loss,
                test_loss,
                accuracy,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(batch_y.numpy())
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels)
    result = {
        "config": asdict(config),
        "run_name": config.run_name,
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "elapsed_sec": time.time() - started,
        "train_counts": np.bincount(y_train, minlength=PAPER_NUM_CLASSES).tolist(),
        "test_counts": np.bincount(y_test, minlength=PAPER_NUM_CLASSES).tolist(),
        "metrics": _metrics(y_true, y_pred),
    }

    run_dir = output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "paper_protocol": "Santoni et al. 2023 PCA/SVD-CNN reproduction",
            "input_shape": [1, PAPER_FRAME_COUNT, PAPER_REDUCED_DIM],
        },
        run_dir / "paper_cnn.pt",
    )
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "history.jsonl").write_text("\n".join(json.dumps(row) for row in history) + "\n", encoding="utf-8")
    return result


def run_reproduction(
    manifest_path: Path,
    labels_csv: Path,
    output_dir: Path,
    features: list[str],
    repeats: int,
    best_only: bool,
    paper_selection: bool,
    use_csv: bool,
    smote_target: int | None,
    device_name: str,
    seed: int,
    early_stop: bool,
    max_epochs: int | None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Loading OpenFace video matrices | manifest=%s labels=%s", manifest_path, labels_csv)
    raw_x, raw_y, video_ids = _load_video_dataset(
        manifest_path=manifest_path,
        labels_csv=labels_csv,
        use_csv=use_csv,
        paper_selection=paper_selection,
        seed=seed,
    )
    raw_counts = np.bincount(raw_y, minlength=PAPER_NUM_CLASSES).tolist()
    LOGGER.info("Loaded videos=%d raw_counts=%s device=%s", len(video_ids), raw_counts, device)

    all_results: list[dict[str, object]] = []
    feature_metadata: dict[str, object] = {}
    for feature in features:
        reduced_x, metadata = _fit_reduce_normalize(raw_x, feature, seed=seed)
        balanced_x, balanced_y = _smote_multiclass_flat(reduced_x, raw_y, seed=seed, target_count=smote_target)
        feature_metadata[feature] = {
            **metadata,
            "counts_before_smote": raw_counts,
            "counts_after_smote": np.bincount(balanced_y, minlength=PAPER_NUM_CLASSES).tolist(),
        }
        LOGGER.info("%s counts_after_smote=%s", feature, feature_metadata[feature]["counts_after_smote"])
        configs = [config for config in _paper_grid([feature], repeats=repeats, best_only=best_only, max_epochs=max_epochs)]
        for config in configs:
            result = _train_one(
                x=balanced_x,
                y=balanced_y,
                config=config,
                output_dir=output_dir,
                device=device,
                seed=seed,
                early_stop=early_stop,
            )
            all_results.append(result)
            with (output_dir / "all_results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")

    best_by_feature: dict[str, dict[str, object]] = {}
    for feature in features:
        feature_results = [result for result in all_results if result["config"]["feature"] == feature]
        if feature_results:
            best_by_feature[feature] = max(feature_results, key=lambda row: row["metrics"]["accuracy"])

    summary = {
        "protocol": "paper_cnn_reproduction",
        "paper": {
            "title": "Convolutional Neural Network Model based Students' Engagement Detection in Imbalanced DAiSEE Dataset",
            "target_results": {"pca_cnn_accuracy": 0.7288, "svd_cnn_accuracy": 0.7797},
            "notes": [
                "OpenFace 709 per-frame features",
                "PCA and SVD reduce 709 dimensions to 300",
                "Min-max normalization",
                "SMOTE before 80:20 train/test split",
                "CNN conv/pool filters 32,64,128,256 with kernel size 5 and max-pooling 2x2",
            ],
        },
        "manifest": str(manifest_path),
        "labels_csv": str(labels_csv),
        "videos": len(video_ids),
        "paper_selection": paper_selection,
        "use_openface_csv": use_csv,
        "features": feature_metadata,
        "runs": len(all_results),
        "best_by_feature": best_by_feature,
    }
    (output_dir / "paper_cnn_reproduction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Santoni et al. PCA-CNN and SVD-CNN DAiSEE paper pipeline.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=PROCESSED_LABELS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--features", nargs="+", choices=["pca", "svd"], default=["pca", "svd"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--best-only", action="store_true", help="Train only paper best configs: PCA b=4 and SVD b=8.")
    parser.add_argument("--paper-selection", action="store_true", help="Limit classes to paper Stage-2 counts: 59/56/64/73.")
    parser.add_argument("--no-csv", action="store_true", help="Use saved .npy segments instead of OpenFace CSV source rows.")
    parser.add_argument("--smote-target", type=int, default=73, help="Paper SMOTE target per class; use 0 for max class count.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=0, help="Debug cap for epochs; 0 keeps paper values.")
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    smote_target = None if args.smote_target <= 0 else args.smote_target
    summary = run_reproduction(
        manifest_path=args.manifest,
        labels_csv=args.labels,
        output_dir=args.output_dir,
        features=args.features,
        repeats=max(1, args.repeats),
        best_only=args.best_only,
        paper_selection=args.paper_selection,
        use_csv=not args.no_csv,
        smote_target=smote_target,
        device_name=args.device,
        seed=args.seed,
        early_stop=not args.no_early_stop,
        max_epochs=None if args.max_epochs <= 0 else args.max_epochs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
