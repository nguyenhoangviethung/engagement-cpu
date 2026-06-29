import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from engagement_daisee.common.config import FEATURE_MANIFEST_CSV, NUM_WORKERS
from engagement_daisee.common.manifest import normalize_manifest_columns
from engagement_daisee.rnn.dataset import FeatureSequenceDataset
from engagement_daisee.rnn.optimize_inference import _build_model_from_checkpoint


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _clip_probabilities(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(probabilities.astype(np.float64), eps, 1.0 - eps)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probs = _clip_probabilities(probabilities)
    return np.log(probs / (1.0 - probs))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _calibrate_probabilities(
    probabilities: np.ndarray,
    *,
    temperature: float = 1.0,
    source_pos_prior: float | None = None,
    target_pos_prior: float | None = None,
) -> np.ndarray:
    probs = _clip_probabilities(probabilities)
    temp = max(1e-3, float(temperature))
    calibrated = _sigmoid(_logit(probs) / temp)

    if source_pos_prior is None or target_pos_prior is None:
        return calibrated.astype(np.float32)

    source = float(np.clip(source_pos_prior, 1e-4, 1.0 - 1e-4))
    target = float(np.clip(target_pos_prior, 1e-4, 1.0 - 1e-4))
    if math.isclose(source, target, rel_tol=0.0, abs_tol=1e-8):
        return calibrated.astype(np.float32)

    source_odds = source / (1.0 - source)
    target_odds = target / (1.0 - target)
    odds_multiplier = target_odds / source_odds
    odds = calibrated / (1.0 - calibrated)
    adjusted_odds = odds * odds_multiplier
    adjusted = adjusted_odds / (1.0 + adjusted_odds)
    return _clip_probabilities(adjusted).astype(np.float32)


def _compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    precision_pos = _safe_div(tp, tp + fp)
    recall_pos = _safe_div(tp, tp + fn)
    f1_pos = _safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    precision_neg = _safe_div(tn, tn + fn)
    recall_neg = _safe_div(tn, tn + fp)
    f1_neg = _safe_div(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (recall_pos + recall_neg) / 2.0
    f1_macro = (f1_pos + f1_neg) / 2.0

    return {
        "accuracy": accuracy,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "f1_neg": f1_neg,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": f1_macro,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _split_indices(manifest: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    manifest = normalize_manifest_columns(manifest)
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


def _build_loader(dataset: FeatureSequenceDataset, indices: list[int], batch_size: int) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )


def _aggregate_by_video(manifest_subset: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "video_id" not in manifest_subset.columns:
        return labels, probabilities

    frame = manifest_subset[["video_id"]].copy().reset_index(drop=True)
    frame["label"] = labels.astype(np.int64)
    frame["probability"] = probabilities.astype(np.float32)
    grouped = frame.groupby("video_id", sort=False).agg({"label": "max", "probability": "mean"})
    return grouped["label"].to_numpy(dtype=np.int64), grouped["probability"].to_numpy(dtype=np.float32)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    threshold: float,
    feature_mean: torch.Tensor | None,
    feature_std: torch.Tensor | None,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    for features, labels in loader:
        features = features.to("cpu")
        labels = labels.to("cpu")
        if feature_mean is not None and feature_std is not None:
            features = (features - feature_mean) / feature_std

        logits = model(features)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs_np = np.concatenate(all_probs).astype(np.float32)
    labels_np = np.concatenate(all_labels).astype(np.int64)
    metrics = _compute_binary_metrics(labels_np, probs_np, threshold=threshold)
    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, metrics, labels_np, probs_np


def run_eval(
    manifest_path: Path,
    checkpoint_path: Path,
    split: str,
    batch_size: int,
    threshold: float | None,
    output_json: Path | None,
    aggregation: str,
) -> dict:
    dataset = FeatureSequenceDataset(manifest_path)
    manifest_df = dataset.manifest

    train_indices, val_indices, test_indices = _split_indices(manifest_df)
    split_name = split.strip().lower()
    split_map = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
    }
    if split_name not in split_map:
        raise ValueError(f"Unsupported split: {split}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, model_kwargs = _build_model_from_checkpoint(checkpoint)
    model = model.to("cpu").eval()

    ckpt_threshold = float(checkpoint.get("best_threshold", 0.5))
    resolved_threshold = float(threshold) if threshold is not None else ckpt_threshold
    best_temperature = float(checkpoint.get("best_temperature", 1.0))

    feature_mean = None
    feature_std = None
    if checkpoint.get("normalize_features", False):
        mean_raw = checkpoint.get("feature_mean")
        std_raw = checkpoint.get("feature_std")
        if mean_raw is not None and std_raw is not None:
            feature_mean = torch.tensor(mean_raw, dtype=torch.float32).view(1, 1, -1)
            feature_std = torch.tensor(std_raw, dtype=torch.float32).view(1, 1, -1)

    indices = split_map[split_name]
    loader = _build_loader(dataset, indices, batch_size=batch_size)
    loss, row_metrics, labels, probabilities = _evaluate(
        model=model,
        loader=loader,
        threshold=resolved_threshold,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    calibration_cfg = checkpoint.get("prior_shift_calibration", {})
    calibration_enabled = bool(calibration_cfg.get("enabled", False))
    if calibration_enabled:
        probabilities = _calibrate_probabilities(
            probabilities,
            temperature=best_temperature,
            source_pos_prior=calibration_cfg.get("source_pos_prior"),
            target_pos_prior=calibration_cfg.get("target_pos_prior"),
        )
        row_metrics = _compute_binary_metrics(labels, probabilities, threshold=resolved_threshold)
    manifest_subset = manifest_df.iloc[indices].reset_index(drop=True)
    video_labels, video_probabilities = _aggregate_by_video(manifest_subset, labels, probabilities)
    video_metrics = _compute_binary_metrics(video_labels, video_probabilities, threshold=resolved_threshold)
    selected_metrics = video_metrics if aggregation == "video" else row_metrics

    report = {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "rows": int(len(indices)),
        "videos": int(len(video_labels)),
        "batch_size": int(batch_size),
        "threshold": float(resolved_threshold),
        "aggregation": aggregation,
        "loss": float(loss),
        "metrics": selected_metrics,
        "row_metrics": row_metrics,
        "video_metrics": video_metrics,
        "model_kwargs": model_kwargs,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on DAiSEE official split.")
    parser.add_argument("--manifest", type=Path, default=FEATURE_MANIFEST_CSV, help="Feature manifest CSV")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path (.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], help="Split to evaluate")
    parser.add_argument("--batch-size", type=int, default=128, help="Eval batch size")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional report output json")
    parser.add_argument("--aggregation", type=str, default="rows", choices=["rows", "video"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        threshold=args.threshold,
        output_json=args.output_json,
        aggregation=args.aggregation,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
