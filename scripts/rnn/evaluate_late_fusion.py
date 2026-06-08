#!/usr/bin/env python
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from engagement_daisee.common.manifest import normalize_manifest_columns
from engagement_daisee.ml.evaluate import _load_model, _predict_probabilities
from engagement_daisee.ml.train import (
    _apply_feature_preprocessor,
    _build_feature_matrix,
    _load_feature_preprocessor,
    _load_manifest,
)
from engagement_daisee.rnn.dataset import FeatureSequenceDataset
from engagement_daisee.rnn.evaluate import (
    _aggregate_by_video,
    _calibrate_probabilities,
    _compute_binary_metrics,
    _split_indices,
)
from engagement_daisee.rnn.optimize_inference import _build_model_from_checkpoint


def _load_rnn_video_scores(manifest: Path, checkpoint_path: Path, split: str, batch_size: int) -> dict[str, dict]:
    dataset = FeatureSequenceDataset(manifest)
    manifest_df = dataset.manifest
    _, val_indices, test_indices = _split_indices(manifest_df)
    indices = val_indices if split == "validation" else test_indices
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, _ = _build_model_from_checkpoint(checkpoint)
    model.eval()

    feature_mean = feature_std = None
    if checkpoint.get("normalize_features", False):
        feature_mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32).view(1, 1, -1)
        feature_std = torch.tensor(checkpoint["feature_std"], dtype=torch.float32).view(1, 1, -1)

    labels_all = []
    probs_all = []
    with torch.no_grad():
        for features, labels in loader:
            if feature_mean is not None and feature_std is not None:
                features = (features - feature_mean) / feature_std
            probs = torch.sigmoid(model(features)).cpu().numpy()
            probs_all.append(probs.astype(np.float32))
            labels_all.append(labels.cpu().numpy().astype(np.int64))

    labels = np.concatenate(labels_all)
    probs = np.concatenate(probs_all)
    calibration_cfg = checkpoint.get("prior_shift_calibration", {})
    if calibration_cfg.get("enabled", False):
        probs = _calibrate_probabilities(
            probs,
            temperature=float(checkpoint.get("best_temperature", 1.0)),
            source_pos_prior=calibration_cfg.get("source_pos_prior"),
            target_pos_prior=calibration_cfg.get("target_pos_prior"),
        )

    subset_df = manifest_df.iloc[indices].reset_index(drop=True)
    video_labels, video_probs = _aggregate_by_video(subset_df, labels, probs)
    video_ids = (
        subset_df[["video_id"]]
        .assign(label=labels.astype(np.int64), probability=probs.astype(np.float32))
        .groupby("video_id", sort=False)
        .agg({"label": "max", "probability": "mean"})
        .index.astype(str)
        .tolist()
    )
    return {
        vid: {"label": int(label), "probability": float(prob)}
        for vid, label, prob in zip(video_ids, video_labels, video_probs, strict=True)
    }


def _load_ml_video_scores(
    manifest: Path,
    model_path: Path,
    summary_json: Path,
    split: str,
    feature_mode: str,
) -> dict[str, dict]:
    manifest_df = normalize_manifest_columns(_load_manifest(manifest))
    split_df = manifest_df[manifest_df["split"].astype(str).str.lower() == split].reset_index(drop=True)
    x_eval, y_eval = _build_feature_matrix(split_df, feature_mode=feature_mode)

    summary = json.loads(summary_json.read_text())
    preprocessor_path = Path(summary["preprocessor_path"])
    if preprocessor_path.exists():
        x_eval = _apply_feature_preprocessor(x_eval, _load_feature_preprocessor(preprocessor_path))

    backend, model = _load_model(model_path)
    probs = _predict_probabilities(backend, model, x_eval)
    video_labels, video_probs = _aggregate_by_video(split_df, y_eval, probs)
    video_ids = (
        split_df[["video_id"]]
        .assign(label=y_eval.astype(np.int64), probability=probs.astype(np.float32))
        .groupby("video_id", sort=False)
        .agg({"label": "max", "probability": "mean"})
        .index.astype(str)
        .tolist()
    )
    return {
        vid: {"label": int(label), "probability": float(prob)}
        for vid, label, prob in zip(video_ids, video_labels, video_probs, strict=True)
    }


def _align(scores_by_model: dict[str, dict[str, dict]]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    common_ids = sorted(set.intersection(*(set(scores) for scores in scores_by_model.values())))
    labels = np.array([next(iter(scores_by_model.values()))[vid]["label"] for vid in common_ids], dtype=np.int64)
    probabilities = {
        name: np.array([scores[vid]["probability"] for vid in common_ids], dtype=np.float32)
        for name, scores in scores_by_model.items()
    }
    return common_ids, labels, probabilities


def _weight_grid(n_models: int, step: float) -> list[np.ndarray]:
    ticks = int(round(1.0 / step))
    weights = []
    for raw in product(range(ticks + 1), repeat=n_models):
        if sum(raw) == ticks:
            weights.append(np.array(raw, dtype=np.float32) / ticks)
    return weights


def _fuse(probs_by_model: dict[str, np.ndarray], names: list[str], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack([probs_by_model[name] for name in names], axis=0)
    return np.sum(stacked * weights[:, None], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and evaluate video-level late fusion on validation/test splits.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gru", type=Path, required=True)
    parser.add_argument("--tcn", type=Path, required=True)
    parser.add_argument("--xgb", type=Path, required=True)
    parser.add_argument("--xgb-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--feature-mode", default="tsfresh")
    args = parser.parse_args()

    model_specs = {
        "gru": ("rnn", args.gru),
        "tcn": ("rnn", args.tcn),
        "xgboost": ("ml", args.xgb),
    }
    scores = {"validation": {}, "test": {}}
    for split in scores:
        for name, (kind, path) in model_specs.items():
            if kind == "rnn":
                scores[split][name] = _load_rnn_video_scores(args.manifest, path, split, args.batch_size)
            else:
                scores[split][name] = _load_ml_video_scores(
                    args.manifest, path, args.xgb_summary, split, args.feature_mode
                )

    names = list(model_specs)
    val_ids, val_labels, val_probs = _align(scores["validation"])
    test_ids, test_labels, test_probs = _align(scores["test"])

    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 3)
    best = None
    for weights in _weight_grid(len(names), args.weight_step):
        fused = _fuse(val_probs, names, weights)
        for threshold in thresholds:
            metrics = _compute_binary_metrics(val_labels, fused, float(threshold))
            score = metrics["balanced_accuracy"]
            if best is None or score > best["validation_metrics"]["balanced_accuracy"]:
                best = {
                    "weights": {name: float(weight) for name, weight in zip(names, weights, strict=True)},
                    "threshold": float(threshold),
                    "validation_metrics": metrics,
                }

    assert best is not None
    weight_vec = np.array([best["weights"][name] for name in names], dtype=np.float32)
    test_fused = _fuse(test_probs, names, weight_vec)
    test_metrics = _compute_binary_metrics(test_labels, test_fused, best["threshold"])
    baselines = {}
    for split, labels, probs in (
        ("validation", val_labels, val_probs),
        ("test", test_labels, test_probs),
    ):
        baselines[split] = {
            name: _compute_binary_metrics(labels, values, best["threshold"])
            for name, values in probs.items()
        }

    report = {
        "manifest": str(args.manifest),
        "models": {name: str(path) for name, (_, path) in model_specs.items()},
        "xgb_summary": str(args.xgb_summary),
        "video_counts": {"validation": len(val_ids), "test": len(test_ids)},
        "selected": best,
        "test_metrics": test_metrics,
        "baselines_at_selected_threshold": baselines,
        "note": "Weights and threshold selected on validation only; test is held out for final reporting.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
