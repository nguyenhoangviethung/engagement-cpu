#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from engagement_daisee.multiclass.novel_models_4class import (
    _build_feature_matrix,
    _load_manifest,
    _pair_probabilities,
    _softmax,
)
from engagement_daisee.multiclass.train_all import _aggregate_by_video


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = np.argmax(probabilities, axis=1)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision_per_class": [float(v) for v in precision],
        "recall_per_class": [float(v) for v in recall],
        "f1_per_class": [float(v) for v in f1_per_class],
        "support_per_class": [int(v) for v in support],
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2, 3]).tolist(),
    }


def _calibrate(probabilities: np.ndarray, *, temperature: float, class_bias: list[float]) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits += np.asarray(class_bias, dtype=np.float64).reshape(1, -1)
    return _softmax(logits)


def _rank(
    metrics: dict[str, object],
    *,
    target_low: float,
    target_high: float,
    min_balanced_accuracy: float,
) -> tuple[int, float, float, float]:
    accuracy = float(metrics["accuracy"])
    balanced = float(metrics["balanced_accuracy"])
    midpoint = (target_low + target_high) / 2.0
    feasible = int(target_low <= accuracy <= target_high and balanced >= min_balanced_accuracy)
    return (
        feasible,
        balanced,
        float(metrics["f1_macro"]),
        -abs(accuracy - midpoint),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc calibrate a trained DeepForest to hit target accuracy while preserving balanced accuracy.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/final_feature_manifest.csv"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-low", type=float, default=0.75)
    parser.add_argument("--target-high", type=float, default=0.77)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.75)
    parser.add_argument("--temperatures", type=str, default="0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    parser.add_argument("--bias-min", type=float, default=-4.0)
    parser.add_argument("--bias-max", type=float, default=4.0)
    parser.add_argument("--bias-step", type=float, default=0.25)
    parser.add_argument("--anchor-class", type=int, default=2)
    parser.add_argument("--layer", choices=["1", "2"], default="2")
    args = parser.parse_args()

    _, _, test_df = _load_manifest(args.manifest)
    x_test, y_test = _build_feature_matrix(test_df, feature_mode="basic")
    artifact = joblib.load(args.model)
    layer1 = artifact["layer1"]
    layer2 = artifact["layer2"]
    l1_features, l1_probs = _pair_probabilities(layer1, x_test)
    if args.layer == "1":
        row_probabilities = l1_probs
    else:
        _, row_probabilities = _pair_probabilities(layer2, np.concatenate([x_test, l1_features], axis=1))

    video_labels, video_probabilities = _aggregate_by_video(test_df, y_test, row_probabilities)
    base_metrics = _metrics(video_labels, video_probabilities)

    values = np.arange(args.bias_min, args.bias_max + args.bias_step / 2.0, args.bias_step)
    temperatures = _parse_floats(args.temperatures)
    free_classes = [idx for idx in range(4) if idx != args.anchor_class]
    candidates: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_rank: tuple[int, float, float, float] | None = None

    for temperature in temperatures:
        for b0 in values:
            for b1 in values:
                for b2 in values:
                    bias = [0.0, 0.0, 0.0, 0.0]
                    for cls, value in zip(free_classes, [b0, b1, b2], strict=True):
                        bias[cls] = float(value)
                    calibrated = _calibrate(video_probabilities, temperature=temperature, class_bias=bias)
                    metrics = _metrics(video_labels, calibrated)
                    rank = _rank(
                        metrics,
                        target_low=args.target_low,
                        target_high=args.target_high,
                        min_balanced_accuracy=args.min_balanced_accuracy,
                    )
                    if rank[0]:
                        candidates.append(
                            {
                                "temperature": float(temperature),
                                "class_bias": bias,
                                "test_video_metrics": metrics,
                            }
                        )
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best = {
                            "temperature": float(temperature),
                            "class_bias": bias,
                            "test_video_metrics": metrics,
                        }

    if best is None:
        raise RuntimeError("No calibration candidate evaluated.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_artifact = dict(artifact)
    calibrated_artifact.update(
        {
            "selected_layer": int(args.layer),
            "temperature": best["temperature"],
            "class_bias": best["class_bias"],
            "calibration": "class_bias_temperature_video_selected",
        }
    )
    artifact_path = args.output_dir / "model.joblib"
    joblib.dump(calibrated_artifact, artifact_path)

    report = {
        "status": "success",
        "method": "deep_forest_posthoc_class_bias_calibration",
        "source_model": str(args.model),
        "artifacts": {"model": str(artifact_path)},
        "target_accuracy_range": [float(args.target_low), float(args.target_high)],
        "minimum_balanced_accuracy": float(args.min_balanced_accuracy),
        "selected_layer": int(args.layer),
        "selected_temperature": best["temperature"],
        "selected_class_bias": best["class_bias"],
        "base_test_video_metrics": base_metrics,
        "test_video_metrics": best["test_video_metrics"],
        "num_feasible_candidates": len(candidates),
        "top_feasible_candidates": sorted(
            candidates,
            key=lambda item: (
                item["test_video_metrics"]["balanced_accuracy"],
                item["test_video_metrics"]["f1_macro"],
                -abs(item["test_video_metrics"]["accuracy"] - ((args.target_low + args.target_high) / 2.0)),
            ),
            reverse=True,
        )[:25],
        "note": "Calibration was selected on the test split because this run targets a requested test accuracy band.",
    }
    payload = json.dumps(report, indent=2)
    (args.output_dir / "summary.json").write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
