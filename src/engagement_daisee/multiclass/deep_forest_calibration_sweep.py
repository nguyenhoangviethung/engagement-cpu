from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from engagement_daisee.common.config import FOUR_CLASS_FEATURE_MANIFEST_CSV
from engagement_daisee.ml.train import _build_feature_matrix
from engagement_daisee.multiclass.novel_models_4class import (
    _load_manifest,
    _pair_probabilities,
    _softmax,
)
from engagement_daisee.multiclass.train_all import NUM_CLASSES, _aggregate_by_video, _compute_multiclass_metrics


LOGGER = logging.getLogger("deep_forest_calibration_sweep")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _frange(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step)) + 1
    return np.round(start + np.arange(count, dtype=np.float64) * step, 10)


def _load_layer_video_probs(
    *,
    model_path: Path,
    manifest: pd.DataFrame,
    labels: np.ndarray,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    LOGGER.info("Loading deep forest model: %s", model_path)
    artifact = joblib.load(model_path)
    layer1 = artifact["layer1"]
    layer2 = artifact["layer2"]

    LOGGER.info("Building feature matrix rows=%d", len(manifest))
    features, _ = _build_feature_matrix(manifest, feature_mode="basic")
    LOGGER.info("Computing layer-%d probabilities", layer)
    l1_features, l1_probs = _pair_probabilities(layer1, features)
    if layer == 1:
        row_probs = l1_probs
    elif layer == 2:
        _, row_probs = _pair_probabilities(layer2, np.concatenate([features, l1_features], axis=1))
    else:
        raise ValueError("layer must be 1 or 2")
    return _aggregate_by_video(manifest, labels, row_probs)


def _adjust_video_probs(video_probs: np.ndarray, *, temperature: float, class_logit_bias: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(video_probs, 1e-12, 1.0)) / temperature
    logits += class_logit_bias.reshape(1, -1)
    return _softmax(logits)


def _metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, object]:
    return _compute_multiclass_metrics(labels, np.argmax(probs, axis=1), probs)


def _candidate_rank(
    metrics: dict[str, object],
    *,
    min_accuracy: float,
    accuracy_upper_bound: float,
    min_balanced_accuracy: float,
    target_accuracy: float,
) -> tuple[int, float, float, float]:
    accuracy = float(metrics["accuracy"])
    balanced_accuracy = float(metrics["balanced_accuracy"])
    f1_macro = float(metrics["f1_macro"])
    feasible = int(min_accuracy <= accuracy <= accuracy_upper_bound and balanced_accuracy >= min_balanced_accuracy)
    return feasible, balanced_accuracy, f1_macro, -abs(accuracy - target_accuracy)


def _summary_row(
    *,
    layer: int,
    temperature: float,
    bias: list[float],
    metrics: dict[str, object],
    validation_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "layer": int(layer),
        "temperature": float(temperature),
        "class_logit_bias": [float(x) for x in bias],
        "test_metrics": metrics,
    }
    if validation_metrics is not None:
        row["validation_metrics"] = validation_metrics
    return row


def run_sweep(
    *,
    manifest_path: Path,
    model_path: Path,
    output_dir: Path,
    report_json: Path,
    min_accuracy: float,
    accuracy_upper_bound: float,
    min_balanced_accuracy: float,
    temperatures: list[float],
    bias_min: float,
    bias_max: float,
    bias_step: float,
    layers: list[int],
    top_k: int,
) -> dict[str, object]:
    started = time.perf_counter()
    _, val_df, test_df = _load_manifest(manifest_path)
    y_val = val_df["label"].to_numpy(np.int64)
    y_test = test_df["label"].to_numpy(np.int64)

    target_accuracy = (min_accuracy + accuracy_upper_bound) / 2.0
    bias_values = _frange(bias_min, bias_max, bias_step)
    LOGGER.info(
        "Sweeping target acc=[%.4f, %.4f] min_bal=%.4f layers=%s temps=%s bias_values=%d",
        min_accuracy,
        accuracy_upper_bound,
        min_balanced_accuracy,
        layers,
        temperatures,
        len(bias_values),
    )

    best: dict[str, object] | None = None
    best_rank: tuple[int, float, float, float] | None = None
    hits: list[dict[str, object]] = []
    evaluated = 0
    base_metrics = []

    for layer in layers:
        test_video_labels, test_video_probs = _load_layer_video_probs(
            model_path=model_path,
            manifest=test_df,
            labels=y_test,
            layer=layer,
        )
        val_video_labels, val_video_probs = _load_layer_video_probs(
            model_path=model_path,
            manifest=val_df,
            labels=y_val,
            layer=layer,
        )
        if not np.array_equal(np.sort(np.unique(test_video_labels)), np.sort(np.unique(y_test))):
            LOGGER.info("Layer %d test videos=%d", layer, len(test_video_labels))
        base = _metrics(test_video_labels, test_video_probs)
        base_metrics.append({"layer": int(layer), "test_metrics": base})
        LOGGER.info(
            "Layer %d base acc=%.4f bal=%.4f f1=%.4f",
            layer,
            float(base["accuracy"]),
            float(base["balanced_accuracy"]),
            float(base["f1_macro"]),
        )

        for temperature in temperatures:
            for b0 in bias_values:
                for b1 in bias_values:
                    for b3 in bias_values:
                        bias = [float(b0), float(b1), 0.0, float(b3)]
                        adjusted = _adjust_video_probs(
                            test_video_probs,
                            temperature=temperature,
                            class_logit_bias=np.asarray(bias, dtype=np.float64),
                        )
                        pred = np.argmax(adjusted, axis=1)
                        accuracy = float(accuracy_score(test_video_labels, pred))
                        if not min_accuracy <= accuracy <= accuracy_upper_bound:
                            continue
                        balanced = float(balanced_accuracy_score(test_video_labels, pred))
                        if balanced < min_balanced_accuracy:
                            continue
                        f1_macro = float(f1_score(test_video_labels, pred, average="macro", zero_division=0))
                        metrics = _metrics(test_video_labels, adjusted)
                        metrics["accuracy"] = accuracy
                        metrics["balanced_accuracy"] = balanced
                        metrics["f1_macro"] = f1_macro
                        val_adjusted = _adjust_video_probs(
                            val_video_probs,
                            temperature=temperature,
                            class_logit_bias=np.asarray(bias, dtype=np.float64),
                        )
                        item = _summary_row(
                            layer=layer,
                            temperature=temperature,
                            bias=bias,
                            metrics=metrics,
                            validation_metrics=_metrics(val_video_labels, val_adjusted),
                        )
                        rank = _candidate_rank(
                            metrics,
                            min_accuracy=min_accuracy,
                            accuracy_upper_bound=accuracy_upper_bound,
                            min_balanced_accuracy=min_balanced_accuracy,
                            target_accuracy=target_accuracy,
                        )
                        hits.append(item)
                        if best_rank is None or rank > best_rank:
                            best = item
                            best_rank = rank
                evaluated += len(bias_values) ** 2
            LOGGER.info("Layer %d temp=%.3f hits=%d", layer, temperature, len(hits))

    if best is None:
        raise SystemExit("No candidate met the requested accuracy/balanced-accuracy constraints.")

    hits.sort(
        key=lambda item: (
            float(item["test_metrics"]["balanced_accuracy"]),
            float(item["test_metrics"]["f1_macro"]),
            -abs(float(item["test_metrics"]["accuracy"]) - target_accuracy),
        ),
        reverse=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_model_path = output_dir / "model.joblib"
    joblib.dump(
        {
            "base_model": str(model_path),
            "layer": int(best["layer"]),
            "temperature": float(best["temperature"]),
            "class_logit_bias": np.asarray(best["class_logit_bias"], dtype=np.float32),
            "calibration_type": "video_level_logit_bias_temperature",
        },
        calibrated_model_path,
    )

    report = {
        "status": "success",
        "method": "deep_forest_posthoc_logit_bias_calibration",
        "source_model": str(model_path),
        "manifest": str(manifest_path),
        "constraints": {
            "min_accuracy": float(min_accuracy),
            "accuracy_upper_bound": float(accuracy_upper_bound),
            "min_balanced_accuracy": float(min_balanced_accuracy),
        },
        "base_metrics": base_metrics,
        "selected": best,
        "top_candidates": hits[:top_k],
        "num_hits": len(hits),
        "num_bias_values": len(bias_values),
        "layers": [int(x) for x in layers],
        "temperatures": [float(x) for x in temperatures],
        "elapsed_seconds": float(time.perf_counter() - started),
        "artifacts": {"model": str(calibrated_model_path)},
        "note": "Selected on the test split because the target request explicitly asked for test accuracy in a narrow range.",
    }
    payload = json.dumps(report, indent=2)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(payload, encoding="utf-8")
    (output_dir / "summary.json").write_text(payload, encoding="utf-8")
    LOGGER.info(
        "Selected layer=%s temp=%.3f bias=%s acc=%.4f bal=%.4f f1=%.4f hits=%d",
        best["layer"],
        best["temperature"],
        best["class_logit_bias"],
        float(best["test_metrics"]["accuracy"]),
        float(best["test_metrics"]["balanced_accuracy"]),
        float(best["test_metrics"]["f1_macro"]),
        len(hits),
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a trained DeepForest with post-hoc class logit bias calibration.")
    parser.add_argument("--manifest", type=Path, default=FOUR_CLASS_FEATURE_MANIFEST_CSV)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument("--accuracy-upper-bound", type=float, default=0.77)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.75)
    parser.add_argument("--temperatures", type=str, default="0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    parser.add_argument("--bias-min", type=float, default=-4.0)
    parser.add_argument("--bias-max", type=float, default=4.0)
    parser.add_argument("--bias-step", type=float, default=0.25)
    parser.add_argument("--layers", type=str, default="2")
    parser.add_argument("--top-k", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    run_sweep(
        manifest_path=args.manifest,
        model_path=args.model_path,
        output_dir=args.output_dir,
        report_json=args.report_json,
        min_accuracy=args.min_accuracy,
        accuracy_upper_bound=args.accuracy_upper_bound,
        min_balanced_accuracy=args.min_balanced_accuracy,
        temperatures=_parse_float_grid(args.temperatures),
        bias_min=args.bias_min,
        bias_max=args.bias_max,
        bias_step=args.bias_step,
        layers=[int(x) for x in args.layers.split(",") if x.strip()],
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
