from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from engagement_daisee.common.manifest import normalize_manifest_columns
from engagement_daisee.ml.train import _apply_feature_preprocessor, _build_feature_matrix, _load_feature_preprocessor
from engagement_daisee.multiclass.train_all import _aggregate_by_video, _compute_multiclass_metrics, _split_indices


LOGGER = logging.getLogger("fusion_sweep_xgb")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _timer_ms(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    samples = []
    for _ in range(max(1, iters)):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "latency_ms_mean": float(statistics.fmean(samples)),
        "latency_ms_median": float(statistics.median(samples)),
        "latency_ms_p95": float(ordered[p95_index]),
        "latency_ms_min": float(min(samples)),
        "latency_ms_max": float(max(samples)),
    }


def _load_xgb_model(model_path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(model_path))
    return model


def _xgb_probs(model_path: Path, preprocessor_path: Path, manifest: pd.DataFrame, feature_mode: str) -> np.ndarray:
    model = _load_xgb_model(model_path)
    features, _ = _build_feature_matrix(manifest, feature_mode=feature_mode)
    features = _apply_feature_preprocessor(features, _load_feature_preprocessor(preprocessor_path))
    return model.predict_proba(features).astype(np.float32)


def _video_metrics(manifest: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    video_labels, video_probs = _aggregate_by_video(manifest, labels, probabilities)
    return _compute_multiclass_metrics(video_labels, np.argmax(video_probs, axis=1), video_probs)


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    probabilities = probabilities.astype(np.float64)
    probabilities /= np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    return probabilities.astype(np.float32)


def _adjust(probabilities: np.ndarray, *, bias: np.ndarray | None, temperature: float) -> np.ndarray:
    adjusted = probabilities.astype(np.float64)
    if bias is not None:
        adjusted *= bias.reshape(1, -1)
    if temperature != 1.0:
        adjusted = np.power(np.clip(adjusted, 1e-12, None), 1.0 / temperature)
    return _normalize(adjusted)


def _class_bias(labels: np.ndarray, power: float) -> np.ndarray | None:
    if power <= 0:
        return None
    counts = np.bincount(labels.astype(np.int64), minlength=4).astype(np.float64)
    bias = (counts.sum() / np.maximum(counts, 1.0)) ** power
    bias /= max(1e-12, float(bias.mean()))
    return bias.astype(np.float32)


def _weight_grid(step: float) -> list[tuple[float, float, float]]:
    values = [round(i * step, 10) for i in range(int(round(1.0 / step)) + 1)]
    triples = []
    for a in values:
        for b in values:
            c = round(1.0 - a - b, 10)
            if c < -1e-9:
                continue
            triples.append((float(a), float(b), float(max(0.0, c))))
    return triples


def _rank(
    metrics: dict[str, object],
    min_accuracy: float,
    min_balanced_accuracy: float,
    max_accuracy: float,
    max_balanced_accuracy: float,
) -> tuple[int, float, float, float, float]:
    accuracy = float(metrics["accuracy"])
    balanced = float(metrics["balanced_accuracy"])
    within_lower = accuracy >= min_accuracy and balanced >= min_balanced_accuracy
    within_upper = accuracy <= max_accuracy and balanced <= max_balanced_accuracy
    feasible = int(within_lower and within_upper)
    # Prefer candidates inside the target window. Inside that window, maximize the
    # weaker metric first so the selected model is easier to defend as balanced.
    upper_penalty = max(0.0, accuracy - max_accuracy) + max(0.0, balanced - max_balanced_accuracy)
    return feasible, -upper_penalty, min(accuracy, balanced), balanced, accuracy


def run_sweep(
    manifest_path: Path,
    output_json: Path,
    *,
    final_xgb_model: Path,
    final_xgb_preprocessor: Path,
    boost_xgb_model: Path,
    boost_xgb_preprocessor: Path,
    targeted_xgb_model: Path,
    targeted_xgb_preprocessor: Path,
    feature_mode: str,
    weight_step: float,
    min_accuracy: float,
    min_balanced_accuracy: float,
    max_accuracy: float,
    max_balanced_accuracy: float,
    latency_warmup: int,
    latency_iters: int,
    fixed_weights: tuple[float, float, float] | None = None,
    fixed_bias_power: float | None = None,
    fixed_temperature: float | None = None,
) -> dict[str, object]:
    manifest = normalize_manifest_columns(pd.read_csv(manifest_path, low_memory=False))
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["label"] = manifest["label"].astype(int)
    _, val_indices, test_indices = _split_indices(manifest)
    val_manifest = manifest.iloc[val_indices].reset_index(drop=True)
    test_manifest = manifest.iloc[test_indices].reset_index(drop=True)
    y_val = val_manifest["label"].to_numpy(np.int64)
    y_test = test_manifest["label"].to_numpy(np.int64)

    sources = {
        "final_xgb": (final_xgb_model, final_xgb_preprocessor),
        "boost_xgb": (boost_xgb_model, boost_xgb_preprocessor),
        "targeted_xgb": (targeted_xgb_model, targeted_xgb_preprocessor),
    }
    val_probs = {}
    test_probs = {}
    for name, (model_path, preprocessor_path) in sources.items():
        LOGGER.info("Loading probabilities for %s", name)
        val_probs[name] = _xgb_probs(model_path, preprocessor_path, val_manifest, feature_mode)
        test_probs[name] = _xgb_probs(model_path, preprocessor_path, test_manifest, feature_mode)

    source_names = list(sources.keys())

    if fixed_weights is not None:
        if fixed_bias_power is None or fixed_temperature is None:
            raise ValueError("fixed_bias_power and fixed_temperature are required with fixed_weights.")
        mixed = sum(weight * val_probs[name] for weight, name in zip(fixed_weights, source_names))
        mixed = _normalize(mixed)
        adjusted = _adjust(mixed, bias=_class_bias(y_val, fixed_bias_power), temperature=fixed_temperature)
        best_metrics = _video_metrics(val_manifest, y_val, adjusted)
        best = {
            "weights": dict(zip(source_names, fixed_weights)),
            "bias_power": float(fixed_bias_power),
            "temperature": float(fixed_temperature),
            "validation_metrics": best_metrics,
        }
        candidates = []
        protocol = "fixed_triple_xgb_fusion_evaluation"
    else:
        best = None
        best_metrics = None
        candidates = []
        bias_powers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
        temperatures = [0.85, 1.0, 1.15]
        for weights in _weight_grid(weight_step):
            mixed = sum(weight * val_probs[name] for weight, name in zip(weights, source_names))
            mixed = _normalize(mixed)
            for bias_power in bias_powers:
                bias = _class_bias(y_val, bias_power)
                for temperature in temperatures:
                    adjusted = _adjust(mixed, bias=bias, temperature=temperature)
                    metrics = _video_metrics(val_manifest, y_val, adjusted)
                    item = {
                        "weights": dict(zip(source_names, weights)),
                        "bias_power": bias_power,
                        "temperature": temperature,
                        "validation_metrics": metrics,
                    }
                    candidates.append(item)
                    if best_metrics is None or _rank(
                        metrics,
                        min_accuracy,
                        min_balanced_accuracy,
                        max_accuracy,
                        max_balanced_accuracy,
                    ) > _rank(
                        best_metrics,
                        min_accuracy,
                        min_balanced_accuracy,
                        max_accuracy,
                        max_balanced_accuracy,
                    ):
                        best = item
                        best_metrics = metrics
        assert best is not None and best_metrics is not None
        protocol = "validation_selected_triple_xgb_fusion"
    weights = tuple(float(best["weights"][name]) for name in source_names)
    test_mixed = sum(weight * test_probs[name] for weight, name in zip(weights, source_names))
    test_mixed = _normalize(test_mixed)
    test_bias = _class_bias(y_val, float(best["bias_power"]))
    test_adjusted = _adjust(test_mixed, bias=test_bias, temperature=float(best["temperature"]))
    test_metrics = _video_metrics(test_manifest, y_test, test_adjusted)

    sample = test_manifest.iloc[[0]].reset_index(drop=True)
    loaded_models = {name: _load_xgb_model(path_pair[0]) for name, path_pair in sources.items()}
    preprocessors = {name: _load_feature_preprocessor(path_pair[1]) for name, path_pair in sources.items()}

    def predict_sample() -> np.ndarray:
        sample_probs = []
        for name, weight in zip(source_names, weights):
            features, _ = _build_feature_matrix(sample, feature_mode=feature_mode)
            features = _apply_feature_preprocessor(features, preprocessors[name])
            sample_probs.append(weight * loaded_models[name].predict_proba(features).astype(np.float32))
        return _adjust(_normalize(sum(sample_probs)), bias=test_bias, temperature=float(best["temperature"]))

    latency = {
        "latency_kind": "processed_feature_sequence",
        "model_side": {"variant": "xgb_triple_fusion", **_timer_ms(predict_sample, latency_warmup, latency_iters)},
    }
    report = {
        "status": "success",
        "protocol": protocol,
        "manifest": str(manifest_path),
        "feature_mode": feature_mode,
        "minimum_accuracy": min_accuracy,
        "minimum_balanced_accuracy": min_balanced_accuracy,
        "maximum_accuracy": max_accuracy,
        "maximum_balanced_accuracy": max_balanced_accuracy,
        "selected": best,
        "test_metrics": test_metrics,
        "latency": latency,
        "sources": {name: {"model": str(paths[0]), "preprocessor": str(paths[1])} for name, paths in sources.items()},
        "candidates_count": len(candidates),
        "note": "For validation-selected runs, weights, class bias, and temperature were selected on validation only. For fixed runs, parameters were provided explicitly.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info(
        "[fusion] test_acc=%.4f test_bal=%.4f test_f1=%.4f | weights=%s bias=%.2f temp=%.2f",
        float(test_metrics["accuracy"]),
        float(test_metrics["balanced_accuracy"]),
        float(test_metrics["f1_macro"]),
        best["weights"],
        float(best["bias_power"]),
        float(best["temperature"]),
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep lightweight fusion of existing 4-class XGBoost predictors.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--final-xgb-model", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/final_xgb/model.json"))
    parser.add_argument("--final-xgb-preprocessor", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/final_xgb/preprocessor.npz"))
    parser.add_argument("--boost-xgb-model", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/boost_xgb/model.json"))
    parser.add_argument("--boost-xgb-preprocessor", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/boost_xgb/preprocessor.npz"))
    parser.add_argument("--targeted-xgb-model", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/targeted_xgb/model.json"))
    parser.add_argument("--targeted-xgb-preprocessor", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/targeted_xgb/preprocessor.npz"))
    parser.add_argument("--feature-mode", choices=["basic", "tsfresh", "copur"], default="tsfresh")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.75)
    parser.add_argument("--max-accuracy", type=float, default=1.0)
    parser.add_argument("--max-balanced-accuracy", type=float, default=1.0)
    parser.add_argument("--latency-warmup", type=int, default=30)
    parser.add_argument("--latency-iters", type=int, default=200)
    parser.add_argument("--fixed-weights", type=str, default="", help="Optional comma-separated weights in final,boost,targeted order.")
    parser.add_argument("--fixed-bias-power", type=float, default=None)
    parser.add_argument("--fixed-temperature", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    run_sweep(
        manifest_path=args.manifest,
        output_json=args.output_json,
        final_xgb_model=args.final_xgb_model,
        final_xgb_preprocessor=args.final_xgb_preprocessor,
        boost_xgb_model=args.boost_xgb_model,
        boost_xgb_preprocessor=args.boost_xgb_preprocessor,
        targeted_xgb_model=args.targeted_xgb_model,
        targeted_xgb_preprocessor=args.targeted_xgb_preprocessor,
        feature_mode=args.feature_mode,
        weight_step=args.weight_step,
        min_accuracy=args.min_accuracy,
        min_balanced_accuracy=args.min_balanced_accuracy,
        max_accuracy=args.max_accuracy,
        max_balanced_accuracy=args.max_balanced_accuracy,
        latency_warmup=args.latency_warmup,
        latency_iters=args.latency_iters,
        fixed_weights=tuple(float(part) for part in args.fixed_weights.split(",")) if args.fixed_weights else None,
        fixed_bias_power=args.fixed_bias_power,
        fixed_temperature=args.fixed_temperature,
    )


if __name__ == "__main__":
    main()
