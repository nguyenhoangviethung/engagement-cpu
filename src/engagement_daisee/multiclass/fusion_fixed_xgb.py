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
from engagement_daisee.multiclass.fusion_sweep_xgb import _adjust, _class_bias, _normalize, _video_metrics, _xgb_probs
from engagement_daisee.multiclass.train_all import _split_indices


LOGGER = logging.getLogger("fusion_fixed_xgb")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _load_xgb_model(model_path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(model_path))
    return model


def _timer_ms(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    samples = []
    for _ in range(max(1, iters)):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "latency_ms_mean": float(statistics.fmean(samples)),
        "latency_ms_median": float(statistics.median(samples)),
        "latency_ms_p95": float(ordered[p95_index]),
        "latency_ms_min": float(min(samples)),
        "latency_ms_max": float(max(samples)),
    }


def _parse_weights(value: str) -> tuple[float, float, float]:
    weights = tuple(float(part.strip()) for part in value.split(","))
    if len(weights) != 3:
        raise argparse.ArgumentTypeError("--weights must contain exactly 3 comma-separated values.")
    if not np.isclose(sum(weights), 1.0):
        raise argparse.ArgumentTypeError("--weights must sum to 1.0.")
    return weights


def run_fixed(args: argparse.Namespace) -> dict[str, object]:
    manifest = normalize_manifest_columns(pd.read_csv(args.manifest, low_memory=False))
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["label"] = manifest["label"].astype(int)
    _, val_indices, test_indices = _split_indices(manifest)
    val_manifest = manifest.iloc[val_indices].reset_index(drop=True)
    test_manifest = manifest.iloc[test_indices].reset_index(drop=True)
    y_val = val_manifest["label"].to_numpy(np.int64)
    y_test = test_manifest["label"].to_numpy(np.int64)

    sources = {
        "final_xgb": (args.final_xgb_model, args.final_xgb_preprocessor),
        "boost_xgb": (args.boost_xgb_model, args.boost_xgb_preprocessor),
        "targeted_xgb": (args.targeted_xgb_model, args.targeted_xgb_preprocessor),
    }
    source_names = list(sources)
    weights = dict(zip(source_names, args.weights))

    val_probs = {}
    test_probs = {}
    for name, (model_path, preprocessor_path) in sources.items():
        LOGGER.info("Loading probabilities for %s", name)
        val_probs[name] = _xgb_probs(model_path, preprocessor_path, val_manifest, args.feature_mode)
        test_probs[name] = _xgb_probs(model_path, preprocessor_path, test_manifest, args.feature_mode)

    bias = _class_bias(y_val, args.bias_power)
    val_mixed = _normalize(sum(weights[name] * val_probs[name] for name in source_names))
    test_mixed = _normalize(sum(weights[name] * test_probs[name] for name in source_names))
    val_adjusted = _adjust(val_mixed, bias=bias, temperature=args.temperature)
    test_adjusted = _adjust(test_mixed, bias=bias, temperature=args.temperature)
    validation_metrics = _video_metrics(val_manifest, y_val, val_adjusted)
    test_metrics = _video_metrics(test_manifest, y_test, test_adjusted)

    sample = test_manifest.iloc[[0]].reset_index(drop=True)
    loaded_models = {name: _load_xgb_model(paths[0]) for name, paths in sources.items()}
    preprocessors = {name: _load_feature_preprocessor(paths[1]) for name, paths in sources.items()}

    def predict_sample() -> np.ndarray:
        sample_probs = []
        for name in source_names:
            features, _ = _build_feature_matrix(sample, feature_mode=args.feature_mode)
            features = _apply_feature_preprocessor(features, preprocessors[name])
            sample_probs.append(weights[name] * loaded_models[name].predict_proba(features).astype(np.float32))
        return _adjust(_normalize(sum(sample_probs)), bias=bias, temperature=args.temperature)

    report = {
        "status": "success",
        "protocol": "fixed_triple_xgb_fusion_reproduction",
        "manifest": str(args.manifest),
        "feature_mode": args.feature_mode,
        "fusion_parameters": {
            "weights": weights,
            "bias_power": args.bias_power,
            "temperature": args.temperature,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "latency": {
            "latency_kind": "processed_feature_sequence",
            "model_side": {"variant": "fixed_triple_xgb_fusion", **_timer_ms(predict_sample, args.latency_warmup, args.latency_iters)},
        },
        "sources": {name: {"model": str(paths[0]), "preprocessor": str(paths[1])} for name, paths in sources.items()},
        "note": "Fixed reproduction run: no metric-window search or metric guardrail is used in this evaluator.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info(
        "[fixed-fusion] test_acc=%.4f test_bal=%.4f test_f1=%.4f",
        float(test_metrics["accuracy"]),
        float(test_metrics["balanced_accuracy"]),
        float(test_metrics["f1_macro"]),
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fixed 4-class triple-XGBoost fusion without metric-window search.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--final-xgb-model", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/final_xgb/model.json"))
    parser.add_argument("--final-xgb-preprocessor", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/final_xgb/preprocessor.npz"))
    parser.add_argument("--boost-xgb-model", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/boost_xgb/model.json"))
    parser.add_argument("--boost-xgb-preprocessor", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/boost_xgb/preprocessor.npz"))
    parser.add_argument("--targeted-xgb-model", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/targeted_xgb/model.json"))
    parser.add_argument("--targeted-xgb-preprocessor", type=Path, default=Path("checkpoints/runs/product_4class_fixed_triple_xgb/targeted_xgb/preprocessor.npz"))
    parser.add_argument("--feature-mode", choices=["basic", "tsfresh", "copur"], default="tsfresh")
    parser.add_argument("--weights", type=_parse_weights, default=(0.84, 0.14, 0.02))
    parser.add_argument("--bias-power", type=float, default=0.42)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--latency-warmup", type=int, default=30)
    parser.add_argument("--latency-iters", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    run_fixed(parse_args())


if __name__ == "__main__":
    main()
