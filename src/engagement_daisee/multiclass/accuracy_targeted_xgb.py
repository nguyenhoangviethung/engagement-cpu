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
from engagement_daisee.common.config import FOUR_CLASS_FEATURE_MANIFEST_CSV
from engagement_daisee.ml.train import (
    _apply_feature_preprocessor,
    _build_feature_matrix,
    _fit_feature_preprocessor,
)
from engagement_daisee.multiclass.train_all import (
    NUM_CLASSES,
    _aggregate_by_video,
    _compute_multiclass_metrics,
    _split_indices,
)


LOGGER = logging.getLogger("accuracy_targeted_xgb")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def _predict_proba(model: XGBClassifier, features: np.ndarray, rounds: int | None = None) -> np.ndarray:
    kwargs = {"iteration_range": (0, rounds)} if rounds is not None else {}
    return model.predict_proba(features, **kwargs).astype(np.float32)


def _sample_weights(labels: np.ndarray, power: float) -> np.ndarray | None:
    if power <= 0:
        return None
    counts = np.bincount(labels.astype(np.int64), minlength=NUM_CLASSES).astype(np.float64)
    raw = (counts.sum() / np.maximum(counts, 1.0)) ** power
    raw /= max(1e-12, float(raw.mean()))
    return raw[labels].astype(np.float32)


def _video_metrics(manifest: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    video_labels, video_probs = _aggregate_by_video(manifest, labels, probabilities)
    return _compute_multiclass_metrics(video_labels, np.argmax(video_probs, axis=1), video_probs)


def _adjust_probabilities(probabilities: np.ndarray, *, class_bias: np.ndarray | None, temperature: float) -> np.ndarray:
    adjusted = probabilities.astype(np.float64)
    if class_bias is not None:
        adjusted = adjusted * class_bias.reshape(1, -1)
    if temperature != 1.0:
        adjusted = np.power(np.maximum(adjusted, 1e-12), 1.0 / temperature)
    adjusted /= np.clip(adjusted.sum(axis=1, keepdims=True), 1e-12, None)
    return adjusted.astype(np.float32)


def _candidate_rank(
    metrics: dict[str, object],
    *,
    min_accuracy: float,
    min_balanced_accuracy: float,
) -> tuple[int, float, float, float]:
    accuracy = float(metrics["accuracy"])
    balanced_accuracy = float(metrics["balanced_accuracy"])
    feasible = int(accuracy >= min_accuracy and balanced_accuracy >= min_balanced_accuracy)
    return feasible, balanced_accuracy, accuracy, float(metrics["f1_macro"])


def _is_better(
    candidate_metrics: dict[str, object],
    incumbent_metrics: dict[str, object] | None,
    *,
    min_accuracy: float,
    min_balanced_accuracy: float,
) -> bool:
    if incumbent_metrics is None:
        return True
    return _candidate_rank(
        candidate_metrics,
        min_accuracy=min_accuracy,
        min_balanced_accuracy=min_balanced_accuracy,
    ) > _candidate_rank(
        incumbent_metrics,
        min_accuracy=min_accuracy,
        min_balanced_accuracy=min_balanced_accuracy,
    )


def _candidate_grid() -> list[dict[str, float | int | str]]:
    return [
        {"name": "unweighted_d5", "weight_power": 0.0, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 2.0, "gamma": 0.0, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 1.5},
        {"name": "unweighted_d6", "weight_power": 0.0, "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 2.0, "gamma": 0.0, "subsample": 0.90, "colsample_bytree": 0.85, "reg_lambda": 1.5},
        {"name": "unweighted_d7", "weight_power": 0.0, "learning_rate": 0.035, "max_depth": 7, "min_child_weight": 3.0, "gamma": 0.0, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
        {"name": "mild_weight_d5", "weight_power": 0.15, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 2.0, "gamma": 0.0, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 1.5},
        {"name": "mild_weight_d6", "weight_power": 0.15, "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 2.0, "gamma": 0.0, "subsample": 0.90, "colsample_bytree": 0.85, "reg_lambda": 1.5},
        {"name": "balanced_d5", "weight_power": 0.25, "learning_rate": 0.035, "max_depth": 5, "min_child_weight": 3.0, "gamma": 0.1, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
        {"name": "balanced_d6", "weight_power": 0.25, "learning_rate": 0.035, "max_depth": 6, "min_child_weight": 3.0, "gamma": 0.1, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
        {"name": "balanced_d7", "weight_power": 0.35, "learning_rate": 0.03, "max_depth": 7, "min_child_weight": 4.0, "gamma": 0.2, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 2.5},
        {"name": "strong_weight_d6", "weight_power": 0.50, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 4.0, "gamma": 0.25, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 3.0},
    ]


def _bias_powers() -> list[float]:
    return [0.0, 0.10, 0.20, 0.30, 0.40]


def _class_bias(labels: np.ndarray, power: float) -> np.ndarray | None:
    if power <= 0:
        return None
    counts = np.bincount(labels.astype(np.int64), minlength=NUM_CLASSES).astype(np.float64)
    bias = (counts.sum() / np.maximum(counts, 1.0)) ** power
    bias /= max(1e-12, float(bias.mean()))
    return bias.astype(np.float32)


def run_sweep(
    manifest_path: Path,
    output_dir: Path,
    report_json: Path,
    *,
    feature_mode: str,
    n_estimators: int,
    round_step: int,
    cpu_threads: int,
    latency_warmup: int,
    latency_iters: int,
    seed: int,
    min_accuracy: float,
    min_balanced_accuracy: float,
    only_candidates: list[str] | None = None,
) -> dict[str, object]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if n_estimators < 1 or round_step < 1 or cpu_threads < 1:
        raise ValueError("n_estimators, round_step, and cpu_threads must be positive.")
    if not 0.0 <= min_accuracy <= 1.0 or not 0.0 <= min_balanced_accuracy <= 1.0:
        raise ValueError("min_accuracy and min_balanced_accuracy must be between 0 and 1.")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = normalize_manifest_columns(pd.read_csv(manifest_path, low_memory=False))
    required = {"feature_path", "label", "split", "video_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["label"] = manifest["label"].astype(int)
    train_indices, val_indices, test_indices = _split_indices(manifest)
    train_manifest = manifest.iloc[train_indices].reset_index(drop=True)
    val_manifest = manifest.iloc[val_indices].reset_index(drop=True)
    test_manifest = manifest.iloc[test_indices].reset_index(drop=True)

    LOGGER.info("Building %s feature matrices", feature_mode)
    x_train, y_train = _build_feature_matrix(train_manifest, feature_mode=feature_mode)
    x_val, y_val = _build_feature_matrix(val_manifest, feature_mode=feature_mode)
    preprocessor, x_train = _fit_feature_preprocessor(x_train, dim_reduction="none", dim_components=128)
    x_val = _apply_feature_preprocessor(x_val, preprocessor)

    candidates: list[dict[str, object]] = []
    best_model: XGBClassifier | None = None
    best_config: dict[str, float | int | str] | None = None
    best_rounds = 0
    best_bias_power = 0.0
    best_temperature = 1.0
    best_val_metrics: dict[str, object] | None = None
    best_class_bias: np.ndarray | None = None

    candidate_grid = _candidate_grid()
    if only_candidates:
        wanted = {name.strip() for name in only_candidates if name.strip()}
        candidate_grid = [config for config in candidate_grid if str(config["name"]) in wanted]
        if not candidate_grid:
            raise ValueError(f"None of the requested candidates were found: {sorted(wanted)}")

    for config in candidate_grid:
        LOGGER.info("Training candidate %s", config["name"])
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            n_estimators=n_estimators,
            learning_rate=float(config["learning_rate"]),
            max_depth=int(config["max_depth"]),
            min_child_weight=float(config["min_child_weight"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            reg_lambda=float(config["reg_lambda"]),
            reg_alpha=0.0,
            gamma=float(config["gamma"]),
            tree_method="hist",
            n_jobs=cpu_threads,
            random_state=seed,
            eval_metric="mlogloss",
            early_stopping_rounds=80,
        )
        model.fit(
            x_train,
            y_train,
            sample_weight=_sample_weights(y_train, float(config["weight_power"])),
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        best_iteration = getattr(model, "best_iteration", None)
        rounds = int((n_estimators - 1 if best_iteration is None else int(best_iteration)) + 1)
        base_val_probs = _predict_proba(model, x_val)

        candidate_best_metrics: dict[str, object] | None = None
        candidate_best_bias = 0.0
        candidate_best_temperature = 1.0
        candidate_best_probs: np.ndarray | None = None
        candidate_best_rank: tuple[int, float, float, float] | None = None
        for bias_power in _bias_powers():
            class_bias = _class_bias(y_train, bias_power)
            for temperature in (0.85, 1.0, 1.15):
                adjusted = _adjust_probabilities(base_val_probs, class_bias=class_bias, temperature=temperature)
                metrics = _video_metrics(val_manifest, y_val, adjusted)
                rank = _candidate_rank(
                    metrics,
                    min_accuracy=min_accuracy,
                    min_balanced_accuracy=min_balanced_accuracy,
                )
                if candidate_best_rank is None or rank > candidate_best_rank:
                    candidate_best_rank = rank
                    candidate_best_metrics = metrics
                    candidate_best_bias = float(bias_power)
                    candidate_best_temperature = float(temperature)
                    candidate_best_probs = adjusted
        assert candidate_best_metrics is not None and candidate_best_probs is not None

        item = {
            "config": config,
            "selected_rounds": rounds,
            "selected_bias_power": candidate_best_bias,
            "selected_temperature": candidate_best_temperature,
            "validation_metrics": candidate_best_metrics,
        }
        candidates.append(item)
        LOGGER.info(
            "[%s] rounds=%d bias=%.2f temp=%.2f val_acc=%.4f val_bal=%.4f val_f1=%.4f",
            config["name"],
            rounds,
            candidate_best_bias,
            candidate_best_temperature,
            float(candidate_best_metrics["accuracy"]),
            float(candidate_best_metrics["balanced_accuracy"]),
            float(candidate_best_metrics["f1_macro"]),
        )
        if _is_better(
            candidate_best_metrics,
            best_val_metrics,
            min_accuracy=min_accuracy,
            min_balanced_accuracy=min_balanced_accuracy,
        ):
            best_model = model
            best_config = config
            best_rounds = rounds
            best_bias_power = candidate_best_bias
            best_temperature = candidate_best_temperature
            best_val_metrics = candidate_best_metrics
            best_class_bias = _class_bias(y_train, candidate_best_bias)

    assert best_model is not None and best_config is not None and best_val_metrics is not None
    LOGGER.info(
        "Selected %s at %d rounds; bias=%.2f temperature=%.2f; evaluating test split",
        best_config["name"],
        best_rounds,
        best_bias_power,
        best_temperature,
    )
    x_test, y_test = _build_feature_matrix(test_manifest, feature_mode=feature_mode)
    x_test = _apply_feature_preprocessor(x_test, preprocessor)
    test_probs = _adjust_probabilities(
        _predict_proba(best_model, x_test, rounds=best_rounds),
        class_bias=best_class_bias,
        temperature=best_temperature,
    )
    test_row_metrics = _compute_multiclass_metrics(y_test, np.argmax(test_probs, axis=1), test_probs)
    test_video_metrics = _video_metrics(test_manifest, y_test, test_probs)

    model_path = output_dir / "model.json"
    best_model.get_booster().save_model(str(model_path))
    preprocessor_path = output_dir / "preprocessor.npz"
    np.savez(preprocessor_path, **{key: np.asarray(value) for key, value in preprocessor.items()})

    sample_row = test_manifest.iloc[0]
    sample_path = Path(str(sample_row["feature_path"]))
    sample_tabular = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode=feature_mode)[0]
    sample_tabular = _apply_feature_preprocessor(sample_tabular, preprocessor)

    def predict_model_side() -> np.ndarray:
        return _adjust_probabilities(
            _predict_proba(best_model, sample_tabular, rounds=best_rounds),
            class_bias=best_class_bias,
            temperature=best_temperature,
        )

    def predict_end_to_end() -> np.ndarray:
        tabular = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode=feature_mode)[0]
        tabular = _apply_feature_preprocessor(tabular, preprocessor)
        return _adjust_probabilities(
            _predict_proba(best_model, tabular, rounds=best_rounds),
            class_bias=best_class_bias,
            temperature=best_temperature,
        )

    latency = {
        "latency_kind": "processed_feature_sequence",
        "sample_feature_path": str(sample_path),
        "model_side": {"variant": "accuracy_targeted_xgboost", **_timer_ms(predict_model_side, latency_warmup, latency_iters)},
        "end_to_end": {"variant": "accuracy_targeted_xgboost", **_timer_ms(predict_end_to_end, latency_warmup, latency_iters)},
    }
    report = {
        "status": "success",
        "protocol": "validation_selected_xgboost_accuracy_targeted_sweep",
        "manifest": str(manifest_path),
        "feature_mode": feature_mode,
        "selection_objective": "balanced_accuracy",
        "minimum_accuracy": float(min_accuracy),
        "minimum_balanced_accuracy": float(min_balanced_accuracy),
        "selected": {
            "config": best_config,
            "rounds": int(best_rounds),
            "bias_power": float(best_bias_power),
            "temperature": float(best_temperature),
            "validation_metrics": best_val_metrics,
        },
        "test_row_metrics": test_row_metrics,
        "test_video_metrics": test_video_metrics,
        "latency": latency,
        "artifacts": {
            "model": str(model_path),
            "preprocessor": str(preprocessor_path),
        },
        "candidates": candidates,
        "note": "Validation selection prioritized balanced accuracy subject to minimum accuracy guardrail, with class-bias and temperature calibration swept on validation only.",
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info(
        "[selected=%s] test_acc=%.4f test_bal=%.4f test_f1=%.4f | cpu_model=%.3f ms e2e=%.3f ms",
        best_config["name"],
        float(test_video_metrics["accuracy"]),
        float(test_video_metrics["balanced_accuracy"]),
        float(test_video_metrics["f1_macro"]),
        float(latency["model_side"]["latency_ms_mean"]),
        float(latency["end_to_end"]["latency_ms_mean"]),
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted CPU XGBoost sweep with accuracy floor and balanced-accuracy priority.")
    parser.add_argument("--manifest", type=Path, default=FOUR_CLASS_FEATURE_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=["basic", "tsfresh", "copur"], default="tsfresh")
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--round-step", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--latency-warmup", type=int, default=30)
    parser.add_argument("--latency-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-accuracy", type=float, default=0.76)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.70)
    parser.add_argument(
        "--only-candidates",
        type=str,
        default="",
        help="Optional comma-separated subset of candidate names to train, e.g. strong_weight_d6",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    run_sweep(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        report_json=args.report_json,
        feature_mode=args.feature_mode,
        n_estimators=args.n_estimators,
        round_step=args.round_step,
        cpu_threads=args.cpu_threads,
        latency_warmup=args.latency_warmup,
        latency_iters=args.latency_iters,
        seed=args.seed,
        min_accuracy=args.min_accuracy,
        min_balanced_accuracy=args.min_balanced_accuracy,
        only_candidates=[part.strip() for part in args.only_candidates.split(",") if part.strip()] or None,
    )


if __name__ == "__main__":
    main()
