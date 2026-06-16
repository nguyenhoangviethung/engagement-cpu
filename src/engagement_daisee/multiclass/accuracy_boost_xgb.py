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


LOGGER = logging.getLogger("accuracy_boost_xgb")


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


def _video_metrics(
    manifest: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    video_labels, video_probs = _aggregate_by_video(manifest, labels, probabilities)
    return _compute_multiclass_metrics(video_labels, np.argmax(video_probs, axis=1), video_probs)


def _candidate_rank(metrics: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(metrics["accuracy"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["f1_macro"]),
    )


def _select_rounds(
    model: XGBClassifier,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_manifest: pd.DataFrame,
    max_rounds: int,
    step: int,
) -> tuple[int, dict[str, object]]:
    rounds_to_check = sorted(set([1, max_rounds, *range(step, max_rounds + 1, step)]))
    best_rounds = 1
    best_metrics: dict[str, object] | None = None
    for rounds in rounds_to_check:
        probabilities = _predict_proba(model, val_features, rounds=rounds)
        metrics = _video_metrics(val_manifest, val_labels, probabilities)
        if best_metrics is None or _candidate_rank(metrics) > _candidate_rank(best_metrics):
            best_rounds = rounds
            best_metrics = metrics
    assert best_metrics is not None
    return best_rounds, best_metrics


def _candidate_grid() -> list[dict[str, float | int | str]]:
    return [
        {"name": "unweighted_d4", "weight_power": 0.0, "max_depth": 4, "min_child_weight": 3.0, "gamma": 0.0},
        {"name": "unweighted_d5", "weight_power": 0.0, "max_depth": 5, "min_child_weight": 3.0, "gamma": 0.0},
        {"name": "unweighted_d6", "weight_power": 0.0, "max_depth": 6, "min_child_weight": 3.0, "gamma": 0.0},
        {"name": "unweighted_d7", "weight_power": 0.0, "max_depth": 7, "min_child_weight": 4.0, "gamma": 0.0},
        {"name": "mild_weight_d5", "weight_power": 0.25, "max_depth": 5, "min_child_weight": 3.0, "gamma": 0.0},
        {"name": "sqrt_weight_d5", "weight_power": 0.50, "max_depth": 5, "min_child_weight": 3.0, "gamma": 0.0},
    ]


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
) -> dict[str, object]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if n_estimators < 1 or round_step < 1 or cpu_threads < 1:
        raise ValueError("n_estimators, round_step, and cpu_threads must be positive.")

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

    candidates = []
    best_model: XGBClassifier | None = None
    best_config: dict[str, float | int | str] | None = None
    best_rounds = 0
    best_val_metrics: dict[str, object] | None = None

    for config in _candidate_grid():
        LOGGER.info("Training candidate %s", config["name"])
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=int(config["max_depth"]),
            min_child_weight=float(config["min_child_weight"]),
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            reg_alpha=0.0,
            gamma=float(config["gamma"]),
            tree_method="hist",
            n_jobs=cpu_threads,
            random_state=seed,
            eval_metric="mlogloss",
        )
        model.fit(
            x_train,
            y_train,
            sample_weight=_sample_weights(y_train, float(config["weight_power"])),
            verbose=False,
        )
        selected_rounds, val_metrics = _select_rounds(
            model,
            x_val,
            y_val,
            val_manifest,
            max_rounds=n_estimators,
            step=round_step,
        )
        item = {
            "config": config,
            "selected_rounds": int(selected_rounds),
            "validation_metrics": val_metrics,
        }
        candidates.append(item)
        LOGGER.info(
            "[%s] rounds=%d val_acc=%.4f val_bal=%.4f val_f1=%.4f",
            config["name"],
            selected_rounds,
            float(val_metrics["accuracy"]),
            float(val_metrics["balanced_accuracy"]),
            float(val_metrics["f1_macro"]),
        )
        if best_val_metrics is None or _candidate_rank(val_metrics) > _candidate_rank(best_val_metrics):
            best_model = model
            best_config = config
            best_rounds = selected_rounds
            best_val_metrics = val_metrics

    assert best_model is not None and best_config is not None and best_val_metrics is not None
    LOGGER.info("Selected %s at %d rounds; building held-out test matrix", best_config["name"], best_rounds)
    x_test, y_test = _build_feature_matrix(test_manifest, feature_mode=feature_mode)
    x_test = _apply_feature_preprocessor(x_test, preprocessor)
    test_probs = _predict_proba(best_model, x_test, rounds=best_rounds)
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
        return _predict_proba(best_model, sample_tabular, rounds=best_rounds)

    def predict_end_to_end() -> np.ndarray:
        tabular = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode=feature_mode)[0]
        tabular = _apply_feature_preprocessor(tabular, preprocessor)
        return _predict_proba(best_model, tabular, rounds=best_rounds)

    latency = {
        "latency_kind": "processed_feature_sequence",
        "sample_feature_path": str(sample_path),
        "model_side": {"variant": "accuracy_boost_xgboost", **_timer_ms(predict_model_side, latency_warmup, latency_iters)},
        "end_to_end": {"variant": "accuracy_boost_xgboost", **_timer_ms(predict_end_to_end, latency_warmup, latency_iters)},
    }
    report = {
        "status": "success",
        "protocol": "validation_selected_xgboost_accuracy_sweep",
        "manifest": str(manifest_path),
        "feature_mode": feature_mode,
        "selection_objective": "validation_video_accuracy",
        "selected": {
            "config": best_config,
            "rounds": int(best_rounds),
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
        "note": "Candidate and boosting-round selection used validation only; test was evaluated once after selection.",
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
    parser = argparse.ArgumentParser(description="Validation-selected CPU XGBoost sweep optimized for 4-class video accuracy.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--feature-mode", choices=["basic", "tsfresh", "copur"], default="tsfresh")
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--round-step", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--latency-warmup", type=int, default=30)
    parser.add_argument("--latency-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
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
    )


if __name__ == "__main__":
    main()
