from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from engagement_daisee.common.config import CHECKPOINT_DIR, RANDOM_SEED
from engagement_daisee.mediapipe.extract_features import FEATURE_COLUMNS
from engagement_daisee.mediapipe.train_product_models import _compute_metrics, _load_manifest, _select_threshold


LOGGER = logging.getLogger("mediapipe_window_models")
DEFAULT_MANIFEST = Path("data/processed/runs/mediapipe_product_features/mediapipe_feature_manifest.csv")
DEFAULT_OUTPUT_DIR = CHECKPOINT_DIR / "runs" / "mediapipe_window_models"
STATS = ["mean", "std", "min", "max", "p10", "p90", "slope", "mean_abs_diff"]
DEFAULT_ALLOWED = {
    "ear",
    "left_ear",
    "right_ear",
    "gaze_offset",
    "gaze_x",
    "gaze_y",
    "head_tilt",
    "head_yaw_proxy",
    "head_pitch_proxy",
    "mouth_open",
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b <= 0 else float(a / b)


def _window_stats(sequence: np.ndarray, allowed_indices: np.ndarray) -> np.ndarray:
    seq = np.nan_to_num(sequence[:, allowed_indices].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    t = np.arange(len(seq), dtype=np.float32)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc) + 1e-6)
    centered = seq - seq.mean(axis=0, keepdims=True)
    slope = (tc[:, None] * centered).sum(axis=0) / denom
    diff = np.diff(seq, axis=0)
    mad = np.mean(np.abs(diff), axis=0) if len(diff) else np.zeros(seq.shape[1], dtype=np.float32)
    return np.concatenate(
        [
            seq.mean(axis=0),
            seq.std(axis=0),
            seq.min(axis=0),
            seq.max(axis=0),
            np.percentile(seq, 10, axis=0),
            np.percentile(seq, 90, axis=0),
            slope.astype(np.float32),
            mad.astype(np.float32),
        ]
    ).astype(np.float32)


def _iter_windows(sequence: np.ndarray, window_size: int, stride: int) -> list[np.ndarray]:
    if len(sequence) <= window_size:
        return [sequence]
    windows = [sequence[start : start + window_size] for start in range(0, len(sequence) - window_size + 1, stride)]
    if not np.array_equal(windows[-1], sequence[-window_size:]):
        windows.append(sequence[-window_size:])
    return windows


def _build_window_frame(manifest: pd.DataFrame, window_size: int, stride: int, allowed_features: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    allowed_indices = np.asarray([name in allowed_features for name in FEATURE_COLUMNS], dtype=bool)
    base_names = [name for name, keep in zip(FEATURE_COLUMNS, allowed_indices) if keep]
    feature_names = [f"{stat}:{name}" for stat in STATS for name in base_names]
    features = []
    labels = []
    video_ids = []
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        sequence = np.load(Path(str(row.feature_path))).astype(np.float32)
        for window in _iter_windows(sequence, window_size, stride):
            features.append(_window_stats(window, allowed_indices))
            labels.append(int(row.label))
            video_ids.append(str(row.video_id))
        if i % 1000 == 0:
            LOGGER.info("Built windows for %d/%d videos", i, len(manifest))
    return (
        np.stack(features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(video_ids, dtype=object),
        feature_names,
    )


def _aggregate_by_video(video_ids: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"video_id": video_ids, "label": labels, "prob": probabilities})
    if mode == "max":
        grouped = frame.groupby("video_id", sort=False).agg({"label": "max", "prob": "max"})
    elif mode == "p75":
        grouped = frame.groupby("video_id", sort=False).agg({"label": "max", "prob": lambda x: float(np.percentile(x, 75))})
    else:
        grouped = frame.groupby("video_id", sort=False).agg({"label": "max", "prob": "mean"})
    return grouped["label"].to_numpy(np.int64), grouped["prob"].to_numpy(np.float32)


def _benchmark_predict(model, x: np.ndarray, iters: int = 500) -> float:
    for _ in range(20):
        model.predict_proba(x[:1])
    start = time.perf_counter()
    for _ in range(iters):
        model.predict_proba(x[:1])
    return (time.perf_counter() - start) * 1000.0 / iters


def _fit_model(name: str, x: np.ndarray, y: np.ndarray, seed: int):
    counts = np.bincount(y, minlength=2)
    if name == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
            ]
        ).fit(x, y)
    if name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=seed,
        ).fit(x, y)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=seed,
        ).fit(x, y)
    if name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=260,
            max_depth=2,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=10,
            reg_lambda=4.0,
            reg_alpha=0.1,
            tree_method="hist",
            n_jobs=2,
            random_state=seed,
            eval_metric="logloss",
            scale_pos_weight=_safe_div(float(counts[0]), float(counts[1])) if counts[1] else 1.0,
        ).fit(x, y, verbose=False)
    raise ValueError(f"Unsupported model: {name}")


def _save_model(model, model_name: str, path: Path, metadata: dict) -> str:
    path.mkdir(parents=True, exist_ok=True)
    if model_name == "xgboost":
        artifact = path / "model.json"
        model.get_booster().save_model(str(artifact))
    else:
        artifact = path / "model.joblib"
        joblib.dump(model, artifact)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return str(artifact)


def train_window_models(
    manifest_path: Path,
    output_dir: Path,
    window_sizes: list[int],
    stride: int,
    models: list[str],
    aggregation_modes: list[str],
    objective: str,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    splits = {split: manifest[manifest["split"] == split].copy() for split in ("train", "validation", "test")}
    split_counts = {split: np.bincount(df["label"].to_numpy(np.int64), minlength=2).tolist() for split, df in splits.items()}
    LOGGER.info("Window training | split_counts=%s", split_counts)

    results = []
    for window_size in window_sizes:
        LOGGER.info("Building window dataset | window=%d stride=%d", window_size, stride)
        built = {
            split: _build_window_frame(df, window_size, stride, DEFAULT_ALLOWED)
            for split, df in splits.items()
        }
        x_train, y_train, _, feature_names = built["train"]
        x_val, y_val, val_video_ids, _ = built["validation"]
        x_test, y_test, test_video_ids, _ = built["test"]
        LOGGER.info(
            "Window=%d rows train=%d val=%d test=%d features=%d",
            window_size,
            len(y_train),
            len(y_val),
            len(y_test),
            x_train.shape[1],
        )
        for model_name in models:
            started = time.time()
            model = _fit_model(model_name, x_train, y_train, seed=seed)
            val_window_probs = model.predict_proba(x_val)[:, 1].astype(np.float32)
            test_window_probs = model.predict_proba(x_test)[:, 1].astype(np.float32)
            train_window_probs = model.predict_proba(x_train)[:, 1].astype(np.float32)
            for agg in aggregation_modes:
                val_labels_v, val_probs_v = _aggregate_by_video(val_video_ids, y_val, val_window_probs, agg)
                threshold, val_metrics = _select_threshold(val_labels_v, val_probs_v, objective)
                test_labels_v, test_probs_v = _aggregate_by_video(test_video_ids, y_test, test_window_probs, agg)
                train_labels_v, train_probs_v = _aggregate_by_video(built["train"][2], y_train, train_window_probs, agg)
                result = {
                    "model": model_name,
                    "window_size": int(window_size),
                    "stride": int(stride),
                    "aggregation": agg,
                    "threshold": float(threshold),
                    "num_features": int(x_train.shape[1]),
                    "train_video_metrics": _compute_metrics(train_labels_v, train_probs_v, threshold),
                    "validation_video_metrics": val_metrics,
                    "test_video_metrics": _compute_metrics(test_labels_v, test_probs_v, threshold),
                    "latency_ms_per_window": float(_benchmark_predict(model, x_test)),
                    "elapsed_sec": time.time() - started,
                }
                model_dir = output_dir / f"{model_name}_w{window_size}_{agg}"
                result["artifact"] = _save_model(
                    model,
                    model_name,
                    model_dir,
                    {
                        "feature_names": feature_names,
                        "allowed_features": sorted(DEFAULT_ALLOWED),
                        "window_size": window_size,
                        "stride": stride,
                        "aggregation": agg,
                        "threshold": threshold,
                    },
                )
                (model_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                results.append(result)
                m = result["test_video_metrics"]
                LOGGER.info(
                    "%s w=%d agg=%s test_bal=%.4f acc=%.4f r0=%.4f r1=%.4f",
                    model_name,
                    window_size,
                    agg,
                    m["balanced_accuracy"],
                    m["accuracy"],
                    m["recall_neg"],
                    m["recall_pos"],
                )

    results = sorted(
        results,
        key=lambda row: (
            row["test_video_metrics"]["balanced_accuracy"],
            row["test_video_metrics"]["f1_macro"],
            -row["latency_ms_per_window"],
        ),
        reverse=True,
    )
    summary = {
        "protocol": "mediapipe_window_level_video_aggregation",
        "manifest": str(manifest_path),
        "split_counts": split_counts,
        "window_sizes": window_sizes,
        "stride": stride,
        "models": models,
        "aggregation_modes": aggregation_modes,
        "objective": objective,
        "leaderboard": results,
    }
    (output_dir / "leaderboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MediaPipe window-level product models with video aggregation.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=["logistic", "xgboost", "histgb", "rf"], default=["logistic", "xgboost", "histgb"])
    parser.add_argument("--aggregation-modes", nargs="+", choices=["mean", "max", "p75"], default=["mean", "p75"])
    parser.add_argument("--objective", choices=["balanced_accuracy", "f1_macro", "recall_neg"], default="balanced_accuracy")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    summary = train_window_models(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        window_sizes=args.window_sizes,
        stride=args.stride,
        models=args.models,
        aggregation_modes=args.aggregation_modes,
        objective=args.objective,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
