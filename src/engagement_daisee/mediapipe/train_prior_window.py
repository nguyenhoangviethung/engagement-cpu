from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engagement_daisee.common.config import CHECKPOINT_DIR, RANDOM_SEED
from engagement_daisee.mediapipe.train_product_models import _compute_metrics, _load_manifest
from engagement_daisee.mediapipe.train_window_models import (
    DEFAULT_ALLOWED,
    _aggregate_by_video,
    _build_window_frame,
)


LOGGER = logging.getLogger("mediapipe_prior_window")
DEFAULT_MANIFEST = Path("data/processed/runs/mediapipe_product_features/mediapipe_feature_manifest.csv")
DEFAULT_OUTPUT_DIR = CHECKPOINT_DIR / "runs" / "mediapipe_prior_window"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _weighted_accuracy(labels: np.ndarray, preds: np.ndarray, target_pos_prior: float) -> float:
    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)
    val_pos = float(np.mean(labels == 1))
    val_neg = 1.0 - val_pos
    target_pos = float(np.clip(target_pos_prior, 1e-4, 1.0 - 1e-4))
    target_neg = 1.0 - target_pos
    weights = np.where(labels == 1, target_pos / max(val_pos, 1e-6), target_neg / max(val_neg, 1e-6))
    return float(np.sum(weights * (preds == labels)) / np.sum(weights))


def _select_prior_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    target_pos_prior: float,
    min_recall_neg: float,
) -> tuple[float, dict[str, float]]:
    best_t = 0.5
    best_score = -1.0
    best_metrics = _compute_metrics(labels, probs, best_t)
    fallback = (best_metrics["accuracy"], best_t, best_metrics)
    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= threshold).astype(np.int64)
        metrics = _compute_metrics(labels, probs, float(threshold))
        if metrics["accuracy"] > fallback[0]:
            fallback = (metrics["accuracy"], float(threshold), metrics)
        if metrics["recall_neg"] + 1e-9 < min_recall_neg:
            continue
        score = _weighted_accuracy(labels, preds, target_pos_prior)
        if score > best_score + 1e-9 or (
            abs(score - best_score) <= 1e-9 and metrics["accuracy"] > best_metrics["accuracy"]
        ):
            best_score = score
            best_t = float(threshold)
            best_metrics = metrics
            best_metrics = {**best_metrics, "target_prior_weighted_accuracy": float(score)}
    if best_score < 0:
        _, best_t, best_metrics = fallback
        best_metrics = {**best_metrics, "target_prior_weighted_accuracy": _weighted_accuracy(labels, probs >= best_t, target_pos_prior)}
    return best_t, best_metrics


def _fit_logistic(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ]
    ).fit(x_train, y_train)


def train_prior_window(
    manifest_path: Path,
    output_dir: Path,
    window_sizes: list[int],
    stride: int,
    aggregations: list[str],
    min_recall_negs: list[float],
    target_pos_prior: float | None,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    splits = {split: manifest[manifest["split"] == split].copy() for split in ("train", "validation", "test")}
    split_counts = {split: np.bincount(df["label"].to_numpy(np.int64), minlength=2).tolist() for split, df in splits.items()}
    if target_pos_prior is None:
        target_pos_prior = float(splits["train"]["label"].mean())
    LOGGER.info("Prior-window training | split_counts=%s target_pos_prior=%.4f", split_counts, target_pos_prior)

    results = []
    for window_size in window_sizes:
        built = {split: _build_window_frame(df, window_size, stride, DEFAULT_ALLOWED) for split, df in splits.items()}
        x_train, y_train, train_vids, names = built["train"]
        x_val, y_val, val_vids, _ = built["validation"]
        x_test, y_test, test_vids, _ = built["test"]
        model = _fit_logistic(x_train, y_train, seed)
        train_probs = model.predict_proba(x_train)[:, 1].astype(np.float32)
        val_probs = model.predict_proba(x_val)[:, 1].astype(np.float32)
        test_probs = model.predict_proba(x_test)[:, 1].astype(np.float32)
        for aggregation in aggregations:
            train_yv, train_pv = _aggregate_by_video(train_vids, y_train, train_probs, aggregation)
            val_yv, val_pv = _aggregate_by_video(val_vids, y_val, val_probs, aggregation)
            test_yv, test_pv = _aggregate_by_video(test_vids, y_test, test_probs, aggregation)
            for min_recall_neg in min_recall_negs:
                started = time.time()
                threshold, val_metrics = _select_prior_threshold(
                    val_yv,
                    val_pv,
                    target_pos_prior=target_pos_prior,
                    min_recall_neg=min_recall_neg,
                )
                result = {
                    "model": "logistic_prior_window",
                    "window_size": int(window_size),
                    "stride": int(stride),
                    "aggregation": aggregation,
                    "min_recall_neg": float(min_recall_neg),
                    "target_pos_prior": float(target_pos_prior),
                    "threshold": float(threshold),
                    "num_features": int(x_train.shape[1]),
                    "train_video_metrics": _compute_metrics(train_yv, train_pv, threshold),
                    "validation_video_metrics": val_metrics,
                    "test_video_metrics": _compute_metrics(test_yv, test_pv, threshold),
                    "elapsed_sec": time.time() - started,
                }
                model_dir = output_dir / f"logistic_w{window_size}_{aggregation}_rn{min_recall_neg:.2f}".replace(".", "p")
                model_dir.mkdir(parents=True, exist_ok=True)
                artifact = model_dir / "model.joblib"
                joblib.dump(
                    {
                        "model": model,
                        "feature_names": names,
                        "allowed_features": sorted(DEFAULT_ALLOWED),
                        "window_size": window_size,
                        "stride": stride,
                        "aggregation": aggregation,
                        "threshold": threshold,
                        "target_pos_prior": target_pos_prior,
                        "min_recall_neg": min_recall_neg,
                    },
                    artifact,
                )
                result["artifact"] = str(artifact)
                (model_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                results.append(result)
                m = result["test_video_metrics"]
                LOGGER.info(
                    "w=%d agg=%s min_r0=%.2f t=%.2f test_acc=%.4f bal=%.4f r0=%.4f r1=%.4f",
                    window_size,
                    aggregation,
                    min_recall_neg,
                    threshold,
                    m["accuracy"],
                    m["balanced_accuracy"],
                    m["recall_neg"],
                    m["recall_pos"],
                )

    results = sorted(
        results,
        key=lambda row: (
            row["test_video_metrics"]["accuracy"],
            row["test_video_metrics"]["balanced_accuracy"],
            row["test_video_metrics"]["recall_neg"],
        ),
        reverse=True,
    )
    summary = {
        "protocol": "mediapipe_prior_aware_window_threshold",
        "manifest": str(manifest_path),
        "split_counts": split_counts,
        "window_sizes": window_sizes,
        "stride": stride,
        "aggregations": aggregations,
        "min_recall_negs": min_recall_negs,
        "target_pos_prior": target_pos_prior,
        "leaderboard": results,
    }
    (output_dir / "leaderboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prior-aware threshold tuning for MediaPipe logistic window models.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--aggregations", nargs="+", choices=["mean", "p75", "max"], default=["mean", "p75"])
    parser.add_argument("--min-recall-negs", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--target-pos-prior", type=float, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    summary = train_prior_window(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        window_sizes=args.window_sizes,
        stride=args.stride,
        aggregations=args.aggregations,
        min_recall_negs=args.min_recall_negs,
        target_pos_prior=args.target_pos_prior,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
