import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from engagement_daisee.common.config import CHECKPOINT_DIR, FEATURE_MANIFEST_CSV, RANDOM_SEED
from engagement_daisee.ml.train import (
    DEFAULT_CPU_WORKERS,
    _apply_feature_preprocessor,
    _build_feature_matrix,
    _compute_metrics,
    _fit_feature_preprocessor,
    _predict_proba_by_round,
    _random_oversample,
    _resolve_backend,
    _save_feature_preprocessor,
    _select_best_iteration_by_f1,
    _select_threshold,
    _train_classifier,
)


LOGGER = logging.getLogger("paper_baseline")
DEFAULT_OUTPUT_DIR = CHECKPOINT_DIR / "runs" / "paper_baseline"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"feature_path", "label", "video_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    manifest = manifest.copy()
    manifest["label"] = manifest["label"].astype(int)
    manifest["video_id"] = manifest["video_id"].astype(str)
    return manifest


def _video_label_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    labels = manifest.groupby("video_id", sort=False)["label"].agg(lambda values: int(values.max()))
    return labels.reset_index(name="label")


def _split_video_80_20(
    manifest: pd.DataFrame,
    seed: int,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    videos = _video_label_frame(manifest)
    train_videos, test_videos = train_test_split(
        videos,
        test_size=0.2,
        random_state=seed,
        stratify=videos["label"],
    )
    train_videos, val_videos = train_test_split(
        train_videos,
        test_size=validation_size,
        random_state=seed,
        stratify=train_videos["label"],
    )

    train_ids = set(train_videos["video_id"].astype(str))
    val_ids = set(val_videos["video_id"].astype(str))
    test_ids = set(test_videos["video_id"].astype(str))
    train_df = manifest[manifest["video_id"].isin(train_ids)].copy()
    val_df = manifest[manifest["video_id"].isin(val_ids)].copy()
    test_df = manifest[manifest["video_id"].isin(test_ids)].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _split_manifest_official(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" not in manifest.columns:
        raise ValueError("Official split mode requires a split column.")
    split = manifest["split"].astype(str).str.lower()
    train_df = manifest[split == "train"].copy()
    val_df = manifest[split == "validation"].copy()
    test_df = manifest[split == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Official split mode found an empty train/validation/test split.")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _smote_lite_oversample(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y_train, minlength=2)
    if counts.min() == 0 or counts[0] == counts[1]:
        return x_train, y_train

    rng = np.random.default_rng(seed)
    minority = int(np.argmin(counts))
    majority_count = int(counts.max())
    minority_indices = np.flatnonzero(y_train == minority)
    majority_indices = np.flatnonzero(y_train != minority)
    extra_count = majority_count - int(counts[minority])

    anchors = rng.choice(minority_indices, size=extra_count, replace=True)
    neighbors = rng.choice(minority_indices, size=extra_count, replace=True)
    lam = rng.random((extra_count, 1), dtype=np.float32)
    synthetic = x_train[anchors] + lam * (x_train[neighbors] - x_train[anchors])

    x_balanced = np.concatenate([x_train[majority_indices], x_train[minority_indices], synthetic], axis=0)
    y_balanced = np.concatenate(
        [
            y_train[majority_indices],
            y_train[minority_indices],
            np.full(extra_count, minority, dtype=y_train.dtype),
        ],
        axis=0,
    )
    order = rng.permutation(len(y_balanced))
    return x_balanced[order].astype(np.float32), y_balanced[order]


def _aggregate_by_video(manifest: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "video_id": manifest["video_id"].astype(str).to_numpy(),
            "label": labels.astype(np.int64),
            "probability": probabilities.astype(np.float32),
        }
    )
    grouped = frame.groupby("video_id", sort=False).agg({"label": "max", "probability": "mean"})
    return grouped["label"].to_numpy(dtype=np.int64), grouped["probability"].to_numpy(dtype=np.float32)


def run_paper_baseline(
    manifest_path: Path,
    output_dir: Path,
    split_mode: str,
    feature_mode: str,
    backend: str,
    threshold_objective: str,
    dim_reduction: str,
    dim_components: int,
    oversample: str,
    cpu_workers: int,
    seed: int,
) -> dict:
    manifest = _load_manifest(manifest_path)
    if split_mode == "video_80_20":
        train_df, val_df, test_df = _split_video_80_20(manifest, seed=seed, validation_size=0.2)
    elif split_mode == "official":
        train_df, val_df, test_df = _split_manifest_official(manifest)
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    resolved_backend = _resolve_backend(backend)
    resolved_workers = max(1, min(cpu_workers, 8))
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Paper baseline | split=%s rows=%d train=%d val=%d test=%d feature_mode=%s backend=%s",
        split_mode,
        len(manifest),
        len(train_df),
        len(val_df),
        len(test_df),
        feature_mode,
        resolved_backend,
    )

    x_train, y_train = _build_feature_matrix(train_df, feature_mode=feature_mode)
    x_val, y_val = _build_feature_matrix(val_df, feature_mode=feature_mode)
    x_test, y_test = _build_feature_matrix(test_df, feature_mode=feature_mode)

    preprocessor, x_train = _fit_feature_preprocessor(x_train, dim_reduction, dim_components)
    x_val = _apply_feature_preprocessor(x_val, preprocessor)
    x_test = _apply_feature_preprocessor(x_test, preprocessor)

    before_counts = np.bincount(y_train, minlength=2).tolist()
    if oversample == "random":
        x_train, y_train = _random_oversample(x_train, y_train, seed=seed)
    elif oversample == "smote_lite":
        x_train, y_train = _smote_lite_oversample(x_train, y_train, seed=seed)
    elif oversample != "none":
        raise ValueError(f"Unsupported oversample: {oversample}")
    after_counts = np.bincount(y_train, minlength=2).tolist()

    model = _train_classifier(
        x_train=x_train,
        y_train=y_train,
        seed=seed,
        backend=resolved_backend,
        cpu_workers=resolved_workers,
    )
    best_iteration, selected_threshold, val_row_metrics = _select_best_iteration_by_f1(
        model=model,
        backend=resolved_backend,
        x_val=x_val,
        y_val=y_val,
        forced_threshold=None,
        threshold_objective=threshold_objective,
    )

    val_probabilities = _predict_proba_by_round(model, resolved_backend, x_val, best_iteration)
    val_video_labels, val_video_probabilities = _aggregate_by_video(val_df, y_val, val_probabilities)
    selected_threshold, val_video_metrics = _select_threshold(
        val_video_labels,
        val_video_probabilities,
        forced_threshold=selected_threshold,
        objective=threshold_objective,
    )

    test_probabilities = _predict_proba_by_round(model, resolved_backend, x_test, best_iteration)
    test_row_metrics = _compute_metrics(y_test, test_probabilities, selected_threshold)
    test_video_labels, test_video_probabilities = _aggregate_by_video(test_df, y_test, test_probabilities)
    test_video_metrics = _compute_metrics(test_video_labels, test_video_probabilities, selected_threshold)

    model_path = output_dir / ("engagement_xgb.json" if resolved_backend == "xgboost" else "engagement_lgbm.txt")
    if resolved_backend == "xgboost":
        model.get_booster()[:best_iteration].save_model(str(model_path))
    else:
        model.booster_.save_model(str(model_path), num_iteration=best_iteration)

    preprocessor_path = output_dir / "engagement_tree.preprocess.npz"
    _save_feature_preprocessor(preprocessor_path, preprocessor)
    summary = {
        "protocol": "paper_baseline",
        "manifest": str(manifest_path),
        "split_mode": split_mode,
        "split_note": "video-level 80:20 is paper-comparable, not the official DAiSEE split",
        "backend": resolved_backend,
        "feature_mode": feature_mode,
        "threshold_objective": threshold_objective,
        "selected_threshold": float(selected_threshold),
        "best_iteration": int(best_iteration),
        "dim_reduction": dim_reduction,
        "dim_components": int(dim_components),
        "oversample": oversample,
        "train_counts_before": before_counts,
        "train_counts_after": after_counts,
        "rows": {"train": int(len(train_df)), "validation": int(len(val_df)), "test": int(len(test_df))},
        "videos": {
            "train": int(train_df["video_id"].nunique()),
            "validation": int(val_df["video_id"].nunique()),
            "test": int(test_df["video_id"].nunique()),
        },
        "validation_row_metrics": val_row_metrics,
        "validation_video_metrics": val_video_metrics,
        "test_row_metrics": test_row_metrics,
        "test_video_metrics": test_video_metrics,
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
    }
    summary_path = output_dir / "paper_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info(
        "Test video | acc=%.4f bal_acc=%.4f recall1=%.4f recall0=%.4f threshold=%.2f",
        test_video_metrics["accuracy"],
        test_video_metrics["balanced_accuracy"],
        test_video_metrics["recall_pos"],
        test_video_metrics["recall_neg"],
        selected_threshold,
    )
    LOGGER.info("Saved paper baseline summary to %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a paper-comparable DAiSEE tabular baseline.")
    parser.add_argument("--manifest", type=Path, default=FEATURE_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-mode", choices=["video_80_20", "official"], default="video_80_20")
    parser.add_argument("--feature-mode", choices=["basic", "tsfresh", "copur"], default="copur")
    parser.add_argument("--backend", choices=["auto", "xgboost", "lightgbm"], default="xgboost")
    parser.add_argument(
        "--threshold-objective",
        choices=["accuracy", "balanced_accuracy", "f1_pos", "f2_pos"],
        default="accuracy",
    )
    parser.add_argument("--dim-reduction", choices=["none", "pca", "svd"], default="svd")
    parser.add_argument("--dim-components", type=int, default=300)
    parser.add_argument("--oversample", choices=["none", "random", "smote_lite"], default="smote_lite")
    parser.add_argument("--cpu-workers", type=int, default=DEFAULT_CPU_WORKERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    summary = run_paper_baseline(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        split_mode=args.split_mode,
        feature_mode=args.feature_mode,
        backend=args.backend,
        threshold_objective=args.threshold_objective,
        dim_reduction=args.dim_reduction,
        dim_components=args.dim_components,
        oversample=args.oversample,
        cpu_workers=args.cpu_workers,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
