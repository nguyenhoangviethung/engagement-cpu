import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import CHECKPOINT_DIR, FEATURE_MANIFEST_CSV, RANDOM_SEED


LOGGER = logging.getLogger("train_ml")
DEFAULT_OUTPUT_PATH = CHECKPOINT_DIR / "engagement_xgb.json"
RUNS_PROCESSED_DIR = FEATURE_MANIFEST_CSV.parent / "runs"
RUNS_CHECKPOINT_DIR = CHECKPOINT_DIR / "runs"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve_manifest_path(manifest_path: Path, run_id: str | None) -> Path:
    if run_id is None and manifest_path.exists():
        return manifest_path

    candidate_paths: list[Path] = []
    if run_id:
        candidate_paths.extend(
            [
                RUNS_PROCESSED_DIR / f"train_{run_id}" / "feature_manifest.csv",
                RUNS_PROCESSED_DIR / f"pipeline_{run_id}" / "feature_manifest.csv",
                RUNS_PROCESSED_DIR / f"extract_{run_id}" / "feature_manifest.csv",
            ]
        )

    candidate_paths.append(manifest_path)
    if manifest_path != FEATURE_MANIFEST_CSV:
        candidate_paths.append(FEATURE_MANIFEST_CSV)

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            LOGGER.info("Resolved manifest path: %s", candidate_path)
            return candidate_path

    searched_paths = "\n".join(f"- {path}" for path in candidate_paths)
    raise FileNotFoundError(
        "Could not locate a feature manifest. Searched:\n"
        f"{searched_paths}\n"
        "Pass --manifest explicitly or generate features with extract/pipeline first."
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision_neg": float(precision[0]),
        "recall_neg": float(recall[0]),
        "f1_neg": float(f1[0]),
        "precision_pos": float(precision[1]),
        "recall_pos": float(recall[1]),
        "f1_pos": float(f1[1]),
        "balanced_accuracy": float((recall[0] + recall[1]) / 2.0),
        "f2_pos": _safe_div(5 * float(precision[1]) * float(recall[1]), 4 * float(precision[1]) + float(recall[1])),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray, forced_threshold: float | None) -> tuple[float, dict[str, float]]:
    if forced_threshold is not None:
        threshold = float(forced_threshold)
        return threshold, _compute_metrics(labels, probabilities, threshold)

    candidates = np.arange(0.05, 0.51, 0.02)
    best_threshold = 0.3
    best_metrics = _compute_metrics(labels, probabilities, best_threshold)

    for threshold in candidates:
        metrics = _compute_metrics(labels, probabilities, float(threshold))
        if metrics["f2_pos"] > best_metrics["f2_pos"] + 1e-8:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        if abs(metrics["f2_pos"] - best_metrics["f2_pos"]) <= 1e-8:
            if metrics["recall_pos"] > best_metrics["recall_pos"] + 1e-8:
                best_threshold = float(threshold)
                best_metrics = metrics
                continue
            if abs(metrics["recall_pos"] - best_metrics["recall_pos"]) <= 1e-8 and metrics["precision_pos"] > best_metrics["precision_pos"] + 1e-8:
                best_threshold = float(threshold)
                best_metrics = metrics

    return best_threshold, best_metrics


def _sequence_to_tabular_features(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim == 1:
        sequence = sequence[:, None]

    sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
    first_frame = sequence[0]
    last_frame = sequence[-1]
    return np.concatenate(
        [
            sequence.mean(axis=0),
            sequence.std(axis=0),
            sequence.min(axis=0),
            sequence.max(axis=0),
            first_frame,
            last_frame,
            last_frame - first_frame,
            np.array([float(sequence.shape[0])], dtype=np.float32),
        ]
    ).astype(np.float32)


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required_columns = {"feature_path", "label", "split"}
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    manifest = manifest.copy()
    manifest["label"] = manifest["label"].astype(int)
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    return manifest


def _maybe_sample_manifest(manifest: pd.DataFrame, sample: bool, sample_videos: int, seed: int) -> pd.DataFrame:
    if not sample:
        return manifest

    rng = np.random.default_rng(seed)
    if "video_id" not in manifest.columns:
        sampled = manifest.sample(n=min(sample_videos, len(manifest)), random_state=seed)
        return sampled.reset_index(drop=True)

    unique_videos = manifest["video_id"].dropna().astype(str).unique().tolist()
    if not unique_videos:
        sampled = manifest.sample(n=min(sample_videos, len(manifest)), random_state=seed)
        return sampled.reset_index(drop=True)

    chosen_videos = rng.choice(unique_videos, size=min(sample_videos, len(unique_videos)), replace=False)
    sampled = manifest[manifest["video_id"].astype(str).isin(set(chosen_videos))].copy()
    return sampled.reset_index(drop=True)


def _build_feature_matrix(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []

    for row_idx, row in enumerate(manifest.itertuples(index=False), start=1):
        feature_path = Path(getattr(row, "feature_path"))
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")

        sequence = np.load(feature_path)
        features.append(_sequence_to_tabular_features(sequence))
        labels.append(int(getattr(row, "label")))

        if row_idx % 5000 == 0:
            LOGGER.info("Loaded %d/%d feature rows", row_idx, len(manifest))

    if not features:
        raise RuntimeError("No feature rows could be loaded from the manifest.")

    return np.stack(features, axis=0), np.asarray(labels, dtype=np.int64)


def _split_from_manifest(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_column = manifest["split"].astype(str).str.lower()
    if split_column.isin({"train", "validation", "test"}).all():
        train_df = manifest[split_column == "train"].copy()
        val_df = manifest[split_column == "validation"].copy()
        test_df = manifest[split_column == "test"].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                "Manifest split columns exist but one of train/validation/test is empty. "
                "Please check the manifest before training."
            )

        return train_df, val_df, test_df

    LOGGER.warning("Manifest does not contain a usable split column; creating a random stratified split.")
    train_df, temp_df = train_test_split(
        manifest,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=manifest["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _train_classifier(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int) -> XGBClassifier:
    class_counts = np.bincount(y_train, minlength=2)
    scale_pos_weight = _safe_div(class_counts[0], class_counts[1]) if class_counts[1] > 0 else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3.0,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        tree_method="hist",
        n_jobs=max(1, (os.cpu_count() or 1) - 1),
        random_state=seed,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        x_train,
        y_train,
        verbose=False,
    )
    return model


def _select_best_iteration_by_f1(
    model: XGBClassifier,
    x_val: np.ndarray,
    y_val: np.ndarray,
    forced_threshold: float | None,
) -> tuple[int, float, dict[str, float]]:
    total_rounds = model.get_booster().num_boosted_rounds()
    if total_rounds <= 0:
        raise RuntimeError("XGBoost model has zero boosted rounds, cannot select best checkpoint.")

    best_iteration = 1
    best_threshold = 0.5
    best_metrics = {
        "f1_pos": -1.0,
        "recall_pos": -1.0,
    }

    for iteration in range(1, total_rounds + 1):
        val_probabilities = model.predict_proba(x_val, iteration_range=(0, iteration))[:, 1]
        threshold, metrics = _select_threshold(y_val, val_probabilities, forced_threshold)

        if metrics["f1_pos"] > best_metrics["f1_pos"] + 1e-8:
            best_iteration = iteration
            best_threshold = threshold
            best_metrics = metrics
            continue

        if abs(metrics["f1_pos"] - best_metrics["f1_pos"]) <= 1e-8 and metrics["recall_pos"] > best_metrics["recall_pos"] + 1e-8:
            best_iteration = iteration
            best_threshold = threshold
            best_metrics = metrics

    return best_iteration, best_threshold, best_metrics


def _resolve_output_path(output_path: Path, run_id: str | None) -> Path:
    if run_id is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    run_checkpoint_dir = RUNS_CHECKPOINT_DIR / f"trainml_{run_id}"
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return run_checkpoint_dir / output_path.name


def train_ml(
    manifest_path: Path,
    output_path: Path,
    threshold: float | None,
    sample: bool,
    sample_videos: int,
    seed: int,
    run_id: str | None,
) -> Path:
    _set_seed(seed)

    manifest_path = _resolve_manifest_path(manifest_path, run_id)
    output_path = _resolve_output_path(output_path, run_id)

    manifest = _load_manifest(manifest_path)
    manifest = _maybe_sample_manifest(manifest, sample=sample, sample_videos=sample_videos, seed=seed)
    train_df, val_df, test_df = _split_from_manifest(manifest)

    LOGGER.info(
        "Loaded manifest | rows=%d train=%d val=%d test=%d sample=%s",
        len(manifest),
        len(train_df),
        len(val_df),
        len(test_df),
        sample,
    )

    x_train, y_train = _build_feature_matrix(train_df)
    x_val, y_val = _build_feature_matrix(val_df)
    x_test, y_test = _build_feature_matrix(test_df)

    model = _train_classifier(x_train, y_train, x_val, y_val, seed=seed)

    best_iteration, selected_threshold, val_metrics = _select_best_iteration_by_f1(
        model=model,
        x_val=x_val,
        y_val=y_val,
        forced_threshold=threshold,
    )

    test_probabilities = model.predict_proba(x_test, iteration_range=(0, best_iteration))[:, 1]
    test_metrics = _compute_metrics(y_test, test_probabilities, selected_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_booster = model.get_booster()[:best_iteration]
    best_booster.save_model(str(output_path))

    summary = {
        "model_type": "xgboost",
        "manifest": str(manifest_path),
        "rows_used": int(len(manifest)),
        "sample": sample,
        "sample_videos": int(sample_videos),
        "selected_threshold": float(selected_threshold),
        "checkpoint_metric": "validation_f1_pos",
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_iteration": int(best_iteration),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    LOGGER.info(
        "Validation | acc=%.4f recall1=%.4f precision1=%.4f f2=%.4f threshold=%.2f",
        val_metrics["accuracy"],
        val_metrics["recall_pos"],
        val_metrics["precision_pos"],
        val_metrics["f2_pos"],
        selected_threshold,
    )
    LOGGER.info(
        "Test | acc=%.4f recall1=%.4f precision1=%.4f f2=%.4f threshold=%.2f",
        test_metrics["accuracy"],
        test_metrics["recall_pos"],
        test_metrics["precision_pos"],
        test_metrics["f2_pos"],
        selected_threshold,
    )
    LOGGER.info("Saved model to %s and summary to %s", output_path, summary_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CPU-only XGBoost baseline on the DAiSEE feature manifest.")
    parser.add_argument("--manifest", type=Path, default=FEATURE_MANIFEST_CSV, help="Path to feature_manifest.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save the XGBoost model")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Force a decision threshold instead of searching for a low-threshold recall-focused value",
    )
    parser.add_argument("--sample", action="store_true", help="Train on a tiny subset for a quick smoke test")
    parser.add_argument("--sample-videos", type=int, default=10, help="Number of videos to keep when --sample is used")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--run-id", type=str, default=None, help="Run id used to resolve run-scoped manifests")
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    train_ml(
        manifest_path=args.manifest,
        output_path=args.output,
        threshold=args.threshold,
        sample=args.sample,
        sample_videos=args.sample_videos,
        seed=args.seed,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()