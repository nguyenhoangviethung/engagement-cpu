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
from sklearn.decomposition import PCA, TruncatedSVD
from xgboost import XGBClassifier

from engagement_daisee.common.config import CHECKPOINT_DIR, FEATURE_MANIFEST_CSV, RANDOM_SEED
from engagement_daisee.common.manifest import normalize_manifest_columns

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


LOGGER = logging.getLogger("train_ml")
DEFAULT_OUTPUT_PATH = CHECKPOINT_DIR / "engagement_xgb.json"
RUNS_PROCESSED_DIR = FEATURE_MANIFEST_CSV.parent / "runs"
RUNS_CHECKPOINT_DIR = CHECKPOINT_DIR / "runs"
DEFAULT_CPU_WORKERS = 2


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


def _is_better_for_objective(candidate: dict[str, float], incumbent: dict[str, float], objective: str) -> bool:
    if candidate[objective] > incumbent[objective] + 1e-8:
        return True
    if abs(candidate[objective] - incumbent[objective]) <= 1e-8:
        if objective == "accuracy":
            return candidate["balanced_accuracy"] > incumbent["balanced_accuracy"] + 1e-8
        if objective == "balanced_accuracy":
            return candidate["f1_pos"] > incumbent["f1_pos"] + 1e-8
        return candidate["recall_pos"] > incumbent["recall_pos"] + 1e-8
    return False


def _select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    forced_threshold: float | None,
    objective: str,
) -> tuple[float, dict[str, float]]:
    if forced_threshold is not None:
        threshold = float(forced_threshold)
        return threshold, _compute_metrics(labels, probabilities, threshold)

    candidates = np.arange(0.05, 0.51, 0.02)
    best_threshold = 0.3
    best_metrics = _compute_metrics(labels, probabilities, best_threshold)

    for threshold in candidates:
        metrics = _compute_metrics(labels, probabilities, float(threshold))
        if _is_better_for_objective(metrics, best_metrics, objective):
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def _sequence_to_basic_features(sequence: np.ndarray) -> np.ndarray:
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


def _sequence_to_tsfresh_like_features(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    centered = sequence - sequence.mean(axis=0, keepdims=True)
    time_steps = np.arange(sequence.shape[0], dtype=np.float32)
    centered_t = time_steps - time_steps.mean()
    slope_den = float(np.sum(centered_t * centered_t) + 1e-6)

    diff = np.diff(sequence, axis=0)
    mean_abs_diff = np.mean(np.abs(diff), axis=0) if diff.size else np.zeros(sequence.shape[1], dtype=np.float32)
    max_abs_diff = np.max(np.abs(diff), axis=0) if diff.size else np.zeros(sequence.shape[1], dtype=np.float32)

    slope = (centered_t[:, None] * centered).sum(axis=0) / slope_den
    energy = np.mean(sequence * sequence, axis=0)
    iqr = np.percentile(sequence, 75, axis=0) - np.percentile(sequence, 25, axis=0)
    median = np.median(sequence, axis=0)
    q10 = np.percentile(sequence, 10, axis=0)
    q90 = np.percentile(sequence, 90, axis=0)
    value_range = np.ptp(sequence, axis=0)
    centered_std = sequence.std(axis=0) + 1e-6
    skewness = np.mean((centered / centered_std) ** 3, axis=0)
    kurtosis = np.mean((centered / centered_std) ** 4, axis=0) - 3.0
    abs_sum_change = np.sum(np.abs(diff), axis=0) if diff.size else np.zeros(sequence.shape[1], dtype=np.float32)
    mean_second_diff = (
        np.mean(np.abs(np.diff(sequence, n=2, axis=0)), axis=0)
        if sequence.shape[0] >= 3
        else np.zeros(sequence.shape[1], dtype=np.float32)
    )

    if sequence.shape[0] >= 3:
        middle = sequence[1:-1]
        peak_count = ((middle > sequence[:-2]) & (middle > sequence[2:])).sum(axis=0).astype(np.float32)
        peak_rate = peak_count / max(1.0, float(sequence.shape[0] - 2))
    else:
        peak_rate = np.zeros(sequence.shape[1], dtype=np.float32)

    if sequence.shape[0] >= 2:
        signs = np.sign(centered)
        zero_cross = ((signs[1:] * signs[:-1]) < 0).sum(axis=0).astype(np.float32)
        zero_cross_rate = zero_cross / max(1.0, float(sequence.shape[0] - 1))

        auto_num = (centered[:-1] * centered[1:]).sum(axis=0)
        auto_den = (centered * centered).sum(axis=0) + 1e-6
        autocorr_lag1 = auto_num / auto_den
    else:
        zero_cross_rate = np.zeros(sequence.shape[1], dtype=np.float32)
        autocorr_lag1 = np.zeros(sequence.shape[1], dtype=np.float32)

    spectrum = np.abs(np.fft.rfft(centered, axis=0)).astype(np.float32)
    if spectrum.shape[0] >= 2:
        low_band = spectrum[1 : min(3, spectrum.shape[0]), :].sum(axis=0)
        full_band = spectrum.sum(axis=0) + 1e-6
        low_freq_ratio = low_band / full_band
    else:
        low_freq_ratio = np.zeros(sequence.shape[1], dtype=np.float32)

    return np.concatenate(
        [
            _sequence_to_basic_features(sequence),
            slope.astype(np.float32),
            mean_abs_diff.astype(np.float32),
            max_abs_diff.astype(np.float32),
            energy.astype(np.float32),
            iqr.astype(np.float32),
            median.astype(np.float32),
            q10.astype(np.float32),
            q90.astype(np.float32),
            value_range.astype(np.float32),
            skewness.astype(np.float32),
            kurtosis.astype(np.float32),
            abs_sum_change.astype(np.float32),
            mean_second_diff.astype(np.float32),
            peak_rate.astype(np.float32),
            zero_cross_rate.astype(np.float32),
            autocorr_lag1.astype(np.float32),
            low_freq_ratio.astype(np.float32),
        ]
    ).astype(np.float32)


def _sequence_to_copur_features(sequence: np.ndarray, top_k: int = 3) -> np.ndarray:
    """OpenFace/BiLSTM-inspired clip aggregation used by Copur's public code.

    The thesis implementation summarizes short overlapping temporal windows with
    simple statistics plus Fourier-domain descriptors before feeding sequence
    models. Here we expose the same idea as a tabular feature mode so tree
    models can use it without re-extracting video features.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    centered = sequence - sequence.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=0)).astype(np.float32)

    if spectrum.shape[0] <= 1:
        spectrum_no_dc = np.zeros((1, sequence.shape[1]), dtype=np.float32)
    else:
        spectrum_no_dc = spectrum[1:]

    if spectrum_no_dc.shape[0] < top_k:
        padding = np.zeros((top_k - spectrum_no_dc.shape[0], sequence.shape[1]), dtype=np.float32)
        top_coefficients = np.concatenate([spectrum_no_dc, padding], axis=0)
    else:
        top_coefficients = np.sort(spectrum_no_dc, axis=0)[-top_k:][::-1]

    length = np.full(sequence.shape[1], float(sequence.shape[0]), dtype=np.float32)

    return np.concatenate(
        [
            sequence.mean(axis=0),
            sequence.var(axis=0),
            sequence.std(axis=0),
            sequence.min(axis=0),
            sequence.max(axis=0),
            length,
            spectrum_no_dc.mean(axis=0),
            spectrum_no_dc.var(axis=0),
            top_coefficients.reshape(-1),
        ]
    ).astype(np.float32)


def _sequence_to_tabular_features(sequence: np.ndarray, feature_mode: str) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim == 1:
        sequence = sequence[:, None]

    sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
    if feature_mode == "basic":
        return _sequence_to_basic_features(sequence)
    if feature_mode == "tsfresh":
        return _sequence_to_tsfresh_like_features(sequence)
    if feature_mode == "copur":
        return _sequence_to_copur_features(sequence)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = normalize_manifest_columns(pd.read_csv(manifest_path))
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


def _build_feature_matrix(manifest: pd.DataFrame, feature_mode: str) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []

    for row_idx, row in enumerate(manifest.itertuples(index=False), start=1):
        feature_path = Path(getattr(row, "feature_path"))
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")

        sequence = np.load(feature_path)
        features.append(_sequence_to_tabular_features(sequence, feature_mode=feature_mode))
        labels.append(int(getattr(row, "label")))

        if row_idx % 5000 == 0:
            LOGGER.info("Loaded %d/%d feature rows", row_idx, len(manifest))

    if not features:
        raise RuntimeError("No feature rows could be loaded from the manifest.")

    return np.stack(features, axis=0), np.asarray(labels, dtype=np.int64)


def _fit_feature_preprocessor(
    x_train: np.ndarray,
    dim_reduction: str,
    dim_components: int,
) -> tuple[dict, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float32)
    mean = x_train.mean(axis=0).astype(np.float32)
    scale = x_train.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    x_scaled = (x_train - mean) / scale

    config: dict = {
        "dim_reduction": dim_reduction,
        "mean": mean,
        "scale": scale,
    }
    if dim_reduction == "none":
        return config, x_scaled.astype(np.float32)

    n_components = min(max(1, dim_components), x_scaled.shape[0] - 1, x_scaled.shape[1])
    if dim_reduction == "pca":
        reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=RANDOM_SEED)
        x_reduced = reducer.fit_transform(x_scaled).astype(np.float32)
        config.update(
            {
                "components": reducer.components_.astype(np.float32),
                "reducer_mean": reducer.mean_.astype(np.float32),
                "explained_variance_ratio": reducer.explained_variance_ratio_.astype(np.float32),
            }
        )
        return config, x_reduced

    if dim_reduction == "svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        x_reduced = reducer.fit_transform(x_scaled).astype(np.float32)
        config.update(
            {
                "components": reducer.components_.astype(np.float32),
                "explained_variance_ratio": reducer.explained_variance_ratio_.astype(np.float32),
            }
        )
        return config, x_reduced

    raise ValueError(f"Unsupported dim_reduction: {dim_reduction}")


def _apply_feature_preprocessor(x: np.ndarray, config: dict) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_scaled = (x - config["mean"]) / config["scale"]
    dim_reduction = str(config.get("dim_reduction", "none"))
    if dim_reduction == "none":
        return x_scaled.astype(np.float32)
    centered = x_scaled
    if dim_reduction == "pca":
        centered = centered - config["reducer_mean"]
    return (centered @ config["components"].T).astype(np.float32)


def _save_feature_preprocessor(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, value in config.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            arrays[key] = np.asarray(value)
    np.savez(path, **arrays)


def _load_feature_preprocessor(path: Path) -> dict:
    payload = np.load(path, allow_pickle=False)
    config = {key: payload[key] for key in payload.files}
    config["dim_reduction"] = str(config["dim_reduction"].item())
    return config


def _random_oversample(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y_train, minlength=2)
    if counts.min() == 0 or counts[0] == counts[1]:
        return x_train, y_train
    rng = np.random.default_rng(seed)
    target = int(counts.max())
    sampled_indices = []
    for label in (0, 1):
        label_indices = np.flatnonzero(y_train == label)
        if len(label_indices) < target:
            extra = rng.choice(label_indices, size=target - len(label_indices), replace=True)
            label_indices = np.concatenate([label_indices, extra])
        sampled_indices.append(label_indices)
    indices = np.concatenate(sampled_indices)
    rng.shuffle(indices)
    return x_train[indices], y_train[indices]


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


def _resolve_backend(requested_backend: str) -> str:
    backend = requested_backend.lower().strip()
    if backend == "auto":
        return "lightgbm" if LGBMClassifier is not None else "xgboost"
    if backend == "lightgbm" and LGBMClassifier is None:
        LOGGER.warning("lightgbm is not installed, fallback to xgboost backend.")
        return "xgboost"
    if backend not in {"xgboost", "lightgbm"}:
        raise ValueError(f"Unsupported backend: {requested_backend}")
    return backend


def _train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    backend: str,
    cpu_workers: int,
):
    class_counts = np.bincount(y_train, minlength=2)
    scale_pos_weight = _safe_div(class_counts[0], class_counts[1]) if class_counts[1] > 0 else 1.0

    if backend == "lightgbm":
        model = LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=cpu_workers,
            class_weight=None,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(x_train, y_train)
        return model

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
        n_jobs=cpu_workers,
        random_state=seed,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train, y_train, verbose=False)
    return model


def _num_boost_rounds(model, backend: str) -> int:
    if backend == "xgboost":
        return int(model.get_booster().num_boosted_rounds())
    return int(model.booster_.num_trees())


def _predict_proba_by_round(model, backend: str, x: np.ndarray, rounds: int) -> np.ndarray:
    if backend == "xgboost":
        return model.predict_proba(x, iteration_range=(0, rounds))[:, 1]
    return model.predict_proba(x, num_iteration=rounds)[:, 1]


def _select_best_iteration_by_f1(
    model,
    backend: str,
    x_val: np.ndarray,
    y_val: np.ndarray,
    forced_threshold: float | None,
    threshold_objective: str,
) -> tuple[int, float, dict[str, float]]:
    total_rounds = _num_boost_rounds(model, backend)
    if total_rounds <= 0:
        raise RuntimeError("Tree model has zero boosted rounds, cannot select best checkpoint.")

    best_iteration = 1
    best_threshold = 0.5
    first_probabilities = _predict_proba_by_round(model, backend, x_val, 1)
    best_metrics = _compute_metrics(y_val, first_probabilities, threshold=best_threshold)

    for iteration in range(1, total_rounds + 1):
        val_probabilities = _predict_proba_by_round(model, backend, x_val, iteration)
        threshold, metrics = _select_threshold(
            y_val,
            val_probabilities,
            forced_threshold,
            objective=threshold_objective,
        )

        if _is_better_for_objective(metrics, best_metrics, threshold_objective):
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
    backend: str,
    feature_mode: str,
    cpu_workers: int,
    threshold_objective: str,
    dim_reduction: str,
    dim_components: int,
    oversample: str,
) -> Path:
    _set_seed(seed)

    manifest_path = _resolve_manifest_path(manifest_path, run_id)
    output_path = _resolve_output_path(output_path, run_id)

    resolved_backend = _resolve_backend(backend)
    resolved_workers = max(1, min(cpu_workers, os.cpu_count() or cpu_workers))

    manifest = _load_manifest(manifest_path)
    manifest = _maybe_sample_manifest(manifest, sample=sample, sample_videos=sample_videos, seed=seed)
    train_df, val_df, test_df = _split_from_manifest(manifest)

    LOGGER.info(
        "Loaded manifest | rows=%d train=%d val=%d test=%d sample=%s backend=%s feature_mode=%s cpu_workers=%d",
        len(manifest),
        len(train_df),
        len(val_df),
        len(test_df),
        sample,
        resolved_backend,
        feature_mode,
        resolved_workers,
    )

    x_train, y_train = _build_feature_matrix(train_df, feature_mode=feature_mode)
    x_val, y_val = _build_feature_matrix(val_df, feature_mode=feature_mode)
    x_test, y_test = _build_feature_matrix(test_df, feature_mode=feature_mode)

    preprocessor_config, x_train = _fit_feature_preprocessor(
        x_train,
        dim_reduction=dim_reduction,
        dim_components=dim_components,
    )
    x_val = _apply_feature_preprocessor(x_val, preprocessor_config)
    x_test = _apply_feature_preprocessor(x_test, preprocessor_config)

    if oversample == "random":
        before_counts = np.bincount(y_train, minlength=2)
        x_train, y_train = _random_oversample(x_train, y_train, seed=seed)
        after_counts = np.bincount(y_train, minlength=2)
        LOGGER.info(
            "Applied random oversampling | before=%s after=%s",
            before_counts.tolist(),
            after_counts.tolist(),
        )

    model = _train_classifier(
        x_train,
        y_train,
        seed=seed,
        backend=resolved_backend,
        cpu_workers=resolved_workers,
    )

    best_iteration, selected_threshold, val_metrics = _select_best_iteration_by_f1(
        model=model,
        backend=resolved_backend,
        x_val=x_val,
        y_val=y_val,
        forced_threshold=threshold,
        threshold_objective=threshold_objective,
    )

    test_probabilities = _predict_proba_by_round(model, resolved_backend, x_test, best_iteration)
    test_metrics = _compute_metrics(y_test, test_probabilities, selected_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_backend == "xgboost":
        best_booster = model.get_booster()[:best_iteration]
        best_booster.save_model(str(output_path))
    else:
        model.booster_.save_model(str(output_path), num_iteration=best_iteration)

    summary = {
        "model_type": resolved_backend,
        "manifest": str(manifest_path),
        "rows_used": int(len(manifest)),
        "sample": sample,
        "sample_videos": int(sample_videos),
        "selected_threshold": float(selected_threshold),
        "checkpoint_metric": f"validation_{threshold_objective}",
        "threshold_objective": threshold_objective,
        "test_metrics": test_metrics,
        "best_iteration": int(best_iteration),
        "feature_mode": feature_mode,
        "cpu_workers": int(resolved_workers),
        "dim_reduction": dim_reduction,
        "dim_components": int(dim_components),
        "oversample": oversample,
    }
    summary_path = output_path.with_suffix(".summary.json")
    preprocessor_path = output_path.with_suffix(".preprocess.npz")
    _save_feature_preprocessor(preprocessor_path, preprocessor_config)
    summary["preprocessor_path"] = str(preprocessor_path)
    summary_path.write_text(json.dumps(summary, indent=2))
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
    parser = argparse.ArgumentParser(description="Train a CPU-only tree baseline with TS-Fresh-like features.")
    parser.add_argument("--manifest", type=Path, default=FEATURE_MANIFEST_CSV, help="Path to feature_manifest.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Where to save the tree model")
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
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "xgboost", "lightgbm"],
        help="Tree backend to use; auto prefers lightgbm if available",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="tsfresh",
        choices=["basic", "tsfresh", "copur"],
        help="Feature engineering mode from each temporal sequence",
    )
    parser.add_argument(
        "--dim-reduction",
        type=str,
        default="none",
        choices=["none", "pca", "svd"],
        help="Optional train-fitted dimensionality reduction before tree training.",
    )
    parser.add_argument("--dim-components", type=int, default=128, help="PCA/SVD components when enabled")
    parser.add_argument(
        "--oversample",
        type=str,
        default="none",
        choices=["none", "random"],
        help="Train-only minority oversampling. 'random' is dependency-free SMOTE-lite.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=DEFAULT_CPU_WORKERS,
        help="Maximum CPU workers for tree training (recommended: 1-2)",
    )
    parser.add_argument(
        "--threshold-objective",
        type=str,
        default="f2_pos",
        choices=["accuracy", "balanced_accuracy", "f1_pos", "f2_pos"],
        help="Validation metric for selecting the saved tree count and threshold.",
    )
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
        backend=args.backend,
        feature_mode=args.feature_mode,
        cpu_workers=args.cpu_workers,
        threshold_objective=args.threshold_objective,
        dim_reduction=args.dim_reduction,
        dim_components=args.dim_components,
        oversample=args.oversample,
    )


if __name__ == "__main__":
    main()
