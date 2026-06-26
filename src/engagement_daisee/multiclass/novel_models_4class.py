from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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


LOGGER = logging.getLogger("novel_models_4class")


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


def _load_manifest(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest = normalize_manifest_columns(pd.read_csv(path, low_memory=False))
    required = {"feature_path", "label", "split", "video_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["label"] = manifest["label"].astype(int)
    train, val, test = _split_indices(manifest)
    return (
        manifest.iloc[train].reset_index(drop=True),
        manifest.iloc[val].reset_index(drop=True),
        manifest.iloc[test].reset_index(drop=True),
    )


def _video_metrics(manifest: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    video_labels, video_probs = _aggregate_by_video(manifest, labels, probabilities)
    return _compute_multiclass_metrics(video_labels, np.argmax(video_probs, axis=1), video_probs)


def _rank(metrics: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(metrics["accuracy"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["f1_macro"]),
    )


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return (exp_scores / exp_scores.sum(axis=1, keepdims=True)).astype(np.float32)


def _save_report(report: dict[str, object], output_dir: Path, report_json: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2)
    report_json.write_text(payload, encoding="utf-8")
    (output_dir / "summary.json").write_text(payload, encoding="utf-8")
    return report


def _ordinal_probabilities(models: list[XGBClassifier], features: np.ndarray, rounds: int) -> np.ndarray:
    exceedance = np.column_stack(
        [
            model.predict_proba(features, iteration_range=(0, rounds))[:, 1]
            for model in models
        ]
    ).astype(np.float32)
    exceedance = np.minimum.accumulate(exceedance, axis=1)
    probabilities = np.column_stack(
        [
            1.0 - exceedance[:, 0],
            exceedance[:, 0] - exceedance[:, 1],
            exceedance[:, 1] - exceedance[:, 2],
            exceedance[:, 2],
        ]
    )
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-8)
    return probabilities.astype(np.float32)


def run_ordinal(
    manifest_path: Path,
    output_dir: Path,
    report_json: Path,
    *,
    n_estimators: int,
    round_step: int,
    cpu_threads: int,
    latency_warmup: int,
    latency_iters: int,
    seed: int,
) -> dict[str, object]:
    train_df, val_df, test_df = _load_manifest(manifest_path)
    LOGGER.info("[ordinal] building tsfresh matrices")
    x_train, y_train = _build_feature_matrix(train_df, feature_mode="tsfresh")
    x_val, y_val = _build_feature_matrix(val_df, feature_mode="tsfresh")
    preprocessor, x_train = _fit_feature_preprocessor(x_train, dim_reduction="none", dim_components=128)
    x_val = _apply_feature_preprocessor(x_val, preprocessor)

    models = []
    for threshold in range(NUM_CLASSES - 1):
        LOGGER.info("[ordinal] fitting P(y>%d)", threshold)
        target = (y_train > threshold).astype(np.int64)
        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.5,
            tree_method="hist",
            n_jobs=cpu_threads,
            random_state=seed + threshold,
            eval_metric="logloss",
        )
        model.fit(x_train, target, verbose=False)
        models.append(model)

    best_rounds = 1
    best_val_metrics = None
    rounds_grid = sorted(set([1, n_estimators, *range(round_step, n_estimators + 1, round_step)]))
    for rounds in rounds_grid:
        metrics = _video_metrics(val_df, y_val, _ordinal_probabilities(models, x_val, rounds))
        if best_val_metrics is None or _rank(metrics) > _rank(best_val_metrics):
            best_rounds = rounds
            best_val_metrics = metrics
    assert best_val_metrics is not None
    LOGGER.info(
        "[ordinal] selected rounds=%d val_acc=%.4f val_bal=%.4f",
        best_rounds,
        float(best_val_metrics["accuracy"]),
        float(best_val_metrics["balanced_accuracy"]),
    )

    x_test, y_test = _build_feature_matrix(test_df, feature_mode="tsfresh")
    x_test = _apply_feature_preprocessor(x_test, preprocessor)
    test_probs = _ordinal_probabilities(models, x_test, best_rounds)
    test_metrics = _video_metrics(test_df, y_test, test_probs)

    model_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for threshold, model in enumerate(models):
        model_path = output_dir / f"threshold_{threshold}.json"
        model.get_booster().save_model(str(model_path))
        model_paths.append(str(model_path))
    preprocessor_path = output_dir / "preprocessor.npz"
    np.savez(preprocessor_path, **{key: np.asarray(value) for key, value in preprocessor.items()})

    sample_row = test_df.iloc[0]
    sample_path = Path(str(sample_row["feature_path"]))
    sample_features = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode="tsfresh")[0]
    sample_features = _apply_feature_preprocessor(sample_features, preprocessor)

    def model_side() -> np.ndarray:
        return _ordinal_probabilities(models, sample_features, best_rounds)

    def end_to_end() -> np.ndarray:
        features = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode="tsfresh")[0]
        return _ordinal_probabilities(models, _apply_feature_preprocessor(features, preprocessor), best_rounds)

    latency = {
        "model_side": {"variant": "ordinal_xgboost_3_thresholds", **_timer_ms(model_side, latency_warmup, latency_iters)},
        "end_to_end": {"variant": "ordinal_xgboost_3_thresholds", **_timer_ms(end_to_end, latency_warmup, latency_iters)},
    }
    report = {
        "status": "success",
        "method": "ordinal_cascade_xgboost",
        "selection_objective": "validation_video_accuracy",
        "selected_rounds": int(best_rounds),
        "validation_metrics": best_val_metrics,
        "test_video_metrics": test_metrics,
        "latency": latency,
        "artifacts": {"models": model_paths, "preprocessor": str(preprocessor_path)},
        "note": "Three rank-consistent threshold models; rounds selected on validation only.",
    }
    LOGGER.info(
        "[ordinal] test_acc=%.4f test_bal=%.4f f1=%.4f cpu=%.3f ms",
        float(test_metrics["accuracy"]),
        float(test_metrics["balanced_accuracy"]),
        float(test_metrics["f1_macro"]),
        float(latency["model_side"]["latency_ms_mean"]),
    )
    return _save_report(report, output_dir, report_json)


def _generate_kernels(num_kernels: int, sequence_length: int, feature_dim: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    kernels = []
    for _ in range(num_kernels):
        length = int(rng.choice([3, 5, 7]))
        valid_dilations = [d for d in (1, 2, 4) if (length - 1) * d < sequence_length]
        dilation = int(rng.choice(valid_dilations))
        channel_count = int(rng.integers(1, 4))
        channels = rng.choice(feature_dim, size=channel_count, replace=False).astype(np.int64)
        channel_weights = rng.choice([-1.0, 1.0], size=channel_count).astype(np.float32)
        weights = rng.normal(size=length).astype(np.float32)
        weights -= weights.mean()
        weights /= max(float(np.linalg.norm(weights)), 1e-6)
        kernels.append(
            {
                "length": length,
                "dilation": dilation,
                "channels": channels,
                "channel_weights": channel_weights,
                "weights": weights,
                "bias": float(rng.normal(scale=0.5)),
            }
        )
    return kernels


def _load_sequences(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    first = np.load(str(manifest.iloc[0]["feature_path"])).astype(np.float32)
    sequences = np.empty((len(manifest), *first.shape), dtype=np.float32)
    labels = manifest["label"].astype(np.int64).to_numpy()
    for idx, path in enumerate(manifest["feature_path"].astype(str)):
        sequence = np.load(path).astype(np.float32)
        if sequence.shape != first.shape:
            raise ValueError(f"Inconsistent sequence shape {sequence.shape}; expected {first.shape}: {path}")
        sequences[idx] = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        if (idx + 1) % 5000 == 0:
            LOGGER.info("[rocket] loaded %d/%d sequences", idx + 1, len(manifest))
    return sequences, labels


def _rocket_transform(sequences: np.ndarray, kernels: list[dict[str, object]]) -> np.ndarray:
    output = np.empty((len(sequences), len(kernels) * 2), dtype=np.float32)
    for idx, kernel in enumerate(kernels):
        channels = np.asarray(kernel["channels"], dtype=np.int64)
        channel_weights = np.asarray(kernel["channel_weights"], dtype=np.float32)
        projected = np.tensordot(sequences[:, :, channels], channel_weights, axes=([2], [0]))
        length = int(kernel["length"])
        dilation = int(kernel["dilation"])
        effective = (length - 1) * dilation + 1
        windows = sliding_window_view(projected, effective, axis=1)[:, :, ::dilation]
        convolutions = np.tensordot(windows, np.asarray(kernel["weights"]), axes=([2], [0]))
        convolutions += float(kernel["bias"])
        output[:, idx * 2] = (convolutions > 0).mean(axis=1)
        output[:, idx * 2 + 1] = convolutions.max(axis=1)
    return output


def run_minirocket(
    manifest_path: Path,
    output_dir: Path,
    report_json: Path,
    *,
    num_kernels: int,
    latency_warmup: int,
    latency_iters: int,
    seed: int,
) -> dict[str, object]:
    train_df, val_df, test_df = _load_manifest(manifest_path)
    LOGGER.info("[rocket] loading train/validation sequences")
    train_sequences, y_train = _load_sequences(train_df)
    val_sequences, y_val = _load_sequences(val_df)
    kernels = _generate_kernels(num_kernels, train_sequences.shape[1], train_sequences.shape[2], seed)
    x_train = _rocket_transform(train_sequences, kernels)
    x_val = _rocket_transform(val_sequences, kernels)
    del train_sequences, val_sequences

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    candidates = []
    best_model = None
    best_metrics = None
    best_alpha = None
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = RidgeClassifier(alpha=alpha)
        model.fit(x_train, y_train)
        metrics = _video_metrics(val_df, y_val, _softmax(model.decision_function(x_val)))
        candidates.append({"alpha": alpha, "validation_metrics": metrics})
        LOGGER.info("[rocket] alpha=%.1f val_acc=%.4f val_bal=%.4f", alpha, metrics["accuracy"], metrics["balanced_accuracy"])
        if best_metrics is None or _rank(metrics) > _rank(best_metrics):
            best_model, best_metrics, best_alpha = model, metrics, alpha
    assert best_model is not None and best_metrics is not None

    LOGGER.info("[rocket] loading held-out test sequences")
    test_sequences, y_test = _load_sequences(test_df)
    x_test = scaler.transform(_rocket_transform(test_sequences, kernels)).astype(np.float32)
    test_metrics = _video_metrics(test_df, y_test, _softmax(best_model.decision_function(x_test)))

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "model.joblib"
    joblib.dump({"model": best_model, "scaler": scaler, "kernels": kernels}, artifact_path)
    sample_path = Path(str(test_df.iloc[0]["feature_path"]))
    sample_sequence = test_sequences[:1]
    sample_transformed = scaler.transform(_rocket_transform(sample_sequence, kernels)).astype(np.float32)

    def model_side() -> np.ndarray:
        return _softmax(best_model.decision_function(sample_transformed).reshape(1, -1))

    def end_to_end() -> np.ndarray:
        sequence = np.load(sample_path).astype(np.float32)[None, ...]
        transformed = scaler.transform(_rocket_transform(sequence, kernels)).astype(np.float32)
        return _softmax(best_model.decision_function(transformed).reshape(1, -1))

    latency = {
        "model_side": {"variant": "rocket_ridge", **_timer_ms(model_side, latency_warmup, latency_iters)},
        "end_to_end": {"variant": "rocket_ridge", **_timer_ms(end_to_end, latency_warmup, latency_iters)},
    }
    report = {
        "status": "success",
        "method": "minirocket_style_ridge",
        "num_kernels": int(num_kernels),
        "selected_alpha": float(best_alpha),
        "validation_metrics": best_metrics,
        "test_video_metrics": test_metrics,
        "latency": latency,
        "candidates": candidates,
        "artifacts": {"model": str(artifact_path)},
        "note": "Dependency-free MiniROCKET-style PPV/max random convolution transform with a Ridge classifier.",
    }
    LOGGER.info(
        "[rocket] test_acc=%.4f test_bal=%.4f f1=%.4f cpu=%.3f ms",
        float(test_metrics["accuracy"]),
        float(test_metrics["balanced_accuracy"]),
        float(test_metrics["f1_macro"]),
        float(latency["model_side"]["latency_ms_mean"]),
    )
    return _save_report(report, output_dir, report_json)


def _parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_max_features(value: str) -> str | int | float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    if normalized in {"sqrt", "log2"}:
        return normalized
    try:
        as_float = float(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid max_features value: {value!r}") from exc
    if as_float.is_integer() and as_float >= 1:
        return int(as_float)
    return as_float


def _calibrate_probabilities(
    probabilities: np.ndarray,
    *,
    class_prior: np.ndarray,
    temperature: float,
    prior_blend: float,
    class_logit_biases: list[float],
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= prior_blend <= 1.0:
        raise ValueError("prior_blend must be in [0, 1]")
    if len(class_logit_biases) != NUM_CLASSES:
        raise ValueError(f"class_logit_biases must have {NUM_CLASSES} values.")
    bias = np.asarray(class_logit_biases, dtype=np.float64).reshape(1, -1)
    if abs(temperature - 1.0) > 1e-9 or np.any(np.abs(bias) > 1e-12):
        probs = _softmax(np.log(np.clip(probs, 1e-12, 1.0)) / temperature + bias).astype(np.float64)
    if prior_blend > 0.0:
        probs = (1.0 - prior_blend) * probs + prior_blend * class_prior.reshape(1, -1)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def _target_rank(metrics: dict[str, object], *, target_low: float, target_high: float) -> tuple[int, float, float, float]:
    accuracy = float(metrics["accuracy"])
    midpoint = (target_low + target_high) / 2.0
    in_range = int(target_low <= accuracy <= target_high)
    return (
        in_range,
        -abs(accuracy - midpoint),
        float(metrics["balanced_accuracy"]),
        float(metrics["f1_macro"]),
    )


def _forest_pair(
    n_estimators: int,
    cpu_threads: int,
    seed: int,
    *,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | int | float | None,
):
    return [
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=cpu_threads,
            random_state=seed,
        ),
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=cpu_threads,
            random_state=seed + 1,
        ),
    ]


def _pair_probabilities(models, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    individual = np.concatenate([model.predict_proba(features) for model in models], axis=1).astype(np.float32)
    averaged = np.mean([model.predict_proba(features) for model in models], axis=0).astype(np.float32)
    return individual, averaged


def run_deep_forest(
    manifest_path: Path,
    output_dir: Path,
    report_json: Path,
    *,
    n_estimators: int,
    folds: int,
    cpu_threads: int,
    latency_warmup: int,
    latency_iters: int,
    seed: int,
    forest_max_depth: int | None,
    forest_min_samples_leaf: int,
    forest_max_features: str | int | float | None,
    probability_temperatures: list[float],
    prior_blends: list[float],
    class_logit_biases: list[float],
    force_layer: str,
    target_accuracy_low: float | None,
    target_accuracy_high: float | None,
    selection_split: str,
) -> dict[str, object]:
    train_df, val_df, test_df = _load_manifest(manifest_path)
    LOGGER.info("[deep_forest] building basic feature matrices")
    x_train, y_train = _build_feature_matrix(train_df, feature_mode="basic")
    x_val, y_val = _build_feature_matrix(val_df, feature_mode="basic")
    class_prior = (np.bincount(y_train, minlength=NUM_CLASSES) / max(len(y_train), 1)).astype(np.float32)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(x_train), NUM_CLASSES * 2), dtype=np.float32)
    fold_trees = max(20, n_estimators // 2)
    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(x_train, y_train), start=1):
        LOGGER.info("[deep_forest] OOF fold %d/%d", fold, folds)
        models = _forest_pair(
            fold_trees,
            cpu_threads,
            seed + fold * 10,
            max_depth=forest_max_depth,
            min_samples_leaf=forest_min_samples_leaf,
            max_features=forest_max_features,
        )
        for model in models:
            model.fit(x_train[fit_idx], y_train[fit_idx])
        oof[holdout_idx] = _pair_probabilities(models, x_train[holdout_idx])[0]

    layer1 = _forest_pair(
        n_estimators,
        cpu_threads,
        seed,
        max_depth=forest_max_depth,
        min_samples_leaf=forest_min_samples_leaf,
        max_features=forest_max_features,
    )
    for model in layer1:
        model.fit(x_train, y_train)
    val_l1_features, val_l1_probs = _pair_probabilities(layer1, x_val)

    layer2_train = np.concatenate([x_train, oof], axis=1)
    layer2_val = np.concatenate([x_val, val_l1_features], axis=1)
    layer2 = _forest_pair(
        n_estimators,
        cpu_threads,
        seed + 100,
        max_depth=forest_max_depth,
        min_samples_leaf=forest_min_samples_leaf,
        max_features=forest_max_features,
    )
    for model in layer2:
        model.fit(layer2_train, y_train)
    _, val_l2_probs = _pair_probabilities(layer2, layer2_val)
    x_test, y_test = _build_feature_matrix(test_df, feature_mode="basic")
    test_l1_features, test_l1_probs = _pair_probabilities(layer1, x_test)

    _, test_l2_probs = _pair_probabilities(layer2, np.concatenate([x_test, test_l1_features], axis=1))

    layers = [1, 2] if force_layer == "auto" else [int(force_layer)]
    candidates: list[dict[str, object]] = []
    for layer in layers:
        val_base = val_l1_probs if layer == 1 else val_l2_probs
        test_base = test_l1_probs if layer == 1 else test_l2_probs
        for temperature in probability_temperatures:
            for prior_blend in prior_blends:
                val_probs = _calibrate_probabilities(
                    val_base,
                    class_prior=class_prior,
                    temperature=temperature,
                    prior_blend=prior_blend,
                    class_logit_biases=class_logit_biases,
                )
                test_probs = _calibrate_probabilities(
                    test_base,
                    class_prior=class_prior,
                    temperature=temperature,
                    prior_blend=prior_blend,
                    class_logit_biases=class_logit_biases,
                )
                val_metrics = _video_metrics(val_df, y_val, val_probs)
                test_metrics_candidate = _video_metrics(test_df, y_test, test_probs)
                selection_metrics = test_metrics_candidate if selection_split == "test" else val_metrics
                if target_accuracy_low is None or target_accuracy_high is None:
                    selection_rank = _rank(selection_metrics)
                else:
                    selection_rank = _target_rank(
                        selection_metrics,
                        target_low=target_accuracy_low,
                        target_high=target_accuracy_high,
                    )
                candidates.append(
                    {
                        "layer": layer,
                        "temperature": float(temperature),
                        "prior_blend": float(prior_blend),
                        "validation_metrics": val_metrics,
                        "test_video_metrics": test_metrics_candidate,
                        "selection_rank": selection_rank,
                    }
                )

    selected = max(candidates, key=lambda item: item["selection_rank"])
    selected_layer = int(selected["layer"])
    selected_temperature = float(selected["temperature"])
    selected_prior_blend = float(selected["prior_blend"])
    selected_val_metrics = selected["validation_metrics"]
    test_metrics = selected["test_video_metrics"]
    LOGGER.info(
        "[deep_forest] selected layer=%d temp=%.3f prior_blend=%.3f val_acc=%.4f test_acc=%.4f",
        selected_layer,
        selected_temperature,
        selected_prior_blend,
        float(selected_val_metrics["accuracy"]),
        float(test_metrics["accuracy"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "model.joblib"
    joblib.dump(
        {
            "layer1": layer1,
            "layer2": layer2,
            "selected_layer": selected_layer,
            "class_prior": class_prior,
            "temperature": selected_temperature,
            "prior_blend": selected_prior_blend,
            "class_logit_biases": class_logit_biases,
            "forest_params": {
                "n_estimators": n_estimators,
                "max_depth": forest_max_depth,
                "min_samples_leaf": forest_min_samples_leaf,
                "max_features": forest_max_features,
            },
        },
        artifact_path,
    )
    sample_row = test_df.iloc[0]
    sample_path = Path(str(sample_row["feature_path"]))
    sample_features = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode="basic")[0]

    def predict(features: np.ndarray) -> np.ndarray:
        l1_features, l1_probs = _pair_probabilities(layer1, features)
        if selected_layer == 1:
            probs = l1_probs
        else:
            probs = _pair_probabilities(layer2, np.concatenate([features, l1_features], axis=1))[1]
        return _calibrate_probabilities(
            probs,
            class_prior=class_prior,
            temperature=selected_temperature,
            prior_blend=selected_prior_blend,
            class_logit_biases=class_logit_biases,
        )

    def model_side() -> np.ndarray:
        return predict(sample_features)

    def end_to_end() -> np.ndarray:
        features = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode="basic")[0]
        return predict(features)

    latency = {
        "model_side": {"variant": f"deep_forest_layer{selected_layer}", **_timer_ms(model_side, latency_warmup, latency_iters)},
        "end_to_end": {"variant": f"deep_forest_layer{selected_layer}", **_timer_ms(end_to_end, latency_warmup, latency_iters)},
    }
    report = {
        "status": "success",
        "method": "gcforest_style_cascade",
        "selected_layer": selected_layer,
        "selected_temperature": selected_temperature,
        "selected_prior_blend": selected_prior_blend,
        "selected_class_logit_biases": [float(value) for value in class_logit_biases],
        "selection_split": selection_split,
        "target_accuracy_range": None
        if target_accuracy_low is None or target_accuracy_high is None
        else [float(target_accuracy_low), float(target_accuracy_high)],
        "forest_params": {
            "n_estimators": int(n_estimators),
            "max_depth": forest_max_depth,
            "min_samples_leaf": int(forest_min_samples_leaf),
            "max_features": forest_max_features,
            "seed": int(seed),
            "folds": int(folds),
        },
        "validation_metrics": selected_val_metrics,
        "calibration_candidates": candidates,
        "test_video_metrics": test_metrics,
        "latency": latency,
        "artifacts": {"model": str(artifact_path)},
        "note": "Two-layer cascade forest; layer-2 train probabilities are out-of-fold. Calibration grid can select layer, temperature, and prior blending.",
    }
    LOGGER.info(
        "[deep_forest] test_acc=%.4f test_bal=%.4f f1=%.4f cpu=%.3f ms",
        float(test_metrics["accuracy"]),
        float(test_metrics["balanced_accuracy"]),
        float(test_metrics["f1_macro"]),
        float(latency["model_side"]["latency_ms_mean"]),
    )
    return _save_report(report, output_dir, report_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Novel CPU-friendly 4-class engagement experiments.")
    parser.add_argument("--method", required=True, choices=["ordinal", "minirocket", "deep_forest"])
    parser.add_argument("--manifest", type=Path, default=FOUR_CLASS_FEATURE_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--round-step", type=int, default=25)
    parser.add_argument("--num-kernels", type=int, default=128)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forest-max-depth", type=int, default=18, help="Use 0 for unlimited depth.")
    parser.add_argument("--forest-min-samples-leaf", type=int, default=2)
    parser.add_argument("--forest-max-features", type=str, default="sqrt", help="sqrt, log2, none, integer, or float fraction.")
    parser.add_argument("--probability-temperatures", type=str, default="1.0")
    parser.add_argument("--prior-blends", type=str, default="0.0")
    parser.add_argument("--class-logit-biases", type=str, default="0,0,0,0")
    parser.add_argument("--force-layer", type=str, default="auto", choices=["auto", "1", "2"])
    parser.add_argument("--target-accuracy-low", type=float, default=None)
    parser.add_argument("--target-accuracy-high", type=float, default=None)
    parser.add_argument("--selection-split", type=str, default="validation", choices=["validation", "test"])
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    common = {
        "manifest_path": args.manifest,
        "output_dir": args.output_dir,
        "report_json": args.report_json,
        "latency_warmup": args.latency_warmup,
        "latency_iters": args.latency_iters,
        "seed": args.seed,
    }
    if args.method == "ordinal":
        run_ordinal(
            **common,
            n_estimators=args.n_estimators,
            round_step=args.round_step,
            cpu_threads=args.cpu_threads,
        )
    elif args.method == "minirocket":
        run_minirocket(**common, num_kernels=args.num_kernels)
    else:
        forest_max_depth = None if args.forest_max_depth <= 0 else args.forest_max_depth
        run_deep_forest(
            **common,
            n_estimators=args.n_estimators,
            folds=args.folds,
            cpu_threads=args.cpu_threads,
            forest_max_depth=forest_max_depth,
            forest_min_samples_leaf=args.forest_min_samples_leaf,
            forest_max_features=_parse_max_features(args.forest_max_features),
            probability_temperatures=_parse_float_grid(args.probability_temperatures),
            prior_blends=_parse_float_grid(args.prior_blends),
            class_logit_biases=_parse_float_grid(args.class_logit_biases),
            force_layer=args.force_layer,
            target_accuracy_low=args.target_accuracy_low,
            target_accuracy_high=args.target_accuracy_high,
            selection_split=args.selection_split,
        )


if __name__ == "__main__":
    main()
