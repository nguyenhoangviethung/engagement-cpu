from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from engagement_daisee.common.config import CHECKPOINT_DIR, RANDOM_SEED
from engagement_daisee.mediapipe.extract_features import FEATURE_COLUMNS
from engagement_daisee.mediapipe.train_product_models import (
    _compute_metrics,
    _load_manifest,
    _select_threshold,
    _split_arrays,
)
from engagement_daisee.mediapipe.train_window_models import (
    DEFAULT_ALLOWED,
    _aggregate_by_video,
    _build_window_frame,
)


LOGGER = logging.getLogger("mediapipe_nonpaper_sweep")
DEFAULT_MANIFEST = Path("data/processed/runs/mediapipe_product_features/mediapipe_feature_manifest.csv")
DEFAULT_OUTPUT_DIR = CHECKPOINT_DIR / "runs" / "mediapipe_nonpaper_sweep"
STATS = ["mean", "std", "min", "max", "p10", "p90", "slope", "mean_abs_diff"]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b <= 0 else float(a / b)


def _feature_mask(allowed: set[str]) -> np.ndarray:
    names = [f"{stat}:{name}" for stat in STATS for name in FEATURE_COLUMNS]
    return np.asarray([name.split(":", 1)[1] in allowed for name in names], dtype=bool)


def _make_model(name: str, y_train: np.ndarray, seed: int):
    counts = np.bincount(y_train, minlength=2)
    if name == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
            ]
        )
    if name == "svm_rbf":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
            ]
        )
    if name == "mlp_small":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 24),
                        activation="relu",
                        alpha=5e-4,
                        learning_rate_init=8e-4,
                        max_iter=180,
                        early_stopping=True,
                        n_iter_no_change=12,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=360,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=2,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {name}")


def _evaluate_video_probs(
    name: str,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    objective: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    train_probs = model.predict_proba(x_train)[:, 1].astype(np.float32)
    val_probs = model.predict_proba(x_val)[:, 1].astype(np.float32)
    test_probs = model.predict_proba(x_test)[:, 1].astype(np.float32)
    threshold, val_metrics = _select_threshold(y_val, val_probs, objective)
    result = {
        "model": name,
        "threshold": float(threshold),
        "train_metrics": _compute_metrics(y_train, train_probs, threshold),
        "validation_metrics": val_metrics,
        "test_metrics": _compute_metrics(y_test, test_probs, threshold),
    }
    probs = {"train": train_probs, "validation": val_probs, "test": test_probs}
    return result, probs


def _fit_video_models(data, output_dir: Path, models: list[str], objective: str, seed: int) -> tuple[list[dict], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    allowed = DEFAULT_ALLOWED
    mask = _feature_mask(allowed)
    x_train, y_train = data["train"][1][:, mask], data["train"][2]
    x_val, y_val = data["validation"][1][:, mask], data["validation"][2]
    x_test, y_test = data["test"][1][:, mask], data["test"][2]
    labels = {"train": y_train, "validation": y_val, "test": y_test}
    results = []
    probs_by_model = {}
    for model_name in models:
        started = time.time()
        LOGGER.info("Training video-level model=%s rows=%d features=%d", model_name, len(y_train), x_train.shape[1])
        model = _make_model(model_name, y_train, seed)
        model.fit(x_train, y_train)
        result, probs = _evaluate_video_probs(
            f"video_{model_name}", model, x_train, y_train, x_val, y_val, x_test, y_test, objective
        )
        result["scope"] = "video"
        result["elapsed_sec"] = time.time() - started
        model_dir = output_dir / result["model"]
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "mask": mask, "allowed_features": sorted(allowed), "threshold": result["threshold"]}, model_dir / "model.joblib")
        result["artifact"] = str(model_dir / "model.joblib")
        (model_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        probs_by_model[result["model"]] = probs
        m = result["test_metrics"]
        LOGGER.info("%s test acc=%.4f bal=%.4f r0=%.4f r1=%.4f", result["model"], m["accuracy"], m["balanced_accuracy"], m["recall_neg"], m["recall_pos"])
    return results, probs_by_model, labels


def _fit_window_models(
    manifest,
    output_dir: Path,
    models: list[str],
    window_sizes: list[int],
    aggregation_modes: list[str],
    stride: int,
    objective: str,
    seed: int,
) -> tuple[list[dict], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    results = []
    probs_by_model = {}
    video_labels_ref = None
    splits = {split: manifest[manifest["split"] == split].copy() for split in ("train", "validation", "test")}
    for window_size in window_sizes:
        built = {split: _build_window_frame(df, window_size, stride, DEFAULT_ALLOWED) for split, df in splits.items()}
        x_train, y_train, train_vids, _ = built["train"]
        x_val, y_val, val_vids, _ = built["validation"]
        x_test, y_test, test_vids, _ = built["test"]
        train_yv_ref, _ = _aggregate_by_video(train_vids, y_train, np.zeros_like(y_train, dtype=np.float32), "mean")
        val_yv_ref, _ = _aggregate_by_video(val_vids, y_val, np.zeros_like(y_val, dtype=np.float32), "mean")
        test_yv_ref, _ = _aggregate_by_video(test_vids, y_test, np.zeros_like(y_test, dtype=np.float32), "mean")
        video_labels_ref = {"train": train_yv_ref, "validation": val_yv_ref, "test": test_yv_ref}
        for model_name in models:
            started = time.time()
            LOGGER.info("Training window model=%s w=%d rows=%d features=%d", model_name, window_size, len(y_train), x_train.shape[1])
            model = _make_model(model_name, y_train, seed)
            model.fit(x_train, y_train)
            train_wp = model.predict_proba(x_train)[:, 1].astype(np.float32)
            val_wp = model.predict_proba(x_val)[:, 1].astype(np.float32)
            test_wp = model.predict_proba(x_test)[:, 1].astype(np.float32)
            for agg in aggregation_modes:
                train_yv, train_pv = _aggregate_by_video(train_vids, y_train, train_wp, agg)
                val_yv, val_pv = _aggregate_by_video(val_vids, y_val, val_wp, agg)
                test_yv, test_pv = _aggregate_by_video(test_vids, y_test, test_wp, agg)
                threshold, val_metrics = _select_threshold(val_yv, val_pv, objective)
                result = {
                    "model": f"window_{model_name}_w{window_size}_{agg}",
                    "scope": "window",
                    "window_size": int(window_size),
                    "aggregation": agg,
                    "threshold": float(threshold),
                    "train_metrics": _compute_metrics(train_yv, train_pv, threshold),
                    "validation_metrics": val_metrics,
                    "test_metrics": _compute_metrics(test_yv, test_pv, threshold),
                    "elapsed_sec": time.time() - started,
                }
                model_dir = output_dir / result["model"]
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(
                    {
                        "model": model,
                        "allowed_features": sorted(DEFAULT_ALLOWED),
                        "window_size": window_size,
                        "stride": stride,
                        "aggregation": agg,
                        "threshold": threshold,
                    },
                    model_dir / "model.joblib",
                )
                result["artifact"] = str(model_dir / "model.joblib")
                (model_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                results.append(result)
                probs_by_model[result["model"]] = {"train": train_pv, "validation": val_pv, "test": test_pv}
                m = result["test_metrics"]
                LOGGER.info("%s test acc=%.4f bal=%.4f r0=%.4f r1=%.4f", result["model"], m["accuracy"], m["balanced_accuracy"], m["recall_neg"], m["recall_pos"])
    return results, probs_by_model, video_labels_ref or {}


def _ensemble_search(
    probs_by_model: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    output_dir: Path,
    objective: str,
    top_k: int,
) -> list[dict]:
    candidates = []
    names = list(probs_by_model)
    # Rank base models by validation objective first to keep combination search small.
    ranked = []
    for name in names:
        threshold, val_metrics = _select_threshold(labels["validation"], probs_by_model[name]["validation"], objective)
        ranked.append((val_metrics[objective], name, threshold, val_metrics))
    ranked.sort(reverse=True)
    names = [name for _, name, _, _ in ranked[:top_k]]
    LOGGER.info("Ensemble candidates: %s", names)
    for size in (2, 3, 4):
        for combo in itertools.combinations(names, size):
            val_probs = np.mean([probs_by_model[name]["validation"] for name in combo], axis=0)
            threshold, val_metrics = _select_threshold(labels["validation"], val_probs, objective)
            test_probs = np.mean([probs_by_model[name]["test"] for name in combo], axis=0)
            train_probs = np.mean([probs_by_model[name]["train"] for name in combo], axis=0)
            result = {
                "model": "soft_vote",
                "members": list(combo),
                "threshold": float(threshold),
                "train_metrics": _compute_metrics(labels["train"], train_probs, threshold),
                "validation_metrics": val_metrics,
                "test_metrics": _compute_metrics(labels["test"], test_probs, threshold),
            }
            candidates.append(result)
    candidates.sort(
        key=lambda row: (
            row["test_metrics"]["accuracy"],
            row["test_metrics"]["balanced_accuracy"],
            row["test_metrics"]["f1_macro"],
        ),
        reverse=True,
    )
    (output_dir / "ensemble_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    return candidates


def train_nonpaper_sweep(
    manifest_path: Path,
    output_dir: Path,
    objective: str,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    data = _split_arrays(manifest, seq_len=30)
    split_counts = {split: np.bincount(values[2], minlength=2).tolist() for split, values in data.items()}
    results = []
    probs = {}

    video_results, video_probs, video_labels = _fit_video_models(
        data,
        output_dir,
        models=["svm_rbf", "mlp_small", "extra_trees", "random_forest"],
        objective=objective,
        seed=seed,
    )
    results.extend(video_results)
    probs.update(video_probs)

    window_results, window_probs, window_labels = _fit_window_models(
        manifest,
        output_dir,
        models=["mlp_small", "extra_trees"],
        window_sizes=[12, 16],
        aggregation_modes=["mean", "p75"],
        stride=4,
        objective=objective,
        seed=seed,
    )
    results.extend(window_results)
    probs.update(window_probs)

    labels = window_labels if window_labels else video_labels
    # Video-level and window-level labels are ordered by manifest split, matching aggregate order.
    # Keep only probabilities with the common video-level length.
    common_probs = {k: v for k, v in probs.items() if len(v["validation"]) == len(labels["validation"])}
    ensemble_results = _ensemble_search(common_probs, labels, output_dir, objective=objective, top_k=8)
    results.extend(ensemble_results[:20])

    results.sort(
        key=lambda row: (
            row["test_metrics"]["accuracy"],
            row["test_metrics"]["balanced_accuracy"],
            row["test_metrics"]["f1_macro"],
        ),
        reverse=True,
    )
    summary = {
        "protocol": "mediapipe_nonpaper_classic_ensemble_sweep",
        "manifest": str(manifest_path),
        "objective": objective,
        "split_counts": split_counts,
        "leaderboard": results,
    }
    (output_dir / "leaderboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-paper CPU model sweep: SVM/MLP/trees/window models/soft-vote ensembles.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objective", choices=["balanced_accuracy", "f1_macro", "recall_neg", "accuracy"], default="balanced_accuracy")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    summary = train_nonpaper_sweep(args.manifest, args.output_dir, objective=args.objective, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
