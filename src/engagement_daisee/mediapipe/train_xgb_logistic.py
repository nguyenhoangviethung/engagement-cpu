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
from xgboost import XGBClassifier

from engagement_daisee.common.config import CHECKPOINT_DIR, RANDOM_SEED
from engagement_daisee.mediapipe.extract_features import FEATURE_COLUMNS
from engagement_daisee.mediapipe.train_product_models import (
    _benchmark_callable,
    _compute_metrics,
    _load_manifest,
    _select_threshold,
    _split_arrays,
)


LOGGER = logging.getLogger("mediapipe_xgb_logistic")
DEFAULT_MANIFEST = Path("data/processed/runs/mediapipe_product_features/mediapipe_feature_manifest.csv")
DEFAULT_OUTPUT_DIR = CHECKPOINT_DIR / "runs" / "mediapipe_xgb_logistic"
STATS = ["mean", "std", "min", "max", "p10", "p90", "slope", "mean_abs_diff"]


VARIANTS = {
    "all": set(FEATURE_COLUMNS),
    "no_absolute_position": {
        "face_present",
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
    },
    "behavior_plus_size": {
        "face_present",
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
        "face_width",
        "face_height",
    },
    "eyes_head_mouth": {
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
    },
}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _feature_names() -> list[str]:
    return [f"{stat}:{name}" for stat in STATS for name in FEATURE_COLUMNS]


def _variant_mask(variant: str) -> tuple[np.ndarray, list[str]]:
    allowed = VARIANTS[variant]
    names = _feature_names()
    mask = np.asarray([name.split(":", 1)[1] in allowed for name in names], dtype=bool)
    return mask, [name for name, keep in zip(names, mask) if keep]


def _safe_div(a: float, b: float) -> float:
    return 0.0 if b <= 0 else float(a / b)


def _top_coefficients(model: Pipeline, selected_names: list[str], top_k: int = 25) -> list[dict[str, object]]:
    coef = model.named_steps["clf"].coef_.reshape(-1)
    order = np.argsort(np.abs(coef))[::-1][:top_k]
    return [{"feature": selected_names[i], "coefficient": float(coef[i])} for i in order]


def _coef_group_summary(model: Pipeline, selected_names: list[str]) -> dict[str, dict[str, float]]:
    coef = model.named_steps["clf"].coef_.reshape(-1)
    by_base: dict[str, float] = {}
    by_stat: dict[str, float] = {}
    for name, value in zip(selected_names, coef):
        stat, base = name.split(":", 1)
        by_base[base] = by_base.get(base, 0.0) + abs(float(value))
        by_stat[stat] = by_stat.get(stat, 0.0) + abs(float(value))
    return {
        "by_base_feature": dict(sorted(by_base.items(), key=lambda kv: kv[1], reverse=True)),
        "by_temporal_stat": dict(sorted(by_stat.items(), key=lambda kv: kv[1], reverse=True)),
    }


def _train_logistic_variant(data, mask: np.ndarray, names: list[str], out_dir: Path, variant: str, objective: str, seed: int) -> dict:
    x_train, y_train = data["train"][1][:, mask], data["train"][2]
    x_val, y_val = data["validation"][1][:, mask], data["validation"][2]
    x_test, y_test = data["test"][1][:, mask], data["test"][2]
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ]
    )
    started = time.time()
    model.fit(x_train, y_train)
    threshold, val_metrics = _select_threshold(y_val, model.predict_proba(x_val)[:, 1], objective)
    test_probs = model.predict_proba(x_test)[:, 1]
    test_metrics = _compute_metrics(y_test, test_probs, threshold)
    train_metrics = _compute_metrics(y_train, model.predict_proba(x_train)[:, 1], threshold)
    latency = _benchmark_callable(lambda x: model.predict_proba(x.reshape(1, -1))[:, 1], x_test[0])

    model_dir = out_dir / f"logistic_{variant}"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "mask": mask, "feature_names": names, "threshold": threshold, "variant": variant}, model_dir / "model.joblib")
    result = {
        "model": "logistic",
        "variant": variant,
        "num_features": int(mask.sum()),
        "threshold": float(threshold),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "latency_ms": float(latency),
        "elapsed_sec": time.time() - started,
        "artifact": str(model_dir / "model.joblib"),
        "top_coefficients": _top_coefficients(model, names),
        "coefficient_groups": _coef_group_summary(model, names),
    }
    (model_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _train_xgb_variant(data, mask: np.ndarray, names: list[str], out_dir: Path, variant: str, objective: str, seed: int) -> dict:
    x_train, y_train = data["train"][1][:, mask], data["train"][2]
    x_val, y_val = data["validation"][1][:, mask], data["validation"][2]
    x_test, y_test = data["test"][1][:, mask], data["test"][2]
    counts = np.bincount(y_train, minlength=2)
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=220,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=8,
        reg_lambda=3.0,
        reg_alpha=0.05,
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
        eval_metric="logloss",
        scale_pos_weight=_safe_div(float(counts[0]), float(counts[1])) if counts[1] else 1.0,
    )
    started = time.time()
    model.fit(x_train, y_train, verbose=False)
    threshold, val_metrics = _select_threshold(y_val, model.predict_proba(x_val)[:, 1], objective)
    test_probs = model.predict_proba(x_test)[:, 1]
    test_metrics = _compute_metrics(y_test, test_probs, threshold)
    train_metrics = _compute_metrics(y_train, model.predict_proba(x_train)[:, 1], threshold)
    latency = _benchmark_callable(lambda x: model.predict_proba(x.reshape(1, -1))[:, 1], x_test[0])

    model_dir = out_dir / f"xgboost_{variant}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(model_dir / "model.json"))
    metadata = {"mask": mask.astype(int).tolist(), "feature_names": names, "threshold": threshold, "variant": variant}
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = {
        "model": "xgboost_shallow",
        "variant": variant,
        "num_features": int(mask.sum()),
        "threshold": float(threshold),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "latency_ms": float(latency),
        "elapsed_sec": time.time() - started,
        "artifact": str(model_dir / "model.json"),
    }
    (model_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_xgb_logistic(
    manifest_path: Path,
    output_dir: Path,
    seq_len: int,
    objective: str,
    variants: list[str],
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    data = _split_arrays(manifest, seq_len=seq_len)
    split_counts = {
        split: np.bincount(values[2], minlength=2).astype(int).tolist()
        for split, values in data.items()
    }
    LOGGER.info("Loaded MediaPipe data | manifest=%s split_counts=%s", manifest_path, split_counts)

    results = []
    for variant in variants:
        mask, selected_names = _variant_mask(variant)
        LOGGER.info("Training variant=%s selected_features=%d", variant, int(mask.sum()))
        results.append(_train_logistic_variant(data, mask, selected_names, output_dir, variant, objective, seed))
        results.append(_train_xgb_variant(data, mask, selected_names, output_dir, variant, objective, seed))

    results = sorted(
        results,
        key=lambda item: (
            item["test_metrics"]["balanced_accuracy"],
            item["test_metrics"]["f1_macro"],
            -item["latency_ms"],
        ),
        reverse=True,
    )
    summary = {
        "protocol": "mediapipe_xgb_logistic_product_ablation",
        "manifest": str(manifest_path),
        "seq_len": int(seq_len),
        "objective": objective,
        "split_counts": split_counts,
        "variants": variants,
        "leaderboard": results,
    }
    (output_dir / "leaderboard.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train focused XGBoost shallow + logistic explainable MediaPipe product models.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--objective", choices=["balanced_accuracy", "f1_macro", "recall_neg"], default="balanced_accuracy")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    summary = train_xgb_logistic(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        objective=args.objective,
        variants=args.variants,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
