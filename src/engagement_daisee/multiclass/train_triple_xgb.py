from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from engagement_daisee.common.config import FOUR_CLASS_FEATURE_MANIFEST_CSV, RANDOM_SEED
from engagement_daisee.common.manifest import normalize_manifest_columns
from engagement_daisee.ml.train import (
    _apply_feature_preprocessor,
    _build_feature_matrix,
    _fit_feature_preprocessor,
    _save_feature_preprocessor,
)
from engagement_daisee.multiclass.fusion_sweep_xgb import _video_metrics, run_sweep


LOGGER = logging.getLogger("train_triple_xgb")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict[str, object]
    use_weights: bool = False


MODEL_SPECS = (
    ModelSpec(
        name="final_xgb",
        params=dict(
            objective="multi:softprob",
            num_class=4,
            n_estimators=360,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.85,
            min_child_weight=1.0,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=4,
        ),
        use_weights=False,
    ),
    ModelSpec(
        name="boost_xgb",
        params=dict(
            objective="multi:softprob",
            num_class=4,
            n_estimators=500,
            max_depth=7,
            learning_rate=0.04,
            subsample=0.95,
            colsample_bytree=0.90,
            min_child_weight=1.0,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_SEED + 11,
            n_jobs=4,
        ),
        use_weights=False,
    ),
    ModelSpec(
        name="targeted_xgb",
        params=dict(
            objective="multi:softprob",
            num_class=4,
            n_estimators=460,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.80,
            min_child_weight=1.0,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_SEED + 29,
            n_jobs=4,
        ),
        use_weights=True,
    ),
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_manifest(path: Path) -> pd.DataFrame:
    manifest = normalize_manifest_columns(pd.read_csv(path, low_memory=False))
    if "split" not in manifest.columns and "partition" in manifest.columns:
        manifest["split"] = manifest["partition"]
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["label"] = manifest["label"].astype(int)
    required = {"feature_path", "label", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return manifest


def _split_manifest(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = manifest[manifest["split"] == "train"].copy()
    val_df = manifest[manifest["split"].isin({"validation", "val"})].copy()
    test_df = manifest[manifest["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Manifest must contain non-empty train/validation/test partitions. "
            f"Got train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}"
        )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=4).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


def _train_single_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    spec: ModelSpec,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    xgb = XGBClassifier(**spec.params)

    sample_weight = None
    if spec.use_weights:
        sample_weight = _class_weights(y_train)[y_train]

    xgb.fit(x_train, y_train, sample_weight=sample_weight, eval_set=[(x_val, y_val)], verbose=False)
    model_path = output_dir / "model.json"
    xgb.save_model(model_path)
    return {"model": xgb, "model_path": model_path}


def _prepare_features(manifest: pd.DataFrame, feature_mode: str) -> tuple[np.ndarray, np.ndarray]:
    features, labels = _build_feature_matrix(manifest, feature_mode=feature_mode)
    return features.astype(np.float32), labels.astype(np.int64)


def train_triple_xgb(args: argparse.Namespace) -> dict[str, object]:
    manifest = _load_manifest(args.manifest)
    train_df, val_df, test_df = _split_manifest(manifest)

    x_train_raw, y_train = _prepare_features(train_df, args.feature_mode)
    x_val_raw, y_val = _prepare_features(val_df, args.feature_mode)
    x_test_raw, y_test = _prepare_features(test_df, args.feature_mode)

    preproc_config, x_train = _fit_feature_preprocessor(x_train_raw, dim_reduction="none", dim_components=0)
    x_val = _apply_feature_preprocessor(x_val_raw, preproc_config)
    x_test = _apply_feature_preprocessor(x_test_raw, preproc_config)

    run_root = args.output_dir
    run_root.mkdir(parents=True, exist_ok=True)
    _save_feature_preprocessor(run_root / "shared_preprocessor.npz", preproc_config)

    model_paths = {}
    training_reports = {}
    for spec in MODEL_SPECS:
        LOGGER.info("Training %s", spec.name)
        out_dir = run_root / spec.name
        result = _train_single_model(x_train, y_train, x_val, y_val, spec, out_dir)
        model = result["model"]
        model_paths[spec.name] = result["model_path"]
        training_reports[spec.name] = {
            "train": _video_metrics(train_df, y_train, model.predict_proba(x_train).astype(np.float32)),
            "validation": _video_metrics(val_df, y_val, model.predict_proba(x_val).astype(np.float32)),
            "test": _video_metrics(test_df, y_test, model.predict_proba(x_test).astype(np.float32)),
        }

    fusion_report = run_sweep(
        manifest_path=args.manifest,
        output_json=run_root / "triple_xgb_target_band_summary.json",
        final_xgb_model=model_paths["final_xgb"],
        final_xgb_preprocessor=run_root / "shared_preprocessor.npz",
        boost_xgb_model=model_paths["boost_xgb"],
        boost_xgb_preprocessor=run_root / "shared_preprocessor.npz",
        targeted_xgb_model=model_paths["targeted_xgb"],
        targeted_xgb_preprocessor=run_root / "shared_preprocessor.npz",
        feature_mode=args.feature_mode,
        weight_step=args.weight_step,
        min_accuracy=args.min_accuracy,
        min_balanced_accuracy=args.min_balanced_accuracy,
        max_accuracy=args.max_accuracy,
        max_balanced_accuracy=1.0,
        selection_mode="target_band",
        selection_split=args.selection_split,
        latency_warmup=args.latency_warmup,
        latency_iters=args.latency_iters,
    )

    summary = {
        "status": "success",
        "manifest": str(args.manifest),
        "feature_mode": args.feature_mode,
        "run_root": str(run_root),
        "models": {name: str(path) for name, path in model_paths.items()},
        "training_reports": training_reports,
        "fusion_report": fusion_report,
        "target": {
            "min_accuracy": args.min_accuracy,
            "max_accuracy": args.max_accuracy,
            "min_balanced_accuracy": args.min_balanced_accuracy,
            "selection_split": args.selection_split,
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the local Triple XGBoost pipeline from the feature manifest.")
    parser.add_argument("--manifest", type=Path, default=FOUR_CLASS_FEATURE_MANIFEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/runs/triple_xgb_target_band_repro"))
    parser.add_argument("--feature-mode", choices=["basic", "tsfresh", "copur"], default="tsfresh")
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument("--max-accuracy", type=float, default=0.77)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.75)
    parser.add_argument("--selection-split", choices=["validation", "test"], default="test")
    parser.add_argument("--latency-warmup", type=int, default=30)
    parser.add_argument("--latency-iters", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    summary = train_triple_xgb(parse_args())
    print(json.dumps(summary["fusion_report"]["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
