import argparse
import json
from pathlib import Path

import numpy as np

from engagement_daisee.common.config import FEATURE_MANIFEST_CSV
from engagement_daisee.ml.train import (
    _apply_feature_preprocessor,
    _build_feature_matrix,
    _compute_metrics,
    _load_feature_preprocessor,
    _load_manifest,
)

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


class MLModelLoadError(RuntimeError):
    pass


def _load_model(model_path: Path):
    errors: list[str] = []

    if xgb is not None:
        try:
            model = xgb.Booster()
            model.load_model(str(model_path))
            return "xgboost", model
        except Exception as exc:  # pragma: no cover
            errors.append(f"xgboost: {exc}")

    if lgb is not None:
        try:
            model = lgb.Booster(model_file=str(model_path))
            return "lightgbm", model
        except Exception as exc:  # pragma: no cover
            errors.append(f"lightgbm: {exc}")

    raise MLModelLoadError(
        f"Unable to load model at {model_path}. Errors: {errors if errors else 'No backend available.'}"
    )


def _predict_probabilities(model_backend: str, model, features: np.ndarray) -> np.ndarray:
    if model_backend == "xgboost":
        return model.predict(xgb.DMatrix(features)).astype(np.float32)
    return np.asarray(model.predict(features), dtype=np.float32)


def _resolve_threshold(threshold: float | None, summary_json: Path | None) -> float:
    if threshold is not None:
        return float(threshold)
    if summary_json is not None and summary_json.exists():
        payload = json.loads(summary_json.read_text())
        if "selected_threshold" in payload:
            return float(payload["selected_threshold"])
    return 0.5


def _resolve_preprocessor(summary_json: Path | None) -> dict | None:
    if summary_json is None or not summary_json.exists():
        return None
    payload = json.loads(summary_json.read_text())
    path_raw = payload.get("preprocessor_path")
    if not path_raw:
        return None
    preprocessor_path = Path(path_raw)
    if not preprocessor_path.exists():
        return None
    return _load_feature_preprocessor(preprocessor_path)


def _split_indices(split_series: np.ndarray, split_name: str) -> np.ndarray:
    indices = np.where(split_series == split_name)[0]
    if indices.size == 0:
        raise ValueError(f"Split '{split_name}' is empty in manifest.")
    return indices


def _aggregate_by_video(manifest_subset, labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "video_id" not in manifest_subset.columns:
        return labels, probabilities

    import pandas as pd

    frame = pd.DataFrame(
        {
            "video_id": manifest_subset["video_id"].astype(str).to_numpy(),
            "label": labels.astype(np.int64),
            "probability": probabilities.astype(np.float32),
        }
    )
    grouped = frame.groupby("video_id", sort=False).agg({"label": "max", "probability": "mean"})
    return grouped["label"].to_numpy(dtype=np.int64), grouped["probability"].to_numpy(dtype=np.float32)


def run_eval(
    manifest_path: Path,
    model_path: Path,
    split: str,
    threshold: float | None,
    summary_json: Path | None,
    feature_mode: str,
    output_json: Path | None,
    aggregation: str,
) -> dict:
    manifest = _load_manifest(manifest_path)

    split_name = split.strip().lower()
    split_series = manifest["split"].astype(str).str.lower().to_numpy()
    split_indices = _split_indices(split_series, split_name)

    eval_manifest = manifest.iloc[split_indices].reset_index(drop=True)
    x_eval, y_eval = _build_feature_matrix(eval_manifest, feature_mode=feature_mode)
    preprocessor = _resolve_preprocessor(summary_json)
    if preprocessor is not None:
        x_eval = _apply_feature_preprocessor(x_eval, preprocessor)

    backend, model = _load_model(model_path)
    probabilities = _predict_probabilities(backend, model, x_eval)

    resolved_threshold = _resolve_threshold(threshold=threshold, summary_json=summary_json)
    row_metrics = _compute_metrics(y_eval, probabilities, resolved_threshold)
    video_labels, video_probabilities = _aggregate_by_video(eval_manifest, y_eval, probabilities)
    video_metrics = _compute_metrics(video_labels, video_probabilities, resolved_threshold)
    selected_metrics = video_metrics if aggregation == "video" else row_metrics

    report = {
        "manifest": str(manifest_path),
        "model": str(model_path),
        "summary_json": str(summary_json) if summary_json is not None else None,
        "backend": backend,
        "split": split_name,
        "rows": int(len(eval_manifest)),
        "videos": int(len(video_labels)),
        "feature_mode": feature_mode,
        "threshold": resolved_threshold,
        "aggregation": aggregation,
        "metrics": selected_metrics,
        "row_metrics": row_metrics,
        "video_metrics": video_metrics,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tree-based ML model on DAiSEE manifest split.")
    parser.add_argument("--manifest", type=Path, default=FEATURE_MANIFEST_CSV)
    parser.add_argument("--model", type=Path, required=True, help="Path to XGBoost/LightGBM model file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--feature-mode", type=str, default="tsfresh", choices=["basic", "tsfresh", "copur"])
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional training summary json containing selected_threshold",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--aggregation", type=str, default="rows", choices=["rows", "video"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(
        manifest_path=args.manifest,
        model_path=args.model,
        split=args.split,
        threshold=args.threshold,
        summary_json=args.summary_json,
        feature_mode=args.feature_mode,
        output_json=args.output_json,
        aggregation=args.aggregation,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
