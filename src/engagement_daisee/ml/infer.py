import argparse
import json
from pathlib import Path

import numpy as np

from engagement_daisee.ml.evaluate import _load_model, _predict_probabilities, _resolve_threshold
from engagement_daisee.ml.train import _sequence_to_tabular_features


def run_infer(
    model_path: Path,
    sequence_path: Path,
    feature_mode: str,
    threshold: float | None,
    summary_json: Path | None,
) -> dict:
    sequence = np.load(sequence_path)
    features = _sequence_to_tabular_features(sequence=sequence, feature_mode=feature_mode).reshape(1, -1)

    backend, model = _load_model(model_path)
    probability = float(_predict_probabilities(backend, model, features)[0])
    resolved_threshold = _resolve_threshold(threshold=threshold, summary_json=summary_json)

    return {
        "model": str(model_path),
        "sequence": str(sequence_path),
        "backend": backend,
        "feature_mode": feature_mode,
        "threshold": resolved_threshold,
        "probability": probability,
        "prediction": int(probability >= resolved_threshold),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for tree-based ML model on one sequence file.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model file")
    parser.add_argument("--sequence", type=Path, required=True, help="Path to .npy sequence file")
    parser.add_argument("--feature-mode", type=str, default="tsfresh", choices=["basic", "tsfresh", "copur"])
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional summary json with selected_threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_infer(
        model_path=args.model,
        sequence_path=args.sequence,
        feature_mode=args.feature_mode,
        threshold=args.threshold,
        summary_json=args.summary_json,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
