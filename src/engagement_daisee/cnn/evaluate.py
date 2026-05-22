import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from engagement_daisee.cnn.dataset import DAiSEECNNFrameDataset
from engagement_daisee.cnn.model import build_cnn_model
from engagement_daisee.cnn.train import _build_transform, _compute_metrics, _run_epoch


DEFAULT_MANIFEST = Path("data/processed/cnn_frame_manifest.csv")


def _split_indices(manifest: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    if "split" not in manifest.columns:
        raise ValueError("Manifest must include split column.")

    split_series = manifest["split"].astype(str).str.strip().str.lower()
    train_indices = split_series[split_series == "train"].index.tolist()
    val_indices = split_series[split_series == "validation"].index.tolist()
    test_indices = split_series[split_series == "test"].index.tolist()

    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            "Official split requires non-empty train/validation/test rows. "
            f"Got train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}"
        )

    return train_indices, val_indices, test_indices


def _build_loader(dataset: DAiSEECNNFrameDataset, indices: list[int], batch_size: int) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def run_eval(
    manifest_path: Path,
    checkpoint_path: Path,
    split: str,
    batch_size: int,
    threshold: float | None,
    output_json: Path | None,
) -> dict:
    manifest_df = pd.read_csv(manifest_path)
    train_indices, val_indices, test_indices = _split_indices(manifest_df)

    split_name = split.strip().lower()
    split_map = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices,
    }
    if split_name not in split_map:
        raise ValueError(f"Unsupported split: {split}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = str(checkpoint.get("model_name", "mobilenet_v3_small"))
    image_size = int(checkpoint.get("image_size", 112))
    default_threshold = float(checkpoint.get("best_threshold", 0.5))
    resolved_threshold = float(threshold) if threshold is not None else default_threshold

    model = build_cnn_model(model_name=model_name, pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to("cpu").eval()

    dataset = DAiSEECNNFrameDataset(manifest_csv=manifest_path, transform=_build_transform(image_size=image_size, train=False))
    loader = _build_loader(dataset=dataset, indices=split_map[split_name], batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss()
    loss, labels, probs = _run_epoch(model=model, loader=loader, criterion=criterion, device="cpu", optimizer=None)
    metrics = _compute_metrics(labels=labels, probs=probs, threshold=resolved_threshold)

    report = {
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "split": split_name,
        "rows": int(len(split_map[split_name])),
        "batch_size": int(batch_size),
        "model_name": model_name,
        "image_size": image_size,
        "threshold": resolved_threshold,
        "loss": float(loss),
        "metrics": metrics,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CNN checkpoint on DAiSEE official split.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Frame manifest CSV")
    parser.add_argument("--checkpoint", type=Path, required=True, help="CNN checkpoint (.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional report output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        threshold=args.threshold,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
