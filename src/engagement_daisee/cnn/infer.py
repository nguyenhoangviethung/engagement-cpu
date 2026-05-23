import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from engagement_daisee.cnn.model import build_cnn_model
from engagement_daisee.cnn.train import _build_transform


def run_infer(
    checkpoint_path: Path,
    image_path: Path,
    threshold: float | None,
    image_size: int | None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = str(checkpoint.get("model_name", "mobilenet_v3_small"))
    resolved_image_size = int(image_size) if image_size is not None else int(checkpoint.get("image_size", 112))

    model = build_cnn_model(model_name=model_name, pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to("cpu").eval()

    transform = _build_transform(image_size=resolved_image_size, train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probability = float(torch.sigmoid(logits).view(-1)[0].item())

    default_threshold = float(checkpoint.get("best_threshold", 0.5))
    resolved_threshold = float(threshold) if threshold is not None else default_threshold
    prediction = int(probability >= resolved_threshold)

    return {
        "checkpoint": str(checkpoint_path),
        "image": str(image_path),
        "model_name": model_name,
        "image_size": resolved_image_size,
        "threshold": resolved_threshold,
        "probability": probability,
        "prediction": prediction,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image CNN inference.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="CNN checkpoint (.pt)")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_infer(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        threshold=args.threshold,
        image_size=args.image_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
