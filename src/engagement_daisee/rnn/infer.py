import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_sequence(path: Path) -> torch.Tensor:
    array = np.load(path)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected sequence shape (T, F) or (B, T, F), got {array.shape}")
    return torch.as_tensor(array, dtype=torch.float32)


def run_inference(artifact: Path, sequence: torch.Tensor, cpu_threads: int) -> np.ndarray:
    torch.set_num_threads(max(1, cpu_threads))
    model = torch.jit.load(str(artifact), map_location="cpu")
    model.eval()
    with torch.no_grad():
        probs = model(sequence)
    return probs.detach().cpu().numpy().reshape(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU inference from optimized TorchScript artifact.")
    parser.add_argument("--artifact", type=Path, required=True, help="Path to .ts artifact")
    parser.add_argument("--sequence", type=Path, required=True, help="Path to sequence .npy")
    parser.add_argument("--meta", type=Path, default=None, help="Optional inference_meta.json for threshold")
    parser.add_argument("--cpu-threads", type=int, default=2, help="CPU threads for inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = load_sequence(args.sequence)
    probs = run_inference(args.artifact, sequence, cpu_threads=args.cpu_threads)

    threshold = 0.5
    if args.meta is not None and args.meta.exists():
        meta = json.loads(args.meta.read_text())
        threshold = float(meta.get("threshold", threshold))

    preds = (probs >= threshold).astype(np.int64)
    print(json.dumps({"threshold": threshold, "probabilities": probs.tolist(), "predictions": preds.tolist()}, indent=2))


if __name__ == "__main__":
    main()
