import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from engagement_daisee.common.config import CHECKPOINT_DIR, FEATURE_DIM, HIDDEN_SIZE, DROPOUT, SEQUENCE_LENGTH
from engagement_daisee.rnn.models.builder import build_sequence_model


class InferenceWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        feature_mean: torch.Tensor | None,
        feature_std: torch.Tensor | None,
        normalize: bool,
        input_feature_dim: int,
    ):
        super().__init__()
        self.model = model
        self.normalize = bool(normalize)

        if feature_mean is None:
            feature_mean = torch.zeros(1, 1, input_feature_dim, dtype=torch.float32)
        if feature_std is None:
            feature_std = torch.ones(1, 1, input_feature_dim, dtype=torch.float32)

        self.register_buffer("feature_mean", feature_mean.to(dtype=torch.float32))
        self.register_buffer("feature_std", feature_std.to(dtype=torch.float32))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = sequence
        if self.normalize:
            x = (x - self.feature_mean) / self.feature_std
        logits = self.model(x)
        probabilities = torch.sigmoid(logits)
        return probabilities


class LegacySimpleGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(sequence)
        logits = self.classifier(output[:, -1, :])
        return logits.view(-1)


def _infer_model_kwargs_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict:
    keys = list(state_dict.keys())
    if any(key.startswith("gru.") for key in keys):
        input_size = FEATURE_DIM
        hidden_size = HIDDEN_SIZE
        if "feature_extractor.1.weight" in state_dict:
            input_size = int(state_dict["feature_extractor.1.weight"].shape[1])
            hidden_size = int(state_dict["feature_extractor.1.weight"].shape[0])
        elif "gru.weight_ih_l0" in state_dict:
            input_size = int(state_dict["gru.weight_ih_l0"].shape[1])
            hidden_size = int(state_dict["gru.weight_ih_l0"].shape[0] // 3)

        layer_indices = set()
        for key in keys:
            if key.startswith("gru.weight_ih_l"):
                suffix = key.split("gru.weight_ih_l", 1)[1]
                suffix = suffix.replace("_reverse", "")
                if suffix.isdigit():
                    layer_indices.add(int(suffix))
        num_layers = max(layer_indices) + 1 if layer_indices else 2
        return {
            "model_name": "gru",
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": int(num_layers),
            "dropout": DROPOUT,
            "max_seq_len": SEQUENCE_LENGTH,
        }

    if any(key.startswith("tcn.") for key in keys):
        input_size = int(state_dict.get("input_proj.1.weight", torch.empty(HIDDEN_SIZE, FEATURE_DIM)).shape[1])
        hidden_size = int(state_dict.get("input_proj.1.weight", torch.empty(HIDDEN_SIZE, FEATURE_DIM)).shape[0])
        block_indices = set()
        for key in keys:
            if key.startswith("tcn.") and ".conv1.weight" in key:
                # Key format: tcn.<idx>.conv1.weight
                try:
                    idx = int(key.split(".")[1])
                    block_indices.add(idx)
                except Exception:
                    pass
        tcn_blocks = max(block_indices) + 1 if block_indices else 3
        return {
            "model_name": "tcn",
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": 2,
            "dropout": DROPOUT,
            "tcn_blocks": int(tcn_blocks),
            "kernel_size": 3,
            "max_seq_len": SEQUENCE_LENGTH,
        }

    if any(key.startswith("encoder.layers.") for key in keys):
        input_size = int(state_dict.get("input_proj.1.weight", torch.empty(HIDDEN_SIZE, FEATURE_DIM)).shape[1])
        hidden_size = int(state_dict.get("input_proj.1.weight", torch.empty(HIDDEN_SIZE, FEATURE_DIM)).shape[0])
        layer_indices = set()
        for key in keys:
            if key.startswith("encoder.layers."):
                try:
                    idx = int(key.split(".")[2])
                    layer_indices.add(idx)
                except Exception:
                    pass
        num_layers = max(layer_indices) + 1 if layer_indices else 2
        return {
            "model_name": "transformer",
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": int(num_layers),
            "dropout": DROPOUT,
            "num_heads": 4,
            "max_seq_len": SEQUENCE_LENGTH,
        }

    return {
        "model_name": "gru",
        "input_size": FEATURE_DIM,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": 2,
        "dropout": DROPOUT,
        "max_seq_len": SEQUENCE_LENGTH,
    }


def _build_model_from_checkpoint(checkpoint: dict) -> tuple[nn.Module, dict]:
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model_kwargs = checkpoint.get("model_kwargs")
    if model_kwargs:
        model_kwargs = dict(model_kwargs)
        # Backward compatibility for older checkpoints that stored tcn_kernel_size.
        if "kernel_size" not in model_kwargs and "tcn_kernel_size" in model_kwargs:
            model_kwargs["kernel_size"] = model_kwargs["tcn_kernel_size"]
        model_kwargs.setdefault("model_name", checkpoint.get("model_name", "gru"))
        model_kwargs.setdefault("max_seq_len", SEQUENCE_LENGTH)
        model = build_sequence_model(**model_kwargs)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, model_kwargs

    # Legacy checkpoint branch (older GRU-only checkpoint format).
    if all(key.startswith("gru.") or key.startswith("classifier.") for key in state_dict.keys()):
        input_size = int(state_dict["gru.weight_ih_l0"].shape[1])
        hidden_size = int(state_dict["gru.weight_hh_l0"].shape[1])
        layer_indices = set()
        for key in state_dict.keys():
            if key.startswith("gru.weight_ih_l"):
                suffix = key.split("gru.weight_ih_l", 1)[1]
                if suffix.isdigit():
                    layer_indices.add(int(suffix))
        num_layers = max(layer_indices) + 1 if layer_indices else 1
        model = LegacySimpleGRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        model.load_state_dict(state_dict, strict=True)
        resolved = {
            "model_name": "legacy_gru",
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": int(num_layers),
            "dropout": 0.0,
            "max_seq_len": SEQUENCE_LENGTH,
        }
        model.eval()
        return model, resolved

    model_kwargs = _infer_model_kwargs_from_state_dict(state_dict)
    model = build_sequence_model(**model_kwargs)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, model_kwargs


def _quantize_dynamic_if_possible(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU, nn.LSTM},
        dtype=torch.qint8,
    )


def _benchmark(module: torch.jit.ScriptModule, seq_len: int, feature_dim: int, threads: int, iters: int = 200) -> float:
    torch.set_num_threads(max(1, threads))
    x = torch.randn(1, seq_len, feature_dim, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(30):
            _ = module(x)
        start = time.perf_counter()
        for _ in range(iters):
            _ = module(x)
        end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def optimize_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    cpu_threads: int,
    benchmark_iters: int,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, resolved_model_kwargs = _build_model_from_checkpoint(checkpoint)

    normalize_features = bool(checkpoint.get("normalize_features", False))
    feature_mean_raw = checkpoint.get("feature_mean")
    feature_std_raw = checkpoint.get("feature_std")
    feature_mean = torch.tensor(feature_mean_raw, dtype=torch.float32).view(1, 1, -1) if feature_mean_raw is not None else None
    feature_std = torch.tensor(feature_std_raw, dtype=torch.float32).view(1, 1, -1) if feature_std_raw is not None else None

    model_kwargs = dict(checkpoint.get("model_kwargs") or resolved_model_kwargs)
    if "kernel_size" not in model_kwargs and "tcn_kernel_size" in model_kwargs:
        model_kwargs["kernel_size"] = model_kwargs["tcn_kernel_size"]
    feature_dim = int(model_kwargs.get("input_size", FEATURE_DIM))
    seq_len = int(model_kwargs.get("max_seq_len", SEQUENCE_LENGTH))

    wrapped_fp32 = InferenceWrapper(
        model,
        feature_mean,
        feature_std,
        normalize=normalize_features,
        input_feature_dim=feature_dim,
    ).eval()
    scripted_fp32 = torch.jit.script(wrapped_fp32)
    optimized_fp32 = torch.jit.optimize_for_inference(scripted_fp32)

    quantized_model = _quantize_dynamic_if_possible(model)
    wrapped_int8 = InferenceWrapper(
        quantized_model,
        feature_mean,
        feature_std,
        normalize=normalize_features,
        input_feature_dim=feature_dim,
    ).eval()
    scripted_int8 = torch.jit.script(wrapped_int8)
    optimized_int8 = torch.jit.optimize_for_inference(scripted_int8)

    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "engagement_fp32.ts"
    int8_path = output_dir / "engagement_int8_dynamic.ts"
    meta_path = output_dir / "inference_meta.json"

    optimized_fp32.save(str(fp32_path))
    optimized_int8.save(str(int8_path))

    fp32_latency_ms = _benchmark(optimized_fp32, seq_len=seq_len, feature_dim=feature_dim, threads=cpu_threads, iters=benchmark_iters)
    int8_latency_ms = _benchmark(optimized_int8, seq_len=seq_len, feature_dim=feature_dim, threads=cpu_threads, iters=benchmark_iters)

    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "fp32_artifact": str(fp32_path),
        "int8_artifact": str(int8_path),
        "threshold": float(checkpoint.get("best_threshold", 0.5)),
        "model_kwargs": model_kwargs,
        "normalize_features": normalize_features,
        "cpu_threads": int(cpu_threads),
        "benchmark_iters": int(benchmark_iters),
        "fp32_latency_ms": float(fp32_latency_ms),
        "int8_latency_ms": float(int8_latency_ms),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-train optimization for CPU inference (TorchScript + dynamic int8).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "engagement_gru.pt",
        help="Path to training checkpoint (.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHECKPOINT_DIR / "deploy",
        help="Directory for optimized inference artifacts",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="CPU threads for benchmark (recommended: 1-2)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=200,
        help="Benchmark iterations per artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = optimize_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        cpu_threads=args.cpu_threads,
        benchmark_iters=args.benchmark_iters,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
