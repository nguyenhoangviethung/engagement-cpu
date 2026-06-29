from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from engagement_daisee.common.config import BATCH_SIZE, DROPOUT, EPOCHS, FOUR_CLASS_FEATURE_MANIFEST_CSV, HIDDEN_SIZE, LEARNING_RATE, RANDOM_SEED, WEIGHT_DECAY
from engagement_daisee.common.manifest import normalize_manifest_columns
from engagement_daisee.ml.train import _apply_feature_preprocessor, _build_feature_matrix, _fit_feature_preprocessor
from engagement_daisee.multiclass.models import NUM_CLASSES, build_multiclass_model
from engagement_daisee.rnn.dataset import FeatureSequenceDataset


LOGGER = logging.getLogger("multiclass_train_all")
DEFAULT_OUTPUT_DIR = Path("checkpoints/runs/daisee_4class_train_all")
DEFAULT_REPORT_JSON = Path("checkpoints/runs/daisee_4class_train_all/report.json")
DEFAULT_REPORT_CSV = Path("checkpoints/runs/daisee_4class_train_all/report.csv")
DEFAULT_HISTORY_JSONL = Path("checkpoints/runs/daisee_4class_train_all/history.jsonl")
DEFAULT_MODELS = [
    "gru",
    "tcn",
    "gru_basic",
    "tiny_transformer",
    "bilstm",
    "cnn_gru_fusion",
    "hybrid",
    "residual_bigru_attn",
    "xgboost",
]
# With the 32-GiB training profile, keep the richer temporal statistics. XGBoost
# is constrained below (64 histogram bins and isolated process) to retain a
# comfortable memory margin on the full 12,097-column matrix.
DEFAULT_FEATURE_MODE = "tsfresh"
DEFAULT_DEVICE = "cpu"
DEFAULT_CPU_THREADS = 4
DEFAULT_XGB_THREADS = 8
DEFAULT_LATENCY_THREADS = 2
DEFAULT_LATENCY_WARMUP = 30
DEFAULT_LATENCY_ITERS = 200

MODEL_PRESETS: dict[str, dict[str, object]] = {
    "gru": {"hidden_size": 192, "num_layers": 3, "dropout": 0.25},
    "gru_basic": {"hidden_size": 160, "num_layers": 1, "dropout": 0.25},
    "bilstm": {"hidden_size": 160, "num_layers": 2, "dropout": 0.25},
    "tcn": {"hidden_size": 192, "num_layers": 2, "dropout": 0.25, "kernel_size": 5, "tcn_blocks": 4},
    "stgcn": {"hidden_size": 128, "num_layers": 2, "dropout": 0.25, "tcn_blocks": 4},
    "hybrid": {"hidden_size": 160, "num_layers": 2, "dropout": 0.25, "num_heads": 4, "max_seq_len": 30},
    "cnn_gru_fusion": {"hidden_size": 160, "num_layers": 2, "dropout": 0.25, "kernel_size": 5},
    "residual_bigru_attn": {"hidden_size": 160, "num_layers": 2, "dropout": 0.25, "num_heads": 4},
    "tiny_transformer": {"hidden_size": 128, "num_layers": 2, "dropout": 0.25, "num_heads": 4, "max_seq_len": 30},
    "xgboost": {"feature_mode": DEFAULT_FEATURE_MODE, "dim_reduction": "none", "dim_components": 128, "oversample": "none"},
}

INT8_CANDIDATES = {"gru", "gru_basic", "bilstm", "residual_bigru_attn"}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    normalized = device_arg.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested --device {device_arg} but CUDA is not available.")
    return torch.device(device_arg)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _split_indices(manifest: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    split_series = manifest["split"].astype(str).str.strip().str.lower()
    train_indices = split_series[split_series == "train"].index.tolist()
    val_indices = split_series[split_series == "validation"].index.tolist()
    test_indices = split_series[split_series == "test"].index.tolist()

    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            "Official split requires non-empty train/validation/test rows in manifest. "
            f"Got train={len(train_indices)}, validation={len(val_indices)}, test={len(test_indices)}"
        )

    return train_indices, val_indices, test_indices


def _subset_frame(manifest: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    return manifest.iloc[indices].reset_index(drop=True)


def _make_loader(dataset: FeatureSequenceDataset, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        drop_last=False,
    )


def _compute_feature_stats(dataset: FeatureSequenceDataset, indices: list[int], eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    if not indices:
        raise ValueError("Cannot compute feature stats from an empty index list.")

    sample_features, _ = dataset[indices[0]]
    feature_dim = int(sample_features.shape[-1])
    sum_features = torch.zeros(feature_dim, dtype=torch.float64)
    sumsq_features = torch.zeros(feature_dim, dtype=torch.float64)
    total_frames = 0

    for index in indices:
        features, _ = dataset[index]
        frame_features = features.to(dtype=torch.float64)
        sum_features += frame_features.sum(dim=0)
        sumsq_features += (frame_features * frame_features).sum(dim=0)
        total_frames += int(frame_features.shape[0])

    mean = sum_features / max(1, total_frames)
    variance = (sumsq_features / max(1, total_frames)) - (mean * mean)
    std = torch.sqrt(torch.clamp(variance, min=eps))
    return mean.to(dtype=torch.float32), std.to(dtype=torch.float32)


def _cached_feature_stats(
    dataset: FeatureSequenceDataset,
    indices: list[int],
    output_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_path = output_dir / "shared_feature_mean.pt"
    std_path = output_dir / "shared_feature_std.pt"
    expected_dim = int(dataset[indices[0]][0].shape[-1])
    if mean_path.exists() and std_path.exists():
        try:
            mean = torch.load(mean_path, map_location="cpu", weights_only=True)
            std = torch.load(std_path, map_location="cpu", weights_only=True)
            if mean.ndim == 1 and std.ndim == 1 and mean.numel() == expected_dim and std.numel() == expected_dim:
                LOGGER.info("Reusing shared feature normalization cache (%d dimensions)", expected_dim)
                return mean.to(dtype=torch.float32), std.to(dtype=torch.float32)
            LOGGER.warning("Ignoring incompatible shared feature normalization cache")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            LOGGER.warning("Could not load shared feature normalization cache: %s", exc)

    LOGGER.info("Computing shared feature normalization statistics")
    mean, std = _compute_feature_stats(dataset, indices)
    torch.save(mean, mean_path)
    torch.save(std, std_path)
    return mean, std


def _class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    weights = np.zeros_like(counts)
    total = float(counts.sum())
    for idx, count in enumerate(counts):
        weights[idx] = total / (num_classes * max(1.0, float(count)))
    weights = weights / max(1e-12, float(weights.mean()))
    return torch.tensor(weights, dtype=torch.float32)


def _labels_from_subset(manifest: pd.DataFrame, indices: list[int]) -> np.ndarray:
    return manifest.iloc[indices]["label"].astype(np.int64).to_numpy()


def _aggregate_by_video(manifest_subset: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "video_id" not in manifest_subset.columns:
        return labels.astype(np.int64), probabilities.astype(np.float32)

    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim == 1:
        probs = probs[:, None]

    frame = pd.DataFrame({"video_id": manifest_subset["video_id"].astype(str).to_numpy(), "label": labels.astype(np.int64)})
    prob_cols = {f"p_{idx}": probs[:, idx] for idx in range(probs.shape[1])}
    frame = pd.concat([frame, pd.DataFrame(prob_cols)], axis=1)
    grouped = frame.groupby("video_id", sort=False)
    video_labels = grouped["label"].first().to_numpy(dtype=np.int64)
    video_probabilities = grouped[[f"p_{idx}" for idx in range(probs.shape[1])]].mean().to_numpy(dtype=np.float32)
    return video_labels, video_probabilities


def _compute_multiclass_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, object]:
    labels = labels.astype(np.int64)
    predictions = predictions.astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "precision_per_class": [float(value) for value in precision.tolist()],
        "recall_per_class": [float(value) for value in recall.tolist()],
        "f1_per_class": [float(value) for value in f1.tolist()],
        "support_per_class": [int(value) for value in support.tolist()],
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES))).tolist(),
    }
    if probabilities is not None and probabilities.ndim == 2 and probabilities.shape[1] == NUM_CLASSES:
        clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0)
        row_indices = np.arange(len(labels))
        metrics["cross_entropy"] = float(-np.mean(np.log(clipped[row_indices, labels])))
    return metrics


def _timer_ms(fn: Callable[[], object], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
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


class MulticlassInferenceWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        feature_mean: torch.Tensor | None,
        feature_std: torch.Tensor | None,
        normalize: bool,
        input_size: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalize = bool(normalize)

        if feature_mean is None:
            feature_mean = torch.zeros(1, 1, input_size, dtype=torch.float32)
        if feature_std is None:
            feature_std = torch.ones(1, 1, input_size, dtype=torch.float32)

        self.register_buffer("feature_mean", feature_mean.to(dtype=torch.float32))
        self.register_buffer("feature_std", feature_std.to(dtype=torch.float32))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = sequence
        if self.normalize:
            x = (x - self.feature_mean) / self.feature_std
        return self.model(x)


def _scripted_or_eager_module(
    model_name: str,
    model: nn.Module,
    feature_mean: torch.Tensor | None,
    feature_std: torch.Tensor | None,
    normalize: bool,
    input_size: int,
) -> tuple[nn.Module, str]:
    def _make_wrapper(base_model: nn.Module) -> nn.Module:
        return MulticlassInferenceWrapper(
            base_model.eval(), feature_mean, feature_std, normalize=normalize, input_size=input_size
        ).eval()

    wrapper = _make_wrapper(model)
    variant = "torchscript_fp32"
    if model_name in INT8_CANDIDATES:
        try:
            quantized = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.GRU, nn.LSTM}, dtype=torch.qint8)
            wrapper = _make_wrapper(quantized)
            variant = "int8_dynamic"
        except Exception as exc:  # pragma: no cover - optional acceleration path
            LOGGER.warning("Dynamic quantization failed for %s, falling back to TorchScript FP32: %s", model_name, exc)
            wrapper = _make_wrapper(model)

        if variant == "int8_dynamic":
            try:
                probe = torch.zeros(1, 2, input_size, dtype=torch.float32)
                with torch.no_grad():
                    _ = wrapper(probe)
            except Exception as exc:  # pragma: no cover - model-specific quantization fallback
                LOGGER.warning(
                    "Dynamic quantization produced an invalid forward for %s, falling back to FP32: %s",
                    model_name,
                    exc,
                )
                wrapper = _make_wrapper(model)
                variant = "torchscript_fp32"

    try:
        scripted = torch.jit.optimize_for_inference(torch.jit.script(wrapper))
        return scripted, variant
    except Exception as exc:  # pragma: no cover - fallback for scripting edge cases
        LOGGER.warning("TorchScript optimization failed for %s, using eager wrapper: %s", model_name, exc)
        return wrapper, f"eager_fallback_{variant}"


def _build_model_kwargs(model_name: str, sequence_length: int, input_size: int) -> dict[str, object]:
    preset = dict(MODEL_PRESETS.get(model_name, {}))
    preset.setdefault("input_size", input_size)
    preset.setdefault("hidden_size", HIDDEN_SIZE)
    preset.setdefault("num_layers", 2)
    preset.setdefault("dropout", DROPOUT)
    preset.setdefault("num_heads", 4)
    preset.setdefault("kernel_size", 3)
    preset.setdefault("tcn_blocks", 3)
    preset.setdefault("max_seq_len", sequence_length)
    preset.setdefault("num_classes", NUM_CLASSES)
    if model_name == "gru_basic":
        preset["num_layers"] = 1
    if model_name == "tiny_transformer":
        preset["max_seq_len"] = sequence_length
    if model_name == "hybrid":
        preset["max_seq_len"] = sequence_length
    return preset


def _predict_probs_from_model(model: nn.Module, features: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def _evaluate_neural(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
    amp_enabled: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    running_total = 0
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device).long()
        if feature_mean is not None and feature_std is not None:
            features = (features - feature_mean) / feature_std

        if amp_enabled and device.type == "cuda":
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(features)
                loss = criterion(logits, labels)
        else:
            logits = model(features)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        running_total += batch_size
        probs = torch.softmax(logits, dim=-1)
        all_probabilities.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    probabilities = np.concatenate(all_probabilities).astype(np.float32)
    labels_np = np.concatenate(all_labels).astype(np.int64)
    average_loss = running_loss / max(1, running_total)
    return average_loss, labels_np, probabilities


def _train_neural_model(
    model_name: str,
    manifest: pd.DataFrame,
    dataset: FeatureSequenceDataset,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    device: torch.device,
    output_dir: Path,
    *,
    batch_size: int,
    epochs: int,
    patience: int,
    min_epochs: int,
    lr: float,
    weight_decay: float,
    objective: str,
    cpu_threads: int,
    latency_threads: int,
    latency_warmup: int,
    latency_iters: int,
    amp_enabled: bool,
    normalize_features: bool,
    feature_mean_cpu: torch.Tensor | None,
    feature_std_cpu: torch.Tensor | None,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    num_heads: int,
    kernel_size: int,
    tcn_blocks: int,
) -> dict[str, object]:
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, cpu_threads))

    sample_sequence = dataset[train_indices[0]][0]
    input_size = int(sample_sequence.shape[-1])
    model_kwargs = _build_model_kwargs(
        model_name,
        sequence_length=int(sample_sequence.shape[0]),
        input_size=input_size,
    )
    model_kwargs.update(
        {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "num_heads": num_heads,
            "kernel_size": kernel_size,
            "tcn_blocks": tcn_blocks,
            "max_seq_len": model_kwargs.get("max_seq_len", int(dataset[train_indices[0]][0].shape[0])),
            "num_classes": NUM_CLASSES,
        }
    )
    model = build_multiclass_model(model_name=model_name, **model_kwargs).to(device)

    train_labels = _labels_from_subset(manifest, train_indices)
    class_weight_tensor = _class_weights(train_labels, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(1, patience // 2))

    train_loader = _make_loader(dataset, train_indices, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(dataset, val_indices, batch_size=batch_size, shuffle=False)
    test_loader = _make_loader(dataset, test_indices, batch_size=batch_size, shuffle=False)

    feature_mean = feature_std = None
    if normalize_features:
        if feature_mean_cpu is None or feature_std_cpu is None:
            feature_mean_cpu, feature_std_cpu = _compute_feature_stats(dataset, train_indices)
        feature_mean = feature_mean_cpu.to(device).view(1, 1, -1)
        feature_std = feature_std_cpu.to(device).view(1, 1, -1)

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    best_val_row_metrics: dict[str, object] | None = None
    best_val_video_metrics: dict[str, object] | None = None
    history: list[dict[str, object]] = []

    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0
        train_probs: list[np.ndarray] = []
        train_labels_list: list[np.ndarray] = []

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).long()
            if feature_mean is not None and feature_std is not None:
                features = (features - feature_mean) / feature_std

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled and device.type == "cuda":
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(features)
                    loss = criterion(logits, labels)
                loss.backward()
            else:
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_current = labels.size(0)
            running_loss += float(loss.item()) * batch_size_current
            running_total += batch_size_current
            train_probs.append(torch.softmax(logits.detach(), dim=-1).cpu().numpy())
            train_labels_list.append(labels.detach().cpu().numpy())

        train_loss = running_loss / max(1, running_total)
        train_probabilities = np.concatenate(train_probs).astype(np.float32)
        train_labels_np = np.concatenate(train_labels_list).astype(np.int64)
        train_predictions = np.argmax(train_probabilities, axis=1)
        train_row_metrics = _compute_multiclass_metrics(train_labels_np, train_predictions, train_probabilities)

        val_loss, val_labels_np, val_probabilities = _evaluate_neural(
            model,
            val_loader,
            criterion,
            device,
            feature_mean=feature_mean,
            feature_std=feature_std,
            amp_enabled=amp_enabled,
        )
        val_predictions = np.argmax(val_probabilities, axis=1)
        val_row_metrics = _compute_multiclass_metrics(val_labels_np, val_predictions, val_probabilities)
        val_manifest = _subset_frame(manifest, val_indices)
        val_video_labels, val_video_probabilities = _aggregate_by_video(val_manifest, val_labels_np, val_probabilities)
        val_video_predictions = np.argmax(val_video_probabilities, axis=1)
        val_video_metrics = _compute_multiclass_metrics(val_video_labels, val_video_predictions, val_video_probabilities)

        metrics_for_objective = val_video_metrics
        current_score = float(metrics_for_objective[objective])
        scheduler.step(current_score)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_row_metrics["accuracy"],
                "train_balanced_accuracy": train_row_metrics["balanced_accuracy"],
                "train_f1_macro": train_row_metrics["f1_macro"],
                "val_loss": val_loss,
                "val_accuracy": val_video_metrics["accuracy"],
                "val_balanced_accuracy": val_video_metrics["balanced_accuracy"],
                "val_f1_macro": val_video_metrics["f1_macro"],
                "val_row_loss": val_loss,
                "val_row_accuracy": val_row_metrics["accuracy"],
                "val_row_balanced_accuracy": val_row_metrics["balanced_accuracy"],
                "val_row_f1_macro": val_row_metrics["f1_macro"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        LOGGER.info(
            "[%s] epoch=%02d/%02d train_loss=%.4f train_acc=%.4f train_bal=%.4f train_f1=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_bal=%.4f val_f1=%.4f | score(%s)=%.4f lr=%.6f",
            model_name,
            epoch,
            epochs,
            train_loss,
            float(train_row_metrics["accuracy"]),
            float(train_row_metrics["balanced_accuracy"]),
            float(train_row_metrics["f1_macro"]),
            val_loss,
            float(val_video_metrics["accuracy"]),
            float(val_video_metrics["balanced_accuracy"]),
            float(val_video_metrics["f1_macro"]),
            objective,
            current_score,
            optimizer.param_groups[0]["lr"],
        )

        if current_score > best_score + 1e-8:
            best_score = current_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_val_row_metrics = copy.deepcopy(val_row_metrics)
            best_val_video_metrics = copy.deepcopy(val_video_metrics)

        if epoch >= min_epochs and (epoch - best_epoch) >= patience:
            LOGGER.info("[%s] early stopping at epoch=%d best_epoch=%d", model_name, epoch, best_epoch)
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model = model.to(device)

    test_loss, test_labels_np, test_probabilities = _evaluate_neural(
        model,
        test_loader,
        criterion,
        device,
        feature_mean=feature_mean,
        feature_std=feature_std,
        amp_enabled=amp_enabled,
    )
    test_predictions = np.argmax(test_probabilities, axis=1)
    test_row_metrics = _compute_multiclass_metrics(test_labels_np, test_predictions, test_probabilities)
    test_manifest = _subset_frame(manifest, test_indices)
    test_video_labels, test_video_probabilities = _aggregate_by_video(test_manifest, test_labels_np, test_probabilities)
    test_video_predictions = np.argmax(test_video_probabilities, axis=1)
    test_video_metrics = _compute_multiclass_metrics(test_video_labels, test_video_predictions, test_video_probabilities)

    latency_model = None
    latency_e2e = None
    latency_variant = None
    sample_path = Path(str(test_manifest.iloc[0]["feature_path"]))
    sample_sequence = np.load(sample_path).astype(np.float32)
    sample_tensor = torch.from_numpy(sample_sequence[None, ...]).to(device)

    latency_model_cpu = model.to("cpu").eval()
    if feature_mean is not None and feature_std is not None:
        feature_mean_cpu = feature_mean.detach().cpu()
        feature_std_cpu = feature_std.detach().cpu()
    else:
        feature_mean_cpu = feature_std_cpu = None

    latency_module, latency_variant = _scripted_or_eager_module(
        model_name=model_name,
        model=latency_model_cpu,
        feature_mean=feature_mean_cpu,
        feature_std=feature_std_cpu,
        normalize=normalize_features,
        input_size=input_size,
    )
    torch.set_num_threads(max(1, latency_threads))
    latency_model = _timer_ms(lambda: latency_module(sample_tensor.cpu()), warmup=latency_warmup, iters=latency_iters)

    def _predict_from_path() -> np.ndarray:
        sequence = np.load(sample_path).astype(np.float32)
        tensor = torch.from_numpy(sequence[None, ...])
        with torch.no_grad():
            logits = latency_module(tensor)
            return torch.softmax(logits, dim=-1).cpu().numpy()

    latency_e2e = _timer_ms(_predict_from_path, warmup=latency_warmup, iters=latency_iters)

    payload = {
        "model_name": model_name,
        "model_family": "neural",
        "status": "success",
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / f"{model_name}.pt"),
        "model_kwargs": model_kwargs,
        "objective": objective,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_val_row_metrics": best_val_row_metrics,
        "best_val_video_metrics": best_val_video_metrics,
        "train_history": history,
        "train_counts": np.bincount(train_labels, minlength=NUM_CLASSES).astype(int).tolist(),
        "val_counts": np.bincount(_labels_from_subset(manifest, val_indices), minlength=NUM_CLASSES).astype(int).tolist(),
        "test_counts": np.bincount(_labels_from_subset(manifest, test_indices), minlength=NUM_CLASSES).astype(int).tolist(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "train_row_metrics": train_row_metrics,
        "val_row_metrics": val_row_metrics,
        "val_video_metrics": val_video_metrics,
        "test_row_metrics": test_row_metrics,
        "test_video_metrics": test_video_metrics,
        "latency": {
            "latency_kind": "processed_feature_sequence",
            "sample_feature_path": str(sample_path),
            "model_side": {"variant": latency_variant, **latency_model},
            "end_to_end": {"variant": latency_variant, **latency_e2e},
        },
        "feature_normalization": {
            "enabled": bool(normalize_features),
            "mean_path": str(run_dir / "feature_mean.pt") if normalize_features else None,
            "std_path": str(run_dir / "feature_std.pt") if normalize_features else None,
        },
        "device": str(device),
        "amp_enabled": bool(amp_enabled and device.type == "cuda"),
        "cpu_threads": int(cpu_threads),
    }

    torch.save(
        {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "model_state_dict": best_state,
            "best_epoch": int(best_epoch),
            "objective": objective,
            "best_score": float(best_score),
            "best_val_row_metrics": best_val_row_metrics,
            "best_val_video_metrics": best_val_video_metrics,
            "train_history": history,
            "train_counts": payload["train_counts"],
            "val_counts": payload["val_counts"],
            "test_counts": payload["test_counts"],
            "feature_normalization": payload["feature_normalization"],
            "feature_mean": feature_mean_cpu.numpy().tolist() if feature_mean_cpu is not None else None,
            "feature_std": feature_std_cpu.numpy().tolist() if feature_std_cpu is not None else None,
            "latency": payload["latency"],
        },
        run_dir / f"{model_name}.pt",
    )
    if normalize_features and feature_mean_cpu is not None and feature_std_cpu is not None:
        torch.save(feature_mean_cpu, run_dir / "feature_mean.pt")
        torch.save(feature_std_cpu, run_dir / "feature_std.pt")
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    LOGGER.info(
        "[%s] test video metrics acc=%.4f bal=%.4f f1=%.4f | model latency=%.3f ms | e2e=%.3f ms",
        model_name,
        float(test_video_metrics["accuracy"]),
        float(test_video_metrics["balanced_accuracy"]),
        float(test_video_metrics["f1_macro"]),
        float(latency_model["latency_ms_mean"]),
        float(latency_e2e["latency_ms_mean"]),
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


def _predict_xgb(model: XGBClassifier, x: np.ndarray) -> np.ndarray:
    booster = model.get_booster()
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return model.predict_proba(x, iteration_range=(0, int(best_iteration) + 1)).astype(np.float32)
    return model.predict_proba(x).astype(np.float32)


def _train_xgboost(
    manifest: pd.DataFrame,
    output_dir: Path,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    *,
    feature_mode: str,
    dim_reduction: str,
    dim_components: int,
    oversample: str,
    cpu_threads: int,
    latency_threads: int,
    latency_warmup: int,
    latency_iters: int,
    objective: str,
    seed: int,
) -> dict[str, object]:
    run_dir = output_dir / "xgboost"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = _subset_frame(manifest, train_indices)
    val_manifest = _subset_frame(manifest, val_indices)
    test_manifest = _subset_frame(manifest, test_indices)

    x_train, y_train = _build_feature_matrix(train_manifest, feature_mode=feature_mode)
    x_val, y_val = _build_feature_matrix(val_manifest, feature_mode=feature_mode)
    x_test, y_test = _build_feature_matrix(test_manifest, feature_mode=feature_mode)

    preprocessor_config, x_train = _fit_feature_preprocessor(
        x_train,
        dim_reduction=dim_reduction,
        dim_components=dim_components,
    )
    x_val = _apply_feature_preprocessor(x_val, preprocessor_config)
    x_test = _apply_feature_preprocessor(x_test, preprocessor_config)

    class_weights = _class_weights(y_train, num_classes=NUM_CLASSES).numpy()
    sample_weight = class_weights[y_train]
    if oversample == "random":
        rng = np.random.default_rng(seed)
        target = int(np.bincount(y_train, minlength=NUM_CLASSES).max())
        indices = []
        for label in range(NUM_CLASSES):
            label_indices = np.flatnonzero(y_train == label)
            if not len(label_indices):
                continue
            if len(label_indices) < target:
                extra = rng.choice(label_indices, size=target - len(label_indices), replace=True)
                label_indices = np.concatenate([label_indices, extra])
            indices.append(label_indices)
        if indices:
            merged_indices = np.concatenate(indices)
            rng.shuffle(merged_indices)
            x_train = x_train[merged_indices]
            y_train = y_train[merged_indices]
            sample_weight = class_weights[y_train]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        n_estimators=800,
        learning_rate=0.05,
        max_depth=4,
        max_bin=64,
        min_child_weight=3.0,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        tree_method="hist",
        n_jobs=max(1, cpu_threads),
        random_state=seed,
        eval_metric="mlogloss",
        callbacks=[
            EarlyStopping(
                rounds=40,
                metric_name="mlogloss",
                data_name="validation_0",
                maximize=False,
                save_best=True,
            )
        ],
    )
    LOGGER.info("[xgboost] fitting multiclass model with %d rows", len(x_train))
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    val_probabilities = _predict_xgb(model, x_val)
    val_predictions = np.argmax(val_probabilities, axis=1)
    val_row_metrics = _compute_multiclass_metrics(y_val, val_predictions, val_probabilities)
    val_video_labels, val_video_probabilities = _aggregate_by_video(val_manifest, y_val, val_probabilities)
    val_video_predictions = np.argmax(val_video_probabilities, axis=1)
    val_video_metrics = _compute_multiclass_metrics(val_video_labels, val_video_predictions, val_video_probabilities)

    test_probabilities = _predict_xgb(model, x_test)
    test_predictions = np.argmax(test_probabilities, axis=1)
    test_row_metrics = _compute_multiclass_metrics(y_test, test_predictions, test_probabilities)
    test_video_labels, test_video_probabilities = _aggregate_by_video(test_manifest, y_test, test_probabilities)
    test_video_predictions = np.argmax(test_video_probabilities, axis=1)
    test_video_metrics = _compute_multiclass_metrics(test_video_labels, test_video_predictions, test_video_probabilities)

    sample_path = Path(str(test_manifest.iloc[0]["feature_path"]))
    sample_tabular = _build_feature_matrix(pd.DataFrame([test_manifest.iloc[0].to_dict()]), feature_mode=feature_mode)[0]
    sample_tabular = _apply_feature_preprocessor(sample_tabular, preprocessor_config)

    def _predict_model_side() -> np.ndarray:
        return _predict_xgb(model, sample_tabular)

    def _predict_e2e() -> np.ndarray:
        sequence = np.load(sample_path).astype(np.float32)
        tabular = _build_feature_matrix(pd.DataFrame([{"feature_path": str(sample_path), "label": int(test_manifest.iloc[0]["label"]), "split": "test"}]), feature_mode=feature_mode)[0]
        tabular = _apply_feature_preprocessor(tabular, preprocessor_config)
        return _predict_xgb(model, tabular)

    torch.set_num_threads(max(1, latency_threads))
    latency_model = _timer_ms(_predict_model_side, warmup=latency_warmup, iters=latency_iters)
    latency_e2e = _timer_ms(_predict_e2e, warmup=latency_warmup, iters=latency_iters)

    model_path = run_dir / "model.json"
    model.get_booster().save_model(str(model_path))
    preprocessor_path = run_dir / "preprocessor.npz"
    np.savez(preprocessor_path, **{k: np.asarray(v) for k, v in preprocessor_config.items()})
    summary = {
        "model_name": "xgboost",
        "model_family": "ml",
        "status": "success",
        "run_dir": str(run_dir),
        "checkpoint_path": str(model_path),
        "model_kwargs": {
            "feature_mode": feature_mode,
            "dim_reduction": dim_reduction,
            "dim_components": dim_components,
            "oversample": oversample,
        },
        "objective": objective,
        "best_epoch": int(getattr(model, "best_iteration", 0) or 0),
        "best_score": float(val_video_metrics[objective]),
        "best_val_row_metrics": val_row_metrics,
        "best_val_video_metrics": val_video_metrics,
        "train_counts": np.bincount(y_train, minlength=NUM_CLASSES).astype(int).tolist(),
        "val_counts": np.bincount(y_val, minlength=NUM_CLASSES).astype(int).tolist(),
        "test_counts": np.bincount(y_test, minlength=NUM_CLASSES).astype(int).tolist(),
        "train_loss": None,
        "val_loss": float(val_row_metrics.get("cross_entropy", float("nan"))),
        "test_loss": float(test_row_metrics.get("cross_entropy", float("nan"))),
        "train_row_metrics": None,
        "val_row_metrics": val_row_metrics,
        "val_video_metrics": val_video_metrics,
        "test_row_metrics": test_row_metrics,
        "test_video_metrics": test_video_metrics,
        "latency": {
            "latency_kind": "processed_feature_sequence",
            "sample_feature_path": str(sample_path),
            "model_side": {"variant": "xgboost", **latency_model},
            "end_to_end": {"variant": "xgboost", **latency_e2e},
        },
        "feature_normalization": {
            "enabled": True,
            "preprocessor_path": str(preprocessor_path),
        },
        "device": "cpu",
        "amp_enabled": False,
        "cpu_threads": int(cpu_threads),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _finalize_report(items: list[dict[str, object]], output_dir: Path, report_json: Path | None, report_csv: Path | None, history_jsonl: Path | None, manifest: pd.DataFrame, models: list[str]) -> dict[str, object]:
    successful = [item for item in items if item.get("status") == "success"]
    failed = [item for item in items if item.get("status") != "success"]

    def _metric(item: dict[str, object], key: str) -> float:
        metrics = item.get("test_video_metrics") or {}
        return float(metrics.get(key, -math.inf))

    def _shallow(item: dict[str, object] | None) -> dict[str, object] | None:
        if not item:
            return None
        return {
            "model_name": item.get("model_name"),
            "model_family": item.get("model_family"),
            "status": item.get("status"),
            "best_epoch": item.get("best_epoch"),
            "best_score": item.get("best_score"),
            "checkpoint_path": item.get("checkpoint_path"),
            "test_video_metrics": item.get("test_video_metrics"),
            "latency": item.get("latency"),
        }

    payload = {
        "manifest": str(output_dir / "manifest_snapshot.csv"),
        "rows": int(len(manifest)),
        "videos": int(manifest["video_id"].nunique()) if "video_id" in manifest.columns else None,
        "splits": {k: int(v) for k, v in manifest["split"].value_counts().sort_index().items()},
        "models_requested": models,
        "items": items,
        "successful": len(successful),
        "failed": len(failed),
        "failed_models": [item.get("model_name") for item in failed],
        "best_by_accuracy": _shallow(max(successful, key=lambda item: _metric(item, "accuracy"))) if successful else None,
        "best_by_balanced_accuracy": _shallow(max(successful, key=lambda item: _metric(item, "balanced_accuracy"))) if successful else None,
        "best_by_f1_macro": _shallow(max(successful, key=lambda item: _metric(item, "f1_macro"))) if successful else None,
    }

    summary_json_path = output_dir / "train_all_summary.json"
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest.to_csv(output_dir / "manifest_snapshot.csv", index=False)

    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if report_csv is not None:
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in items:
            row = {
                "model_name": item.get("model_name"),
                "model_family": item.get("model_family"),
                "status": item.get("status"),
                "best_epoch": item.get("best_epoch"),
                "best_score": item.get("best_score"),
                "test_accuracy": (item.get("test_video_metrics") or {}).get("accuracy"),
                "test_balanced_accuracy": (item.get("test_video_metrics") or {}).get("balanced_accuracy"),
                "test_f1_macro": (item.get("test_video_metrics") or {}).get("f1_macro"),
                "val_accuracy": (item.get("val_video_metrics") or {}).get("accuracy"),
                "val_balanced_accuracy": (item.get("val_video_metrics") or {}).get("balanced_accuracy"),
                "val_f1_macro": (item.get("val_video_metrics") or {}).get("f1_macro"),
                "model_side_latency_ms_mean": ((item.get("latency") or {}).get("model_side") or {}).get("latency_ms_mean"),
                "model_side_latency_ms_median": ((item.get("latency") or {}).get("model_side") or {}).get("latency_ms_median"),
                "model_side_latency_ms_p95": ((item.get("latency") or {}).get("model_side") or {}).get("latency_ms_p95"),
                "e2e_latency_ms_mean": ((item.get("latency") or {}).get("end_to_end") or {}).get("latency_ms_mean"),
                "e2e_latency_ms_median": ((item.get("latency") or {}).get("end_to_end") or {}).get("latency_ms_median"),
                "e2e_latency_ms_p95": ((item.get("latency") or {}).get("end_to_end") or {}).get("latency_ms_p95"),
            }
            rows.append(row)
        pd.DataFrame(rows).to_csv(report_csv, index=False)

    if history_jsonl is not None:
        history_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with history_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return payload


def run_experiment(
    manifest_path: Path,
    output_dir: Path,
    models: list[str],
    *,
    device: str,
    batch_size: int,
    epochs: int,
    patience: int,
    min_epochs: int,
    lr: float,
    weight_decay: float,
    objective: str,
    feature_mode: str,
    dim_reduction: str,
    dim_components: int,
    oversample: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    num_heads: int,
    kernel_size: int,
    tcn_blocks: int,
    cpu_threads: int,
    xgb_threads: int,
    latency_threads: int,
    latency_warmup: int,
    latency_iters: int,
    amp_enabled: bool,
    report_json: Path | None,
    report_csv: Path | None,
    history_jsonl: Path | None,
    finalize_report: bool = True,
) -> dict[str, object]:
    _set_seed(RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = normalize_manifest_columns(pd.read_csv(manifest_path, low_memory=False))
    required_columns = {"feature_path", "label", "split"}
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["label"] = manifest["label"].astype(int)
    train_indices, val_indices, test_indices = _split_indices(manifest)
    models_to_run = [name.strip().lower() for name in models if name.strip()]
    needs_neural_dataset = any(model_name != "xgboost" for model_name in models_to_run)
    dataset = FeatureSequenceDataset(manifest_path) if needs_neural_dataset else None
    if dataset is not None:
        feature_mean_cpu, feature_std_cpu = _cached_feature_stats(dataset, train_indices, output_dir)
    else:
        feature_mean_cpu = feature_std_cpu = None

    LOGGER.info(
        "Loaded manifest | rows=%d videos=%s train=%d val=%d test=%d device=%s",
        len(manifest),
        int(manifest["video_id"].nunique()) if "video_id" in manifest.columns else None,
        len(train_indices),
        len(val_indices),
        len(test_indices),
        device,
    )

    device_obj = _resolve_device(device)
    items: list[dict[str, object]] = []
    for model_name in models_to_run:
        LOGGER.info("=== Running model: %s ===", model_name)
        try:
            if model_name == "xgboost":
                item = _train_xgboost(
                    manifest,
                    output_dir,
                    train_indices,
                    val_indices,
                    test_indices,
                    feature_mode=feature_mode,
                    dim_reduction=dim_reduction,
                    dim_components=dim_components,
                    oversample=oversample,
                    cpu_threads=xgb_threads,
                    latency_threads=latency_threads,
                    latency_warmup=latency_warmup,
                    latency_iters=latency_iters,
                    objective=objective,
                    seed=RANDOM_SEED,
                )
            else:
                if dataset is None:
                    raise RuntimeError("Neural model requested without an initialized sequence dataset")
                item = _train_neural_model(
                    model_name,
                    manifest,
                    dataset,
                    train_indices,
                    val_indices,
                    test_indices,
                    device_obj,
                    output_dir,
                    batch_size=batch_size,
                    epochs=epochs,
                    patience=patience,
                    min_epochs=min_epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    objective=objective,
                    cpu_threads=cpu_threads,
                    latency_threads=latency_threads,
                    latency_warmup=latency_warmup,
                    latency_iters=latency_iters,
                    amp_enabled=amp_enabled,
                    normalize_features=True,
                    feature_mean_cpu=feature_mean_cpu,
                    feature_std_cpu=feature_std_cpu,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    tcn_blocks=tcn_blocks,
                )
            items.append(item)
        except Exception as exc:  # pragma: no cover - pipeline resilience
            LOGGER.exception("Model %s failed: %s", model_name, exc)
            items.append(
                {
                    "model_name": model_name,
                    "model_family": "ml" if model_name == "xgboost" else "neural",
                    "status": "failed",
                    "error": str(exc),
                    "device": str(device_obj),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if finalize_report:
        return _finalize_report(items, output_dir, report_json, report_csv, history_jsonl, manifest, models_to_run)
    return {"items": items}


def _format_subprocess_failure(model_name: str, returncode: int) -> str:
    if returncode < 0:
        try:
            signal_name = signal.Signals(-returncode).name
        except ValueError:
            signal_name = f"signal {-returncode}"
        return f"isolated worker for {model_name} was terminated by {signal_name}"
    return f"isolated worker for {model_name} exited with code {returncode}"


def _run_isolated_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Run each model in its own process so an OOM cannot abort train-all."""
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [name.strip().lower() for name in args.models if name.strip()]
    items: list[dict[str, object]] = []

    common_args = [
        "--manifest", str(args.manifest),
        "--output-dir", str(output_dir),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--min-epochs", str(args.min_epochs),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--objective", args.objective,
        "--feature-mode", args.feature_mode,
        "--dim-reduction", args.dim_reduction,
        "--dim-components", str(args.dim_components),
        "--oversample", args.oversample,
        "--hidden-size", str(args.hidden_size),
        "--num-layers", str(args.num_layers),
        "--dropout", str(args.dropout),
        "--num-heads", str(args.num_heads),
        "--kernel-size", str(args.kernel_size),
        "--tcn-blocks", str(args.tcn_blocks),
        "--cpu-threads", str(args.cpu_threads),
        "--xgb-threads", str(args.xgb_threads),
        "--latency-threads", str(args.latency_threads),
        "--latency-warmup", str(args.latency_warmup),
        "--latency-iters", str(args.latency_iters),
        "--isolated-worker",
    ]
    if args.no_amp:
        common_args.append("--no-amp")

    worker_env = dict(os.environ)
    thread_count = str(max(1, args.cpu_threads))
    worker_env.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
        }
    )

    progress_path = output_dir / "train_all_progress.json"
    for position, model_name in enumerate(models, start=1):
        summary_path = output_dir / model_name / "summary.json"
        if args.resume and summary_path.exists():
            try:
                cached = json.loads(summary_path.read_text(encoding="utf-8"))
                if cached.get("status") == "success":
                    LOGGER.info("[%d/%d] Resuming: model %s is already complete", position, len(models), model_name)
                    items.append(cached)
                    progress_path.write_text(
                        json.dumps({"models_requested": models, "items": items}, indent=2), encoding="utf-8"
                    )
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        LOGGER.info("[%d/%d] Starting isolated model: %s", position, len(models), model_name)
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            "engagement_daisee.multiclass.train_all",
            "--models",
            model_name,
            *common_args,
            "--report-json",
            str(model_dir / "worker_report.json"),
            "--report-csv",
            str(model_dir / "worker_report.csv"),
            "--history-jsonl",
            str(model_dir / "worker_history.jsonl"),
        ]
        model_env = dict(worker_env)
        if model_name == "xgboost":
            xgb_thread_count = str(max(1, args.xgb_threads))
            model_env.update(
                {
                    "OMP_NUM_THREADS": xgb_thread_count,
                    "MKL_NUM_THREADS": xgb_thread_count,
                    "OPENBLAS_NUM_THREADS": xgb_thread_count,
                    "NUMEXPR_NUM_THREADS": xgb_thread_count,
                }
            )
        completed = subprocess.run(command, env=model_env, check=False)

        item: dict[str, object]
        if completed.returncode == 0 and summary_path.exists():
            try:
                item = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                item = {
                    "model_name": model_name,
                    "model_family": "ml" if model_name == "xgboost" else "neural",
                    "status": "failed",
                    "error": f"worker produced an invalid summary: {exc}",
                }
        else:
            item = {
                "model_name": model_name,
                "model_family": "ml" if model_name == "xgboost" else "neural",
                "status": "failed",
                "error": _format_subprocess_failure(model_name, completed.returncode),
                "returncode": int(completed.returncode),
            }
            summary_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
            LOGGER.error("Model %s failed, continuing with the remaining models: %s", model_name, item["error"])

        items.append(item)
        progress_path.write_text(
            json.dumps({"models_requested": models, "completed": position, "total": len(models), "items": items}, indent=2),
            encoding="utf-8",
        )

    manifest = normalize_manifest_columns(pd.read_csv(args.manifest, low_memory=False))
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["label"] = manifest["label"].astype(int)
    return _finalize_report(
        items,
        output_dir,
        args.report_json,
        args.report_csv,
        args.history_jsonl,
        manifest,
        models,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate all DAiSEE 4-class feature models.")
    parser.add_argument("--manifest", type=Path, default=FOUR_CLASS_FEATURE_MANIFEST_CSV, help="4-class feature manifest")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for run artifacts")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Ordered list of models to train")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Neural model batch size")
    parser.add_argument("--epochs", type=int, default=min(EPOCHS, 24), help="Max epochs for neural models")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--min-epochs", type=int, default=5, help="Minimum epochs before early stop")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument(
        "--objective",
        type=str,
        default="balanced_accuracy",
        choices=["accuracy", "balanced_accuracy", "f1_macro"],
        help="Validation objective used for checkpoint selection",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default=DEFAULT_FEATURE_MODE,
        choices=["basic", "tsfresh", "copur"],
        help="Tabular feature mode used by XGBoost",
    )
    parser.add_argument("--dim-reduction", type=str, default="none", choices=["none", "pca", "svd"], help="Optional feature reduction for XGBoost")
    parser.add_argument("--dim-components", type=int, default=128, help="PCA/SVD components for XGBoost")
    parser.add_argument("--oversample", type=str, default="none", choices=["none", "random"], help="Optional random oversampling for XGBoost")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE, help="Base hidden size for neural models")
    parser.add_argument("--num-layers", type=int, default=2, help="Base recurrent/transformer depth")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout rate for neural models")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads for transformer-like models")
    parser.add_argument("--kernel-size", type=int, default=5, help="Kernel size for TCN/CNN fusion")
    parser.add_argument("--tcn-blocks", type=int, default=4, help="TCN block count")
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS, help="PyTorch CPU threads")
    parser.add_argument("--xgb-threads", type=int, default=DEFAULT_XGB_THREADS, help="XGBoost CPU threads")
    parser.add_argument("--latency-threads", type=int, default=DEFAULT_LATENCY_THREADS, help="CPU threads for latency benchmark")
    parser.add_argument("--latency-warmup", type=int, default=DEFAULT_LATENCY_WARMUP, help="Latency warmup iterations")
    parser.add_argument("--latency-iters", type=int, default=DEFAULT_LATENCY_ITERS, help="Latency benchmark iterations")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even on CUDA")
    parser.add_argument(
        "--isolate-models",
        action="store_true",
        help="Run every model in a separate worker process and continue after worker crashes/OOM",
    )
    parser.add_argument("--resume", action="store_true", help="Skip successful model summaries already in output-dir")
    parser.add_argument("--isolated-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON, help="Aggregate report JSON output")
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV, help="Aggregate report CSV output")
    parser.add_argument("--history-jsonl", type=Path, default=DEFAULT_HISTORY_JSONL, help="Aggregate history JSONL output")
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    if args.isolate_models and not args.isolated_worker:
        payload = _run_isolated_experiment(args)
        print(
            json.dumps(
                {
                    "successful": payload.get("successful"),
                    "failed": payload.get("failed"),
                    "failed_models": payload.get("failed_models"),
                    "summary": str(args.output_dir / "train_all_summary.json"),
                    "report_json": str(args.report_json),
                    "report_csv": str(args.report_csv),
                },
                indent=2,
            )
        )
        return
    payload = run_experiment(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        models=args.models,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        objective=args.objective,
        feature_mode=args.feature_mode,
        dim_reduction=args.dim_reduction,
        dim_components=args.dim_components,
        oversample=args.oversample,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=args.num_heads,
        kernel_size=args.kernel_size,
        tcn_blocks=args.tcn_blocks,
        cpu_threads=args.cpu_threads,
        xgb_threads=args.xgb_threads,
        latency_threads=args.latency_threads,
        latency_warmup=args.latency_warmup,
        latency_iters=args.latency_iters,
        amp_enabled=not args.no_amp,
        report_json=args.report_json,
        report_csv=args.report_csv,
        history_jsonl=args.history_jsonl,
        finalize_report=not args.isolated_worker,
    )
    if args.isolated_worker:
        worker_items = payload.get("items") or []
        compact = [
            {"model_name": item.get("model_name"), "status": item.get("status"), "error": item.get("error")}
            for item in worker_items
        ]
        print(json.dumps({"worker_items": compact}))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
