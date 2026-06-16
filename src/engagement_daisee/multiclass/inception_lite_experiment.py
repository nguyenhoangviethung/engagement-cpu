from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from engagement_daisee.common.config import FEATURE_DIM, HIDDEN_SIZE, LEARNING_RATE, RANDOM_SEED, WEIGHT_DECAY
from engagement_daisee.common.manifest import normalize_manifest_columns
from engagement_daisee.ml.train import _apply_feature_preprocessor, _build_feature_matrix, _fit_feature_preprocessor, _load_feature_preprocessor
from engagement_daisee.multiclass.train_all import (
    DEFAULT_CPU_THREADS,
    DEFAULT_LATENCY_ITERS,
    DEFAULT_LATENCY_THREADS,
    DEFAULT_LATENCY_WARMUP,
    NUM_CLASSES,
    _aggregate_by_video,
    _compute_multiclass_metrics,
    _split_indices,
)
from engagement_daisee.rnn.dataset import FeatureSequenceDataset


LOGGER = logging.getLogger("inception_lite_experiment")


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


def _timer_ms(fn, warmup: int, iters: int) -> dict[str, float]:
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


class InceptionTimeBlock(nn.Module):
    def __init__(self, channels: int, bottleneck_channels: int, kernel_sizes: tuple[int, int, int] = (3, 5, 7), dropout: float = 0.15):
        super().__init__()
        self.bottleneck = nn.Conv1d(channels, bottleneck_channels, kernel_size=1, bias=False) if bottleneck_channels > 0 else nn.Identity()
        conv_in = bottleneck_channels if bottleneck_channels > 0 else channels
        branch_channels = max(8, channels // 4)
        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(conv_in, branch_channels, kernel_size=k, padding=k // 2, bias=False)
                for k in kernel_sizes
            ]
        )
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(channels, branch_channels, kernel_size=1, bias=False),
        )
        total_out = branch_channels * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(total_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(channels, total_out, kernel_size=1, bias=False) if channels != total_out else nn.Identity()
        self.residual_bn = nn.BatchNorm1d(total_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_bn(self.residual(x))
        base = self.bottleneck(x)
        branches = [conv(base) for conv in self.conv_branches]
        branches.append(self.maxpool_branch(x))
        out = torch.cat(branches, dim=1)
        out = self.bn(out)
        out = self.activation(out + residual)
        return self.dropout(out)


class InceptionLiteClassifier(nn.Module):
    def __init__(self, input_size: int = FEATURE_DIM, hidden_size: int = 160, num_blocks: int = 4, dropout: float = 0.2, num_classes: int = NUM_CLASSES):
        super().__init__()
        stem_channels = hidden_size
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_size, stem_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
        )
        blocks = []
        channels = stem_channels
        for _ in range(num_blocks):
            block = InceptionTimeBlock(channels=channels, bottleneck_channels=max(8, channels // 4), dropout=dropout)
            blocks.append(block)
            channels = max(8, channels // 4) * 4
        self.blocks = nn.ModuleList(blocks)
        self.post = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.post(x)
        pooled = torch.cat([torch.mean(x, dim=-1), torch.amax(x, dim=-1)], dim=-1)
        return self.classifier(pooled)


@torch.no_grad()
def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
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


def _make_loader(dataset: FeatureSequenceDataset, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False, drop_last=False)


def _train_inception(
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
    min_balanced_accuracy: float,
    cpu_threads: int,
    latency_threads: int,
    latency_warmup: int,
    latency_iters: int,
    amp_enabled: bool,
    hidden_size: int,
    num_blocks: int,
    dropout: float,
) -> dict[str, object]:
    run_dir = output_dir / "inception_lite"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = InceptionLiteClassifier(
        input_size=FEATURE_DIM,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        dropout=dropout,
        num_classes=NUM_CLASSES,
    ).to(device)

    train_labels = _labels_from_subset(manifest, train_indices)
    class_weight_tensor = _class_weights(train_labels, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(1, patience // 2))
    amp_active = bool(amp_enabled and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_active)

    train_loader = _make_loader(dataset, train_indices, batch_size=batch_size, shuffle=True)
    train_eval_loader = _make_loader(dataset, train_indices, batch_size=batch_size, shuffle=False)
    val_loader = _make_loader(dataset, val_indices, batch_size=batch_size, shuffle=False)
    test_loader = _make_loader(dataset, test_indices, batch_size=batch_size, shuffle=False)

    feature_mean_cpu, feature_std_cpu = _compute_feature_stats(dataset, train_indices)
    feature_mean = feature_std = None
    if feature_mean_cpu is not None and feature_std_cpu is not None:
        feature_mean = feature_mean_cpu.to(device).view(1, 1, -1)
        feature_std = feature_std_cpu.to(device).view(1, 1, -1)

    best_score: tuple[int, float, float, float] | None = None
    best_epoch = 0
    best_state = None
    best_val_video_metrics: dict[str, object] | None = None
    history: list[dict[str, object]] = []

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
            if amp_active:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(features)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
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

        val_loss, val_labels_np, val_probabilities = _evaluate_model(
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
        val_manifest = manifest.iloc[val_indices].reset_index(drop=True)
        val_video_labels, val_video_probabilities = _aggregate_by_video(val_manifest, val_labels_np, val_probabilities)
        val_video_predictions = np.argmax(val_video_probabilities, axis=1)
        val_video_metrics = _compute_multiclass_metrics(val_video_labels, val_video_predictions, val_video_probabilities)
        current_rank = _rank_candidate(
            val_video_metrics,
            objective=objective,
            min_balanced_accuracy=min_balanced_accuracy,
        )
        scheduler.step(float(val_video_metrics[objective]))

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
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        LOGGER.info(
            "[inception_lite] epoch=%02d/%02d train_loss=%.4f train_acc=%.4f train_bal=%.4f train_f1=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_bal=%.4f val_f1=%.4f | score(%s)=%.4f lr=%.6f",
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
            float(val_video_metrics[objective]),
            optimizer.param_groups[0]["lr"],
        )

        if best_score is None or current_rank > best_score:
            best_score = current_rank
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_video_metrics = copy.deepcopy(val_video_metrics)

        if epoch >= min_epochs and (epoch - best_epoch) >= patience:
            LOGGER.info("[inception_lite] early stopping at epoch=%d best_epoch=%d", epoch, best_epoch)
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model = model.to(device)

    train_loss, train_labels_np, train_probabilities = _evaluate_model(
        model,
        train_eval_loader,
        criterion,
        device,
        feature_mean=feature_mean,
        feature_std=feature_std,
        amp_enabled=amp_active,
    )
    train_predictions = np.argmax(train_probabilities, axis=1)
    train_row_metrics = _compute_multiclass_metrics(train_labels_np, train_predictions, train_probabilities)

    val_loss, val_labels_np, val_probabilities = _evaluate_model(
        model,
        val_loader,
        criterion,
        device,
        feature_mean=feature_mean,
        feature_std=feature_std,
        amp_enabled=amp_active,
    )
    val_predictions = np.argmax(val_probabilities, axis=1)
    val_row_metrics = _compute_multiclass_metrics(val_labels_np, val_predictions, val_probabilities)
    val_manifest = manifest.iloc[val_indices].reset_index(drop=True)
    val_video_labels, val_video_probabilities = _aggregate_by_video(val_manifest, val_labels_np, val_probabilities)
    val_video_predictions = np.argmax(val_video_probabilities, axis=1)
    val_video_metrics = _compute_multiclass_metrics(val_video_labels, val_video_predictions, val_video_probabilities)

    test_loss, test_labels_np, test_probabilities = _evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        feature_mean=feature_mean,
        feature_std=feature_std,
        amp_enabled=amp_active,
    )
    test_predictions = np.argmax(test_probabilities, axis=1)
    test_row_metrics = _compute_multiclass_metrics(test_labels_np, test_predictions, test_probabilities)
    test_manifest = manifest.iloc[test_indices].reset_index(drop=True)
    test_video_labels, test_video_probabilities = _aggregate_by_video(test_manifest, test_labels_np, test_probabilities)
    test_video_predictions = np.argmax(test_video_probabilities, axis=1)
    test_video_metrics = _compute_multiclass_metrics(test_video_labels, test_video_predictions, test_video_probabilities)

    sample_path = Path(str(test_manifest.iloc[0]["feature_path"]))
    sample_sequence = np.load(sample_path).astype(np.float32)
    sample_tensor = torch.from_numpy(sample_sequence[None, ...])

    latency_model_cpu = model.to("cpu").eval()
    feature_mean_cpu_for_latency = feature_mean_cpu.detach().cpu()
    feature_std_cpu_for_latency = feature_std_cpu.detach().cpu()

    def _predict_new_model() -> np.ndarray:
        seq = sample_tensor
        seq = (seq - feature_mean_cpu_for_latency.view(1, 1, -1)) / feature_std_cpu_for_latency.view(1, 1, -1)
        with torch.no_grad():
            logits = latency_model_cpu(seq)
            return torch.softmax(logits, dim=-1).cpu().numpy()

    torch.set_num_threads(max(1, latency_threads))
    latency_model = _timer_ms(_predict_new_model, latency_warmup, latency_iters)

    payload = {
        "model_name": "inception_lite",
        "model_family": "neural",
        "status": "success",
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / "inception_lite.pt"),
        "model_kwargs": {
            "input_size": FEATURE_DIM,
            "hidden_size": hidden_size,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "num_classes": NUM_CLASSES,
        },
        "objective": objective,
        "best_epoch": int(best_epoch),
        "best_score": {
            "feasible": int(best_score[0]),
            "objective_value": float(best_score[1]),
            "balanced_accuracy": float(best_score[2]),
            "f1_macro": float(best_score[3]),
        },
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
            "model_side": {"variant": "inception_lite", **latency_model},
            "end_to_end": {"variant": "inception_lite", **latency_model},
        },
        "feature_normalization": {
            "enabled": True,
            "mean_path": str(run_dir / "feature_mean.pt"),
            "std_path": str(run_dir / "feature_std.pt"),
        },
        "device": str(device),
        "amp_enabled": amp_active,
        "cpu_threads": int(cpu_threads),
    }

    torch.save(
        {
            "model_name": "inception_lite",
            "model_kwargs": payload["model_kwargs"],
            "model_state_dict": best_state,
            "best_epoch": int(best_epoch),
            "objective": objective,
            "best_score": {
                "feasible": int(best_score[0]),
                "objective_value": float(best_score[1]),
                "balanced_accuracy": float(best_score[2]),
                "f1_macro": float(best_score[3]),
            },
            "best_val_video_metrics": best_val_video_metrics,
            "train_history": history,
            "train_counts": payload["train_counts"],
            "val_counts": payload["val_counts"],
            "test_counts": payload["test_counts"],
            "feature_normalization": payload["feature_normalization"],
            "feature_mean": feature_mean_cpu.numpy().tolist(),
            "feature_std": feature_std_cpu.numpy().tolist(),
            "latency": payload["latency"],
        },
        run_dir / "inception_lite.pt",
    )
    torch.save(feature_mean_cpu, run_dir / "feature_mean.pt")
    torch.save(feature_std_cpu, run_dir / "feature_std.pt")
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    LOGGER.info(
        "[inception_lite] test video metrics acc=%.4f bal=%.4f f1=%.4f | model latency=%.3f ms",
        float(test_video_metrics["accuracy"]),
        float(test_video_metrics["balanced_accuracy"]),
        float(test_video_metrics["f1_macro"]),
        float(latency_model["latency_ms_mean"]),
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


def _predict_xgb(model: XGBClassifier, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return model.predict_proba(x, iteration_range=(0, int(best_iteration) + 1)).astype(np.float32)
    return model.predict_proba(x).astype(np.float32)


def _load_xgb_component(manifest_df: pd.DataFrame, indices: list[int], run_root: Path, feature_mode: str) -> dict[str, object]:
    model_path = run_root / "xgboost" / "model.json"
    summary_path = run_root / "xgboost" / "summary.json"
    summary = json.loads(summary_path.read_text())
    resolved_feature_mode = str((summary.get("model_kwargs") or {}).get("feature_mode") or feature_mode)

    xgb = XGBClassifier()
    xgb.load_model(str(model_path))

    split_df = manifest_df.iloc[indices].reset_index(drop=True)
    x_eval, y_eval = _build_feature_matrix(split_df, feature_mode=resolved_feature_mode)
    preprocessor_path = None
    feature_norm = summary.get("feature_normalization") or {}
    if feature_norm.get("preprocessor_path"):
        preprocessor_path = Path(feature_norm["preprocessor_path"])
    elif (run_root / "xgboost" / "preprocessor.npz").exists():
        preprocessor_path = run_root / "xgboost" / "preprocessor.npz"

    if preprocessor_path is not None and preprocessor_path.exists():
        x_eval = _apply_feature_preprocessor(x_eval, _load_feature_preprocessor(preprocessor_path))

    probs = _predict_xgb(xgb, x_eval)
    video_labels, video_probs = _aggregate_by_video(split_df, y_eval, probs)
    video_predictions = np.argmax(video_probs, axis=1)
    return {
        "kind": "xgboost",
        "checkpoint_path": str(model_path),
        "summary_path": str(summary_path),
        "feature_mode": resolved_feature_mode,
        "model": xgb,
        "preprocessor_path": str(preprocessor_path) if preprocessor_path is not None else None,
        "labels": y_eval,
        "probs": probs,
        "video_labels": video_labels,
        "video_probs": video_probs,
        "row_metrics": _compute_multiclass_metrics(y_eval, np.argmax(probs, axis=1), probs),
        "video_metrics": _compute_multiclass_metrics(video_labels, video_predictions, video_probs),
        "sample_feature_path": str(split_df.iloc[0]["feature_path"]),
    }


def _align_scores(scores_by_model: dict[str, dict[str, object]]) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    common_ids = None
    for score in scores_by_model.values():
        ids = set(score["video_ids"])
        common_ids = ids if common_ids is None else common_ids.intersection(ids)
    if not common_ids:
        raise ValueError("No common videos available across ensemble components")
    ordered_ids = [vid for vid in scores_by_model[next(iter(scores_by_model))]["video_ids"] if vid in common_ids]
    labels = np.array([int(scores_by_model[next(iter(scores_by_model))]["video_label_map"][vid]) for vid in ordered_ids], dtype=np.int64)
    probs = {
        name: np.array([score["video_prob_map"][vid] for vid in ordered_ids], dtype=np.float32)
        for name, score in scores_by_model.items()
    }
    return ordered_ids, labels, probs


def _load_split_component_scores(
    manifest_path: Path,
    inception_checkpoint_path: Path,
    xgb_run_root: Path,
    split: str,
    *,
    batch_size: int,
    feature_mode: str,
) -> dict[str, dict[str, object]]:
    manifest_df = normalize_manifest_columns(pd.read_csv(manifest_path, low_memory=False))
    manifest_df["split"] = manifest_df["split"].astype(str).str.strip().str.lower()
    manifest_df["label"] = manifest_df["label"].astype(int)
    train_indices, val_indices, test_indices = _split_indices(manifest_df)
    indices = {"train": train_indices, "validation": val_indices, "test": test_indices}[split]
    dataset = FeatureSequenceDataset(manifest_path)
    component_specs = {
        "inception_lite": ("neural", inception_checkpoint_path),
        "xgboost": ("xgboost", xgb_run_root / "xgboost" / "model.json"),
    }

    results: dict[str, dict[str, object]] = {}
    for name, (kind, checkpoint_path) in component_specs.items():
        if kind == "neural":
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model_kwargs = dict(checkpoint.get("model_kwargs") or {})
            model = InceptionLiteClassifier(**model_kwargs).to("cpu").eval()
            state_dict = checkpoint.get("model_state_dict")
            if state_dict is None:
                raise ValueError(f"Missing model_state_dict in checkpoint: {checkpoint_path}")
            model.load_state_dict(state_dict)
            feature_mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32).view(1, 1, -1)
            feature_std = torch.tensor(checkpoint["feature_std"], dtype=torch.float32).view(1, 1, -1)
            loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
            labels_all: list[np.ndarray] = []
            probs_all: list[np.ndarray] = []
            with torch.no_grad():
                for features, labels in loader:
                    features = (features - feature_mean) / feature_std
                    logits = model(features)
                    probs = torch.softmax(logits, dim=-1)
                    labels_all.append(labels.cpu().numpy().astype(np.int64))
                    probs_all.append(probs.cpu().numpy().astype(np.float32))
            labels_np = np.concatenate(labels_all)
            probs_np = np.concatenate(probs_all)
            manifest_subset = manifest_df.iloc[indices].reset_index(drop=True)
            video_labels, video_probs = _aggregate_by_video(manifest_subset, labels_np, probs_np)
            video_ids_ordered = manifest_subset["video_id"].astype(str).drop_duplicates().tolist()
            results[name] = {
                "kind": "neural",
                "checkpoint_path": str(checkpoint_path),
                "model_name": "inception_lite",
                "model": model,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "labels": labels_np,
                "probs": probs_np,
                "video_ids": video_ids_ordered,
                "video_labels": video_labels,
                "video_probs": video_probs,
                "video_label_map": {vid: int(label) for vid, label in zip(video_ids_ordered, video_labels, strict=True)},
                "video_prob_map": {vid: video_probs[idx] for idx, vid in enumerate(video_ids_ordered)},
                "row_metrics": _compute_multiclass_metrics(labels_np, np.argmax(probs_np, axis=1), probs_np),
                "video_metrics": _compute_multiclass_metrics(video_labels, np.argmax(video_probs, axis=1), video_probs),
                "sample_feature_path": str(manifest_subset.iloc[0]["feature_path"]),
            }
        else:
            component = _load_xgb_component(manifest_df, indices, xgb_run_root, feature_mode)
            video_ids_ordered = manifest_df.iloc[indices].reset_index(drop=True)["video_id"].astype(str).drop_duplicates().tolist()
            results[name] = {
                **component,
                "video_ids": video_ids_ordered,
                "video_label_map": {vid: int(label) for vid, label in zip(video_ids_ordered, component["video_labels"], strict=True)},
                "video_prob_map": {vid: component["video_probs"][idx] for idx, vid in enumerate(video_ids_ordered)},
            }
    return results


def _build_latency_payload(
    manifest_path: Path,
    inception_checkpoint_path: Path,
    xgb_run_root: Path,
    fusion_weights: dict[str, float],
    batch_size: int,
    latency_threads: int,
    latency_warmup: int,
    latency_iters: int,
    feature_mode: str,
) -> dict[str, object]:
    manifest_df = normalize_manifest_columns(pd.read_csv(manifest_path, low_memory=False))
    manifest_df["split"] = manifest_df["split"].astype(str).str.strip().str.lower()
    manifest_df["label"] = manifest_df["label"].astype(int)
    _, _, test_indices = _split_indices(manifest_df)
    test_df = manifest_df.iloc[test_indices].reset_index(drop=True)
    sample_row = test_df.iloc[0]
    sample_feature_path = Path(str(sample_row["feature_path"]))
    sample_sequence = np.load(sample_feature_path).astype(np.float32)
    sample_tensor = torch.from_numpy(sample_sequence[None, ...]).to("cpu")

    checkpoint = torch.load(inception_checkpoint_path, map_location="cpu")
    inception_model = InceptionLiteClassifier(**dict(checkpoint["model_kwargs"])).to("cpu").eval()
    inception_model.load_state_dict(checkpoint["model_state_dict"])
    inception_mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32).view(1, 1, -1)
    inception_std = torch.tensor(checkpoint["feature_std"], dtype=torch.float32).view(1, 1, -1)

    xgb_dir = xgb_run_root / "xgboost"
    xgb_summary = json.loads((xgb_dir / "summary.json").read_text(encoding="utf-8"))
    resolved_feature_mode = str((xgb_summary.get("model_kwargs") or {}).get("feature_mode") or feature_mode)
    xgb = XGBClassifier()
    xgb.load_model(str(xgb_dir / "model.json"))
    xgb_preprocessor = None
    feature_norm = xgb_summary.get("feature_normalization") or {}
    preprocessor_path = Path(str(feature_norm["preprocessor_path"])) if feature_norm.get("preprocessor_path") else xgb_dir / "preprocessor.npz"
    if preprocessor_path.exists():
        xgb_preprocessor = _load_feature_preprocessor(preprocessor_path)
    sample_tabular = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode=resolved_feature_mode)[0]
    if xgb_preprocessor is not None:
        sample_tabular = _apply_feature_preprocessor(sample_tabular, xgb_preprocessor)

    inception_weight = float(fusion_weights["inception_lite"])
    xgb_weight = float(fusion_weights["xgboost"])
    active_components = [
        name
        for name, weight in (("inception_lite", inception_weight), ("xgboost", xgb_weight))
        if weight > 1e-8
    ]
    variant = "+".join(active_components)
    torch.set_num_threads(max(1, latency_threads))

    def _fuse_preloaded() -> np.ndarray:
        fused = np.zeros(NUM_CLASSES, dtype=np.float32)
        if inception_weight > 1e-8:
            with torch.no_grad():
                seq = (sample_tensor - inception_mean) / inception_std
                inception_probs = torch.softmax(inception_model(seq), dim=-1).cpu().numpy()[0]
            fused += inception_weight * inception_probs
        if xgb_weight > 1e-8:
            fused += xgb_weight * _predict_xgb(xgb, sample_tabular)[0]
        return fused

    def _fuse_from_disk() -> np.ndarray:
        fused = np.zeros(NUM_CLASSES, dtype=np.float32)
        if inception_weight > 1e-8:
            seq = np.load(sample_feature_path).astype(np.float32)
            seq_tensor = torch.from_numpy(seq[None, ...])
            with torch.no_grad():
                seq_tensor = (seq_tensor - inception_mean) / inception_std
                inception_probs = torch.softmax(inception_model(seq_tensor), dim=-1).cpu().numpy()[0]
            fused += inception_weight * inception_probs
        if xgb_weight > 1e-8:
            tabular = _build_feature_matrix(pd.DataFrame([sample_row.to_dict()]), feature_mode=resolved_feature_mode)[0]
            if xgb_preprocessor is not None:
                tabular = _apply_feature_preprocessor(tabular, xgb_preprocessor)
            fused += xgb_weight * _predict_xgb(xgb, tabular)[0]
        return fused

    model_side = _timer_ms(_fuse_preloaded, latency_warmup, latency_iters)
    end_to_end = _timer_ms(_fuse_from_disk, latency_warmup, latency_iters)
    return {
        "latency_kind": "processed_feature_sequence",
        "sample_feature_path": str(sample_feature_path),
        "fusion_weights": {name: float(weight) for name, weight in fusion_weights.items()},
        "model_side": {"variant": variant, **model_side},
        "end_to_end": {"variant": variant, **end_to_end},
    }


def _fuse_probs(probs_by_model: dict[str, np.ndarray], names: list[str], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack([probs_by_model[name] for name in names], axis=0)
    return np.sum(stacked * weights[:, None, None], axis=0)


def _weight_grid(n_models: int, step: float) -> list[np.ndarray]:
    ticks = int(round(1.0 / step))
    weights = []
    for raw in np.ndindex(*([ticks + 1] * n_models)):
        if sum(raw) == ticks:
            weights.append(np.array(raw, dtype=np.float32) / ticks)
    return weights


def _rank_candidate(metrics: dict[str, object], *, objective: str, min_balanced_accuracy: float) -> tuple[int, float, float, float]:
    balanced_accuracy = float(metrics["balanced_accuracy"])
    feasible = int(balanced_accuracy >= min_balanced_accuracy)
    objective_value = float(metrics[objective])
    return (
        feasible,
        objective_value,
        balanced_accuracy,
        float(metrics["f1_macro"]),
    )


def run_experiment(
    manifest_path: Path,
    output_dir: Path,
    xgb_run_root: Path,
    *,
    device: str,
    batch_size: int,
    epochs: int,
    patience: int,
    min_epochs: int,
    lr: float,
    weight_decay: float,
    objective: str,
    min_balanced_accuracy: float,
    feature_mode: str,
    hidden_size: int,
    num_blocks: int,
    dropout: float,
    cpu_threads: int,
    latency_threads: int,
    latency_warmup: int,
    latency_iters: int,
    amp_enabled: bool,
    report_json: Path,
    report_csv: Path | None,
) -> dict[str, object]:
    _set_seed(RANDOM_SEED)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    for required_path in (
        xgb_run_root / "xgboost" / "model.json",
        xgb_run_root / "xgboost" / "summary.json",
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required XGBoost artifact not found: {required_path}")
    if batch_size < 1 or epochs < 1 or patience < 1 or min_epochs < 1:
        raise ValueError("batch_size, epochs, patience, and min_epochs must all be positive.")
    if min_epochs > epochs:
        raise ValueError("min_epochs cannot be greater than epochs.")
    if not 0.0 <= min_balanced_accuracy <= 1.0:
        raise ValueError("min_balanced_accuracy must be between 0 and 1.")
    if objective not in {"accuracy", "balanced_accuracy", "f1_macro"}:
        raise ValueError(f"Unsupported objective: {objective}")

    torch.set_num_threads(max(1, cpu_threads))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = normalize_manifest_columns(pd.read_csv(manifest_path, low_memory=False))
    required_columns = {"feature_path", "label", "split"}
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest["label"] = manifest["label"].astype(int)
    invalid_labels = sorted(set(manifest["label"].tolist()) - set(range(NUM_CLASSES)))
    if invalid_labels:
        raise ValueError(f"Manifest labels must be in [0, {NUM_CLASSES - 1}], found: {invalid_labels}")
    train_indices, val_indices, test_indices = _split_indices(manifest)
    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Manifest must contain non-empty train, validation, and test splits.")

    dataset = FeatureSequenceDataset(manifest_path)
    if len(dataset) != len(manifest):
        raise RuntimeError(
            "FeatureSequenceDataset skipped one or more manifest rows; refusing to train with misaligned indices. "
            f"manifest_rows={len(manifest)} loaded_rows={len(dataset)}"
        )
    LOGGER.info(
        "Loaded manifest | rows=%d videos=%s train=%d val=%d test=%d device=%s xgb_run_root=%s",
        len(manifest),
        int(manifest["video_id"].nunique()) if "video_id" in manifest.columns else None,
        len(train_indices),
        len(val_indices),
        len(test_indices),
        device,
        xgb_run_root,
    )

    device_obj = _resolve_device(device)
    LOGGER.info(
        "Selection policy | objective=%s min_balanced_accuracy=%.4f resolved_device=%s amp=%s",
        objective,
        min_balanced_accuracy,
        device_obj,
        bool(amp_enabled and device_obj.type == "cuda"),
    )
    inception_item = _train_inception(
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
        min_balanced_accuracy=min_balanced_accuracy,
        cpu_threads=cpu_threads,
        latency_threads=latency_threads,
        latency_warmup=latency_warmup,
        latency_iters=latency_iters,
        amp_enabled=amp_enabled,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        dropout=dropout,
    )

    inception_checkpoint_path = Path(inception_item["checkpoint_path"])
    val_scores = _load_split_component_scores(
        manifest_path,
        inception_checkpoint_path,
        xgb_run_root,
        "validation",
        batch_size=batch_size,
        feature_mode=feature_mode,
    )
    test_scores = _load_split_component_scores(
        manifest_path,
        inception_checkpoint_path,
        xgb_run_root,
        "test",
        batch_size=batch_size,
        feature_mode=feature_mode,
    )
    val_ids, val_labels, val_probs = _align_scores(val_scores)
    test_ids, test_labels, test_probs = _align_scores(test_scores)

    names = ["inception_lite", "xgboost"]
    best: dict[str, object] | None = None
    best_rank: tuple[int, float, float, float] | None = None
    for weights in _weight_grid(len(names), 0.05):
        fused = _fuse_probs(val_probs, names, weights)
        val_predictions = np.argmax(fused, axis=1)
        val_metrics = _compute_multiclass_metrics(val_labels, val_predictions, fused)
        candidate = {
            "weights": {name: float(weight) for name, weight in zip(names, weights, strict=True)},
            "validation_metrics": val_metrics,
        }
        candidate_rank = _rank_candidate(val_metrics, objective=objective, min_balanced_accuracy=min_balanced_accuracy)
        if best is None or best_rank is None or candidate_rank > best_rank:
            best = candidate
            best_rank = candidate_rank

    assert best is not None
    selected_weights = np.array([best["weights"][name] for name in names], dtype=np.float32)
    fused_test = _fuse_probs(test_probs, names, selected_weights)
    test_predictions = np.argmax(fused_test, axis=1)
    test_metrics = _compute_multiclass_metrics(test_labels, test_predictions, fused_test)

    component_reports = {
        "inception_lite": {
            "validation": inception_item["val_video_metrics"],
            "test": inception_item["test_video_metrics"],
        },
        "xgboost": {
            "validation": val_scores["xgboost"]["video_metrics"],
            "test": test_scores["xgboost"]["video_metrics"],
        },
    }

    latency = _build_latency_payload(
        manifest_path=manifest_path,
        inception_checkpoint_path=inception_checkpoint_path,
        xgb_run_root=xgb_run_root,
        fusion_weights=best["weights"],
        batch_size=batch_size,
        latency_threads=latency_threads,
        latency_warmup=latency_warmup,
        latency_iters=latency_iters,
        feature_mode=feature_mode,
    )

    report = {
        "manifest": str(manifest_path),
        "xgb_run_root": str(xgb_run_root),
        "models": {
            "inception_lite": str(output_dir / "inception_lite" / "inception_lite.pt"),
            "xgboost": str(xgb_run_root / "xgboost" / "model.json"),
        },
        "video_counts": {"validation": int(len(val_ids)), "test": int(len(test_ids))},
        "selected": {
            "weights": best["weights"],
            "validation_metrics": best["validation_metrics"],
            "validation_loss": float(best["validation_metrics"]["cross_entropy"]),
            "objective": objective,
            "minimum_balanced_accuracy": float(min_balanced_accuracy),
            "balanced_accuracy_constraint_met": bool(
                float(best["validation_metrics"]["balanced_accuracy"]) >= min_balanced_accuracy
            ),
        },
        "test_metrics": test_metrics,
        "test_loss": float(test_metrics["cross_entropy"]),
        "component_reports": component_reports,
        "latency": latency,
        "inception_item": inception_item,
        "note": "Weights were selected on validation only; test is held out for final reporting.",
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report_csv is not None:
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "model_name": "inception_lite",
                    "test_accuracy": inception_item["test_video_metrics"]["accuracy"],
                    "test_balanced_accuracy": inception_item["test_video_metrics"]["balanced_accuracy"],
                    "test_f1_macro": inception_item["test_video_metrics"]["f1_macro"],
                },
                {
                    "model_name": "xgboost",
                    "test_accuracy": test_scores["xgboost"]["video_metrics"]["accuracy"],
                    "test_balanced_accuracy": test_scores["xgboost"]["video_metrics"]["balanced_accuracy"],
                    "test_f1_macro": test_scores["xgboost"]["video_metrics"]["f1_macro"],
                },
                {
                    "model_name": "fusion",
                    "test_accuracy": test_metrics["accuracy"],
                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "test_f1_macro": test_metrics["f1_macro"],
                },
            ]
        ).to_csv(report_csv, index=False)

    LOGGER.info(
        "[fusion] test video metrics acc=%.4f bal=%.4f f1=%.4f | model latency=%.3f ms | e2e=%.3f ms",
        float(test_metrics["accuracy"]),
        float(test_metrics["balanced_accuracy"]),
        float(test_metrics["f1_macro"]),
        float(latency["model_side"]["latency_ms_mean"]),
        float(latency["end_to_end"]["latency_ms_mean"]),
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an InceptionTime-lite experiment and fuse it with the latest strong XGBoost checkpoint.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/runs/daisee_4class_final_dataset/feature_manifest.csv"))
    parser.add_argument(
        "--xgb-run-root",
        type=Path,
        default=Path("checkpoints/runs/train_all_4class_gpu_final"),
        help="Run root containing the existing 4-class XGBoost checkpoint to fuse with.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/runs/inception_lite_ensemble_4class"))
    parser.add_argument("--report-json", type=Path, default=Path("checkpoints/runs/inception_lite_ensemble_4class/report.json"))
    parser.add_argument("--report-csv", type=Path, default=Path("checkpoints/runs/inception_lite_ensemble_4class/report.csv"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE * 0.8333333333333334)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--objective", type=str, default="accuracy", choices=["accuracy", "balanced_accuracy", "f1_macro"])
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.70, help="Minimum balanced accuracy required when selecting the best checkpoint or ensemble.")
    parser.add_argument("--feature-mode", type=str, default="tsfresh", choices=["basic", "tsfresh", "copur"])
    parser.add_argument("--hidden-size", type=int, default=max(128, HIDDEN_SIZE))
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--latency-threads", type=int, default=DEFAULT_LATENCY_THREADS)
    parser.add_argument("--latency-warmup", type=int, default=DEFAULT_LATENCY_WARMUP)
    parser.add_argument("--latency-iters", type=int, default=DEFAULT_LATENCY_ITERS)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    run_experiment(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        xgb_run_root=args.xgb_run_root,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        objective=args.objective,
        min_balanced_accuracy=args.min_balanced_accuracy,
        feature_mode=args.feature_mode,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        cpu_threads=args.cpu_threads,
        latency_threads=args.latency_threads,
        latency_warmup=args.latency_warmup,
        latency_iters=args.latency_iters,
        amp_enabled=not args.no_amp,
        report_json=args.report_json,
        report_csv=args.report_csv,
    )


if __name__ == "__main__":
    main()
