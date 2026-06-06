import argparse
from contextlib import nullcontext
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from engagement_daisee.common.config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    FEATURE_MANIFEST_CSV,
    GRAD_CLIP_NORM,
    HIDDEN_SIZE,
    DROPOUT,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MODEL_CHECKPOINT_PATH,
    NUM_WORKERS,
    RANDOM_SEED,
    SAMPLE_EPOCHS,
    WEIGHT_DECAY,
)
from engagement_daisee.rnn.dataset import FeatureSequenceDataset
from engagement_daisee.rnn.models.builder import build_sequence_model


LOGGER = logging.getLogger("train")
MODEL_NUM_LAYERS = 2
DEFAULT_CPU_THREADS = 2
RUNS_PROCESSED_DIR = FEATURE_MANIFEST_CSV.parent / "runs"


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        # alpha controls class balance.
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def _objective_score(metrics: dict[str, float], objective: str, min_recall_pos: float) -> float:
    if objective == "balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    if objective == "accuracy":
        return float(metrics["accuracy"])
    if objective == "f1_macro":
        return float(metrics["f1_macro"])
    if objective == "focused_recall":
        # Prefer candidates that satisfy recall target, then rank by class-1 F1.
        if metrics["recall_pos"] >= min_recall_pos:
            return 1.0 + float(metrics["f1_pos"])
        return float(metrics["recall_pos"])
    return float(metrics["f1_pos"])


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 1e-4
    best_score: float = field(default=-math.inf)
    bad_epochs: int = field(default=0)

    def step(self, current_score: float, epoch: int, min_epochs: int = 1) -> bool:
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.bad_epochs = 0
            return False
        if epoch < min_epochs:
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


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


def _official_split_indices(manifest: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    if "split" not in manifest.columns:
        raise ValueError(
            "Manifest is missing required 'split' column. "
            "Re-run preprocess_labels.py and extract_features.py to generate split-aware data."
        )

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


def _resolve_manifest_path(manifest_csv: Path, run_id: str | None) -> Path:
    if run_id is None and manifest_csv.exists():
        return manifest_csv

    candidate_paths: list[Path] = []
    if run_id:
        candidate_paths.extend(
            [
                RUNS_PROCESSED_DIR / f"train_{run_id}" / "feature_manifest.csv",
                RUNS_PROCESSED_DIR / f"pipeline_{run_id}" / "feature_manifest.csv",
                RUNS_PROCESSED_DIR / f"extract_{run_id}" / "feature_manifest.csv",
            ]
        )

    candidate_paths.append(manifest_csv)
    if manifest_csv != FEATURE_MANIFEST_CSV:
        candidate_paths.append(FEATURE_MANIFEST_CSV)

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            LOGGER.info("Resolved manifest path: %s", candidate_path)
            return candidate_path

    searched_paths = "\n".join(f"- {path}" for path in candidate_paths)
    raise FileNotFoundError(
        "Could not locate a feature manifest. Searched:\n"
        f"{searched_paths}\n"
        "Pass --manifest explicitly or generate features with extract/pipeline first."
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _clip_probabilities(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(probabilities.astype(np.float64), eps, 1.0 - eps)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probs = _clip_probabilities(probabilities)
    return np.log(probs / (1.0 - probs))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _calibrate_probabilities(
    probabilities: np.ndarray,
    *,
    temperature: float = 1.0,
    source_pos_prior: float | None = None,
    target_pos_prior: float | None = None,
) -> np.ndarray:
    probs = _clip_probabilities(probabilities)
    temp = max(1e-3, float(temperature))
    calibrated = _sigmoid(_logit(probs) / temp)

    if source_pos_prior is None or target_pos_prior is None:
        return calibrated.astype(np.float32)

    source = float(np.clip(source_pos_prior, 1e-4, 1.0 - 1e-4))
    target = float(np.clip(target_pos_prior, 1e-4, 1.0 - 1e-4))
    if abs(source - target) <= 1e-8:
        return calibrated.astype(np.float32)

    source_odds = source / (1.0 - source)
    target_odds = target / (1.0 - target)
    odds_multiplier = target_odds / source_odds

    odds = calibrated / (1.0 - calibrated)
    adjusted_odds = odds * odds_multiplier
    adjusted = adjusted_odds / (1.0 + adjusted_odds)
    return _clip_probabilities(adjusted).astype(np.float32)


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probs = _clip_probabilities(probabilities)
    labels = labels.astype(np.float64)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def _resolve_temperature_grid(spec: str) -> list[float]:
    values: list[float] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        values = [1.0]
    return sorted(set(max(1e-3, v) for v in values))


def _find_best_temperature(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    temperatures: Iterable[float],
    source_pos_prior: float | None,
    target_pos_prior: float | None,
) -> tuple[float, float]:
    best_temp = 1.0
    best_loss = math.inf
    for temp in temperatures:
        calibrated = _calibrate_probabilities(
            probabilities,
            temperature=float(temp),
            source_pos_prior=source_pos_prior,
            target_pos_prior=target_pos_prior,
        )
        loss = _log_loss(labels, calibrated)
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp, best_loss


def _compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    precision_pos = _safe_div(tp, tp + fp)
    recall_pos = _safe_div(tp, tp + fn)
    f1_pos = _safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    precision_neg = _safe_div(tn, tn + fn)
    recall_neg = _safe_div(tn, tn + fp)
    f1_neg = _safe_div(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (recall_pos + recall_neg) / 2.0
    f1_macro = (f1_pos + f1_neg) / 2.0

    return {
        "accuracy": accuracy,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "f1_neg": f1_neg,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": f1_macro,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _find_best_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    objective: str = "f1_pos",
    min_recall_pos: float = 0.75,
) -> tuple[float, dict[str, float]]:
    thresholds = np.arange(0.1, 0.901, 0.01)
    best_threshold = 0.5
    best_metrics = _compute_binary_metrics(labels, probabilities, threshold=0.5)
    found_recall_feasible = False

    def _is_better(metrics: dict[str, float], incumbent: dict[str, float]) -> bool:
        return _is_better_for_objective(
            metrics,
            incumbent,
            objective=objective,
            min_recall_pos=min_recall_pos,
        )

    for threshold in thresholds:
        metrics = _compute_binary_metrics(labels, probabilities, threshold=float(threshold))
        if objective == "focused_recall" and metrics["recall_pos"] >= min_recall_pos:
            found_recall_feasible = True
        if _is_better(metrics, best_metrics):
            best_threshold = float(threshold)
            best_metrics = metrics

    if objective == "focused_recall" and not found_recall_feasible:
        LOGGER.warning(
            "No validation threshold reached recall_pos >= %.2f; using best available recall-based threshold.",
            min_recall_pos,
        )
    return best_threshold, best_metrics


def _is_better_for_objective(
    candidate: dict[str, float],
    incumbent: dict[str, float],
    objective: str,
    min_recall_pos: float,
) -> bool:
    if objective == "focused_recall":
        candidate_feasible = candidate["recall_pos"] >= min_recall_pos
        incumbent_feasible = incumbent["recall_pos"] >= min_recall_pos
        if candidate_feasible and not incumbent_feasible:
            return True
        if candidate_feasible and incumbent_feasible:
            if candidate["f1_pos"] > incumbent["f1_pos"] + 1e-8:
                return True
            if abs(candidate["f1_pos"] - incumbent["f1_pos"]) <= 1e-8:
                return candidate["precision_pos"] > incumbent["precision_pos"] + 1e-8
            return False
        if not candidate_feasible and incumbent_feasible:
            return False
        if candidate["recall_pos"] > incumbent["recall_pos"] + 1e-8:
            return True
        if abs(candidate["recall_pos"] - incumbent["recall_pos"]) <= 1e-8:
            return candidate["f1_pos"] > incumbent["f1_pos"] + 1e-8
        return False

    if objective == "balanced_accuracy":
        if candidate["balanced_accuracy"] > incumbent["balanced_accuracy"] + 1e-8:
            return True
        if abs(candidate["balanced_accuracy"] - incumbent["balanced_accuracy"]) <= 1e-8:
            return candidate["recall_pos"] > incumbent["recall_pos"] + 1e-8
        return False

    if objective == "accuracy":
        if candidate["accuracy"] > incumbent["accuracy"] + 1e-8:
            return True
        if abs(candidate["accuracy"] - incumbent["accuracy"]) <= 1e-8:
            return candidate["balanced_accuracy"] > incumbent["balanced_accuracy"] + 1e-8
        return False

    if objective == "f1_macro":
        if candidate["f1_macro"] > incumbent["f1_macro"] + 1e-8:
            return True
        if abs(candidate["f1_macro"] - incumbent["f1_macro"]) <= 1e-8:
            return candidate["balanced_accuracy"] > incumbent["balanced_accuracy"] + 1e-8
        return False

    # Default objective: maximize class-1 F1 (focused class quality).
    if candidate["f1_pos"] > incumbent["f1_pos"] + 1e-8:
        return True
    if abs(candidate["f1_pos"] - incumbent["f1_pos"]) <= 1e-8:
        if candidate["recall_pos"] > incumbent["recall_pos"] + 1e-8:
            return True
        if abs(candidate["recall_pos"] - incumbent["recall_pos"]) <= 1e-8:
            return candidate["balanced_accuracy"] > incumbent["balanced_accuracy"] + 1e-8
    return False

def _make_loader(
    dataset: FeatureSequenceDataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _make_train_loader(
    dataset: FeatureSequenceDataset,
    manifest_df: pd.DataFrame,
    indices: list[int],
    batch_size: int,
    sampler_strategy: str = "weighted",
    pin_memory: bool = False,
) -> DataLoader:
    subset = Subset(dataset, indices)
    labels = manifest_df.iloc[indices]["label"].astype(int).to_numpy()
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)

    if class_counts[0] == 0 or class_counts[1] == 0:
        split_table = pd.crosstab(manifest_df["split"], manifest_df["label"]) if "split" in manifest_df.columns else None
        details = ""
        if split_table is not None:
            details = f"\nSplit/label table:\n{split_table.to_string()}"
        raise ValueError(
            "Train split must contain both classes. "
            f"Got class0={int(class_counts[0])}, class1={int(class_counts[1])}. "
            "This often means features were extracted from a small/sample subset. "
            "Re-run full extraction without --sample to train on full data."
            f"{details}"
        )

    if sampler_strategy == "shuffle":
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            persistent_workers=True if NUM_WORKERS > 0 else False,
            pin_memory=pin_memory,
            drop_last=False,
        )

    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _compute_feature_stats(
    dataset: FeatureSequenceDataset,
    indices: list[int],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not indices:
        raise ValueError("Cannot compute feature stats from an empty index list.")

    sample_feature, _ = dataset[indices[0]]
    feature_dim = int(sample_feature.shape[-1])
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


def _build_criterion(
    loss_name: str,
    positives: float,
    negatives: float,
    focal_alpha: float | None,
    focal_gamma: float,
    device: str,
) -> tuple[nn.Module, dict[str, float]]:
    class_total = max(1.0, positives + negatives)
    pos_ratio = positives / class_total
    neg_ratio = negatives / class_total

    if loss_name == "bce_weighted":
        pos_weight_value = negatives / max(1.0, positives)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32)).to(device)
        return criterion, {"pos_weight": float(pos_weight_value), "pos_ratio": float(pos_ratio), "neg_ratio": float(neg_ratio)}

    resolved_alpha = focal_alpha
    if resolved_alpha is None:
        resolved_alpha = float(np.clip(neg_ratio, 0.05, 0.95))
    criterion = FocalLoss(alpha=resolved_alpha, gamma=focal_gamma).to(device)
    return criterion, {"focal_alpha": float(resolved_alpha), "focal_gamma": float(focal_gamma), "pos_ratio": float(pos_ratio), "neg_ratio": float(neg_ratio)}


def _build_model(
    model_name: str,
    input_size: int,
    hidden_size: int,
    dropout: float,
    num_layers: int = MODEL_NUM_LAYERS,
    num_heads: int = 4,
    kernel_size: int = 3,
    tcn_kernel_size: int | None = None,
    tcn_blocks: int = 3,
    max_seq_len: int = 64,
) -> nn.Module:
    resolved_kernel_size = int(tcn_kernel_size) if tcn_kernel_size is not None else int(kernel_size)
    return build_sequence_model(
        model_name=model_name,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        kernel_size=resolved_kernel_size,
        tcn_blocks=tcn_blocks,
        max_seq_len=max_seq_len,
    )


def _train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    amp_enabled: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    threshold: float = 0.5,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
):
    model.train()
    running_loss = 0.0
    running_total = 0
    all_probabilities = []
    all_labels = []

    for features, labels in tqdm(loader, desc="Training", unit="batch"):
        features = features.to(device)
        if feature_mean is not None and feature_std is not None:
            features = (features - feature_mean) / feature_std
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if amp_enabled and device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            logits = model(features)
            loss = criterion(logits, labels)

        if scaler is not None and amp_enabled and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        probabilities = torch.sigmoid(logits)
        running_total += batch_size

        all_probabilities.append(probabilities.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    average_loss = running_loss / max(1, running_total)
    stacked_probabilities = np.concatenate(all_probabilities).astype(np.float32)
    stacked_labels = np.concatenate(all_labels).astype(np.int64)
    metrics = _compute_binary_metrics(stacked_labels, stacked_probabilities, threshold=threshold)
    return average_loss, metrics


@torch.no_grad()
def _evaluate(
    model,
    loader,
    criterion,
    device: torch.device,
    threshold: float,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
    amp_enabled: bool = False,
):
    model.eval()
    running_loss = 0.0
    running_total = 0
    all_probabilities = []
    all_labels = []

    for features, labels in tqdm(loader, desc="Evaluating", unit="batch"):
        features = features.to(device)
        if feature_mean is not None and feature_std is not None:
            features = (features - feature_mean) / feature_std
        labels = labels.to(device)
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if amp_enabled and device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            logits = model(features)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        probabilities = torch.sigmoid(logits)
        running_total += batch_size

        all_probabilities.append(probabilities.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    average_loss = running_loss / max(1, running_total)
    stacked_probabilities = np.concatenate(all_probabilities).astype(np.float32)
    stacked_labels = np.concatenate(all_labels).astype(np.int64)
    metrics = _compute_binary_metrics(stacked_labels, stacked_probabilities, threshold=threshold)
    return average_loss, metrics, stacked_labels, stacked_probabilities


def _build_checkpoint_payload(
    *,
    model: nn.Module,
    model_kwargs: dict,
    model_name: str,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    early_stopping: EarlyStopping,
    best_threshold: float,
    threshold_objective: str,
    min_recall_pos: float,
    loss_name: str,
    criterion_config: dict[str, float],
    sampler_strategy: str,
    normalize_features: bool,
    feature_mean_list: list[float] | None,
    feature_std_list: list[float] | None,
    best_objective_score: float,
    best_val_precision_pos: float,
    best_val_balanced_accuracy: float,
    best_val_f1_macro: float,
    best_val_metrics_for_objective: dict[str, float] | None,
    history: list[dict],
    device: torch.device,
    amp_enabled: bool,
    cpu_threads: int,
    learning_rate: float,
    weight_decay: float,
    scheduler_name: str,
    freeze_feature_epochs: int,
) -> dict:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "model_name": model_name,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "early_stopping_bad_epochs": int(early_stopping.bad_epochs),
        "best_threshold": best_threshold,
        "threshold_objective": threshold_objective,
        "min_recall_pos": min_recall_pos,
        "loss_name": loss_name,
        "criterion_config": criterion_config,
        "sampler_strategy": sampler_strategy,
        "normalize_features": normalize_features,
        "device": str(device),
        "amp_enabled": bool(amp_enabled and device.type == "cuda"),
        "cpu_threads": int(cpu_threads),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "scheduler_name": scheduler_name,
        "freeze_feature_epochs": int(freeze_feature_epochs),
        "feature_mean": feature_mean_list,
        "feature_std": feature_std_list,
        "best_objective_score": best_objective_score,
        "best_val_precision_pos": best_val_precision_pos,
        "best_val_balanced_accuracy": best_val_balanced_accuracy,
        "best_val_f1_macro": best_val_f1_macro,
        "best_val_metrics_for_objective": best_val_metrics_for_objective,
        "history": history,
    }


def _set_frontend_trainable(model: nn.Module, trainable: bool) -> int:
    toggled = 0
    for module_name in ("feature_encoder", "feature_extractor", "input_proj", "temporal_frontend"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = trainable
            toggled += 1
    return toggled


def train(
    manifest_csv: Path,
    output_path: Path,
    sample: bool = False,
    seed: int = RANDOM_SEED,
    log_every: int = 1,
    run_id: str | None = None,
    threshold_objective: str = "balanced_accuracy",
    min_recall_pos: float = 0.75,
    loss_name: str = "bce_weighted",
    focal_alpha: float | None = None,
    focal_gamma: float = 2.0,
    sampler_strategy: str = "weighted",
    normalize_features: bool = True,
    model_name: str = "gru",
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = MODEL_NUM_LAYERS,
    dropout: float = DROPOUT,
    num_heads: int = 4,
    tcn_kernel_size: int = 3,
    tcn_blocks: int = 3,
    batch_size: int = BATCH_SIZE,
    epochs: int | None = None,
    patience: int | None = None,
    min_epochs: int = 1,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    device_arg: str = DEVICE,
    amp_enabled: bool = True,
    resume_from: Path | None = None,
    enable_prior_shift_calibration: bool = True,
    target_pos_prior: float | None = None,
    calibration_temperature_grid: str = "0.75,0.9,1.0,1.1,1.25,1.5",
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    scheduler_name: str = "plateau",
    freeze_feature_epochs: int = 0,
) -> Path:
    _set_seed(seed)
    device = _resolve_device(device_arg)
    resolved_threads = max(1, min(cpu_threads, os.cpu_count() or cpu_threads))
    torch.set_num_threads(resolved_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    manifest_csv = _resolve_manifest_path(manifest_csv, run_id)

    LOGGER.info(
        "Starting training | manifest=%s | output=%s | sample=%s | seed=%d | model=%s | cpu_threads=%d | threshold_objective=%s | min_recall_pos=%.2f | loss=%s | sampler=%s | normalize=%s | resume_from=%s",
        manifest_csv,
        output_path,
        sample,
        seed,
        model_name,
        resolved_threads,
        threshold_objective,
        min_recall_pos,
        loss_name,
        sampler_strategy,
        normalize_features,
        str(resume_from) if resume_from is not None else None,
    )
    LOGGER.info("Resolved device=%s | amp_enabled=%s", device, amp_enabled)

    dataset = FeatureSequenceDataset(manifest_csv)
    manifest_df = dataset.manifest
    if len(manifest_df) < 1000:
        LOGGER.warning(
            "Manifest has only %d rows. This is unusually small for DAiSEE full extraction and can cause fast/unstable training.",
            len(manifest_df),
        )
    train_indices, val_indices, test_indices = _official_split_indices(manifest_df)
    if sample:
        rng = random.Random(seed)
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)
        train_indices = train_indices[: max(1, min(len(train_indices), BATCH_SIZE * 4))]
        val_indices = val_indices[: max(1, min(len(val_indices), BATCH_SIZE * 2))]
        test_indices = test_indices[: max(1, min(len(test_indices), BATCH_SIZE * 2))]

    batch_size = max(1, int(batch_size))
    epochs = SAMPLE_EPOCHS if sample else int(epochs or EPOCHS)
    patience = 2 if sample else int(patience or EARLY_STOPPING_PATIENCE)
    min_epochs = 1 if sample else max(1, min(int(min_epochs), epochs))

    train_loader = _make_train_loader(
        dataset,
        manifest_df,
        train_indices,
        batch_size=batch_size,
        sampler_strategy=sampler_strategy,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = _make_loader(dataset, val_indices, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))
    test_loader = _make_loader(dataset, test_indices, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    LOGGER.info(
        "Data splits | train=%d | validation=%d | test=%d | batch_size=%d",
        len(train_indices),
        len(val_indices),
        len(test_indices),
        batch_size,
    )

    split_priors = {}
    for split_name, indices in (("train", train_indices), ("validation", val_indices), ("test", test_indices)):
        split_labels = manifest_df.iloc[indices]["label"].astype(float).to_numpy()
        split_priors[split_name] = float(split_labels.mean())
    train_pos_prior = split_priors["train"]
    val_pos_prior = split_priors["validation"]
    test_pos_prior = split_priors["test"]
    resolved_target_pos_prior = float(target_pos_prior) if target_pos_prior is not None else train_pos_prior
    temp_grid = _resolve_temperature_grid(calibration_temperature_grid)
    LOGGER.info(
        "Priors | train=%.4f validation=%.4f test=%.4f | target_prior=%.4f | prior_shift_calibration=%s | temperature_grid=%s",
        train_pos_prior,
        val_pos_prior,
        test_pos_prior,
        resolved_target_pos_prior,
        enable_prior_shift_calibration,
        temp_grid,
    )

    train_labels = manifest_df.iloc[train_indices]["label"].astype(float).to_numpy()
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError(
            "Train split has only one class, cannot train a robust binary classifier. "
            f"positives={int(positives)}, negatives={int(negatives)}"
        )

    LOGGER.info(
        "Class balance | positives=%.0f | negatives=%.0f | pos_ratio=%.4f",
        positives,
        negatives,
        positives / max(1.0, positives + negatives),
    )

    feature_mean = None
    feature_std = None
    if normalize_features:
        mean_cpu, std_cpu = _compute_feature_stats(dataset, train_indices)
        feature_mean = mean_cpu.to(device).view(1, 1, -1)
        feature_std = std_cpu.to(device).view(1, 1, -1)
        LOGGER.info(
            "Feature normalization enabled | mean_abs=%.6f | std_min=%.6f | std_median=%.6f",
            float(mean_cpu.abs().mean().item()),
            float(std_cpu.min().item()),
            float(std_cpu.median().item()),
        )

    sample_features, _ = dataset[0]
    model_kwargs = {
        "model_name": model_name,
        "input_size": sample_features.shape[-1],
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "num_heads": num_heads,
        "kernel_size": tcn_kernel_size,
        "tcn_blocks": tcn_blocks,
        "max_seq_len": int(sample_features.shape[0]),
    }
    model = _build_model(**model_kwargs).to(device)
    criterion, criterion_config = _build_criterion(
        loss_name=loss_name,
        positives=positives,
        negatives=negatives,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        device=str(device),
    )
    LOGGER.info("Criterion config: %s", criterion_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_name = scheduler_name.strip().lower()
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=max(1e-7, learning_rate * 0.05),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    early_stopping = EarlyStopping(patience=patience)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = output_path.with_name(f"{output_path.stem}.last.pt")

    best_val_precision_pos = -math.inf
    best_val_balanced_accuracy = -math.inf
    best_val_f1_macro = -math.inf
    best_threshold = 0.5
    best_val_metrics_for_objective: dict[str, float] | None = None
    history = []
    best_objective_score = -math.inf
    best_temperature = 1.0
    start_epoch = 1

    if resume_from is not None:
        if not resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        resume_ckpt = torch.load(resume_from, map_location=device)
        resume_model_kwargs = resume_ckpt.get("model_kwargs", model_kwargs)
        if resume_model_kwargs != model_kwargs:
            LOGGER.warning("Resume checkpoint model kwargs differ; using checkpoint model kwargs: %s", resume_model_kwargs)
            model_kwargs = resume_model_kwargs
            model = _build_model(**model_kwargs).to(device)
        model.load_state_dict(resume_ckpt["model_state_dict"])

        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if scheduler is not None and resume_ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if resume_ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

        history = resume_ckpt.get("history", history)
        best_threshold = float(resume_ckpt.get("best_threshold", best_threshold))
        best_temperature = float(resume_ckpt.get("best_temperature", best_temperature))
        best_objective_score = float(resume_ckpt.get("best_objective_score", best_objective_score))
        best_val_precision_pos = float(resume_ckpt.get("best_val_precision_pos", best_val_precision_pos))
        best_val_balanced_accuracy = float(resume_ckpt.get("best_val_balanced_accuracy", best_val_balanced_accuracy))
        best_val_f1_macro = float(resume_ckpt.get("best_val_f1_macro", best_val_f1_macro))
        restored_best_metrics = resume_ckpt.get("best_val_metrics_for_objective")
        if isinstance(restored_best_metrics, dict):
            best_val_metrics_for_objective = restored_best_metrics
        if best_objective_score > -math.inf:
            early_stopping.best_score = best_objective_score
        early_stopping.bad_epochs = int(resume_ckpt.get("early_stopping_bad_epochs", 0))

        completed_epoch = int(resume_ckpt.get("epoch", 0))
        start_epoch = completed_epoch + 1
        if start_epoch > epochs:
            LOGGER.warning(
                "Resume checkpoint already at epoch=%d while requested epochs=%d. "
                "No additional epochs will run.",
                completed_epoch,
                epochs,
            )
        LOGGER.info("Resuming training from checkpoint=%s | completed_epoch=%d | next_epoch=%d", resume_from, completed_epoch, start_epoch)

    for epoch in range(start_epoch, epochs + 1):
        if freeze_feature_epochs > 0:
            trainable = epoch > freeze_feature_epochs
            toggled = _set_frontend_trainable(model, trainable=trainable)
            if epoch == 1 or epoch == freeze_feature_epochs + 1:
                LOGGER.info(
                    "Frontend fine-tune stage | epoch=%d | trainable=%s | parameters_toggled=%d",
                    epoch,
                    trainable,
                    toggled,
                )
        train_loss, train_metrics = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp_enabled=amp_enabled,
            scaler=scaler,
            threshold=0.5,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        val_loss, _, val_labels, val_probabilities = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.5,
            feature_mean=feature_mean,
            feature_std=feature_std,
            amp_enabled=amp_enabled,
        )
        epoch_temperature = 1.0
        calibrated_val_probabilities = val_probabilities
        calibration_nll = _log_loss(val_labels, val_probabilities)
        if enable_prior_shift_calibration:
            epoch_temperature, calibration_nll = _find_best_temperature(
                val_labels,
                val_probabilities,
                temperatures=temp_grid,
                source_pos_prior=val_pos_prior,
                target_pos_prior=resolved_target_pos_prior,
            )
            calibrated_val_probabilities = _calibrate_probabilities(
                val_probabilities,
                temperature=epoch_temperature,
                source_pos_prior=val_pos_prior,
                target_pos_prior=resolved_target_pos_prior,
            )
        epoch_threshold, val_metrics = _find_best_threshold(
            val_labels,
            calibrated_val_probabilities,
            objective=threshold_objective,
            min_recall_pos=min_recall_pos,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_precision_pos": train_metrics["precision_pos"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_recall_pos": train_metrics["recall_pos"],
                "train_recall_neg": train_metrics["recall_neg"],
                "train_f1_macro": train_metrics["f1_macro"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision_pos": val_metrics["precision_pos"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_recall_pos": val_metrics["recall_pos"],
                "val_recall_neg": val_metrics["recall_neg"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_threshold": epoch_threshold,
                "val_temperature": epoch_temperature,
                "val_calibration_nll": calibration_nll,
            }
        )

        if epoch % max(1, log_every) == 0 or epoch == 1 or epoch == epochs:
            LOGGER.info(
                "Epoch %02d/%02d | "
                "train_loss=%.4f train_acc=%.4f train_prec1=%.4f train_bal_acc=%.4f train_rec1=%.4f train_rec0=%.4f | "
                "val_loss=%.4f val_acc=%.4f val_prec1=%.4f val_bal_acc=%.4f val_rec1=%.4f val_rec0=%.4f val_f1_macro=%.4f | "
                "th=%.2f lr=%.6f objective=%s",
                epoch,
                epochs,
                train_loss,
                train_metrics["accuracy"],
                train_metrics["precision_pos"],
                train_metrics["balanced_accuracy"],
                train_metrics["recall_pos"],
                train_metrics["recall_neg"],
                val_loss,
                val_metrics["accuracy"],
                val_metrics["precision_pos"],
                val_metrics["balanced_accuracy"],
                val_metrics["recall_pos"],
                val_metrics["recall_neg"],
                val_metrics["f1_macro"],
                epoch_threshold,
                optimizer.param_groups[0]["lr"],
                threshold_objective,
        )
            LOGGER.info(
                "Calibration | val_temperature=%.3f val_calibration_nll=%.4f source_prior=%.4f target_prior=%.4f",
                epoch_temperature,
                calibration_nll,
                val_pos_prior,
                resolved_target_pos_prior,
            )

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        objective_score = _objective_score(
            val_metrics,
            objective=threshold_objective,
            min_recall_pos=min_recall_pos,
        )

        feature_mean_list = mean_cpu.numpy().tolist() if normalize_features else None
        feature_std_list = std_cpu.numpy().tolist() if normalize_features else None

        should_save_checkpoint = (
            best_val_metrics_for_objective is None
            or _is_better_for_objective(
                val_metrics,
                best_val_metrics_for_objective,
                objective=threshold_objective,
                min_recall_pos=min_recall_pos,
            )
        )
        if should_save_checkpoint:
            best_val_precision_pos = val_metrics["precision_pos"]
            best_val_balanced_accuracy = val_metrics["balanced_accuracy"]
            best_val_f1_macro = val_metrics["f1_macro"]
            best_threshold = epoch_threshold
            best_temperature = epoch_temperature
            best_val_metrics_for_objective = val_metrics.copy()
            best_objective_score = objective_score
            best_payload = _build_checkpoint_payload(
                model=model,
                model_kwargs=model_kwargs,
                model_name=model_name,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                early_stopping=early_stopping,
                best_threshold=best_threshold,
                threshold_objective=threshold_objective,
                min_recall_pos=min_recall_pos,
                loss_name=loss_name,
                criterion_config=criterion_config,
                sampler_strategy=sampler_strategy,
                normalize_features=normalize_features,
                feature_mean_list=feature_mean_list,
                feature_std_list=feature_std_list,
                best_objective_score=best_objective_score,
                best_val_precision_pos=best_val_precision_pos,
                best_val_balanced_accuracy=best_val_balanced_accuracy,
                best_val_f1_macro=best_val_f1_macro,
                best_val_metrics_for_objective=best_val_metrics_for_objective,
                history=history,
                device=device,
                amp_enabled=amp_enabled,
                cpu_threads=resolved_threads,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                scheduler_name=scheduler_name,
                freeze_feature_epochs=freeze_feature_epochs,
            )
            best_payload["best_temperature"] = float(best_temperature)
            best_payload["prior_shift_calibration"] = {
                "enabled": bool(enable_prior_shift_calibration),
                "source_pos_prior": float(val_pos_prior),
                "target_pos_prior": float(resolved_target_pos_prior),
                "test_pos_prior_observed": float(test_pos_prior),
                "temperature_grid": temp_grid,
            }
            torch.save(best_payload, output_path)

        last_payload = _build_checkpoint_payload(
            model=model,
            model_kwargs=model_kwargs,
            model_name=model_name,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            best_threshold=best_threshold,
            threshold_objective=threshold_objective,
            min_recall_pos=min_recall_pos,
            loss_name=loss_name,
            criterion_config=criterion_config,
            sampler_strategy=sampler_strategy,
            normalize_features=normalize_features,
            feature_mean_list=feature_mean_list,
            feature_std_list=feature_std_list,
            best_objective_score=best_objective_score,
            best_val_precision_pos=best_val_precision_pos,
            best_val_balanced_accuracy=best_val_balanced_accuracy,
            best_val_f1_macro=best_val_f1_macro,
            best_val_metrics_for_objective=best_val_metrics_for_objective,
            history=history,
            device=device,
            amp_enabled=amp_enabled,
            cpu_threads=resolved_threads,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            freeze_feature_epochs=freeze_feature_epochs,
        )
        last_payload["best_temperature"] = float(best_temperature)
        last_payload["prior_shift_calibration"] = {
            "enabled": bool(enable_prior_shift_calibration),
            "source_pos_prior": float(val_pos_prior),
            "target_pos_prior": float(resolved_target_pos_prior),
            "test_pos_prior_observed": float(test_pos_prior),
            "temperature_grid": temp_grid,
        }
        torch.save(last_payload, last_checkpoint_path)

        if early_stopping.step(objective_score, epoch=epoch, min_epochs=min_epochs):
            LOGGER.info("Early stopping triggered at epoch %d", epoch)
            break

    if not output_path.exists():
        if last_checkpoint_path.exists():
            LOGGER.warning("Best checkpoint not found; falling back to last checkpoint: %s", last_checkpoint_path)
            torch.save(torch.load(last_checkpoint_path, map_location=device), output_path)
        elif resume_from is not None and resume_from.exists():
            LOGGER.warning("Best checkpoint not found; falling back to resume checkpoint: %s", resume_from)
            torch.save(torch.load(resume_from, map_location=device), output_path)
        else:
            raise FileNotFoundError(f"No checkpoint available to evaluate. Expected: {output_path}")

    checkpoint = torch.load(output_path, map_location=device)
    checkpoint_model_kwargs = checkpoint.get("model_kwargs", model_kwargs)
    if checkpoint_model_kwargs != model_kwargs:
        LOGGER.info("Reloading model with checkpoint architecture: %s", checkpoint_model_kwargs)
        model = _build_model(**checkpoint_model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = float(checkpoint.get("best_threshold", best_threshold))
    best_temperature = float(checkpoint.get("best_temperature", best_temperature))
    calibration_cfg = checkpoint.get("prior_shift_calibration", {})
    calibration_enabled = bool(calibration_cfg.get("enabled", enable_prior_shift_calibration))
    calibration_source_prior = calibration_cfg.get("source_pos_prior", val_pos_prior)
    calibration_target_prior = calibration_cfg.get("target_pos_prior", resolved_target_pos_prior)

    test_loss, test_metrics, _, _ = _evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=best_threshold,
        feature_mean=feature_mean,
        feature_std=feature_std,
        amp_enabled=amp_enabled,
    )
    if calibration_enabled:
        _, _, test_labels_raw, test_probabilities_raw = _evaluate(
            model,
            test_loader,
            criterion,
            device,
            threshold=0.5,
            feature_mean=feature_mean,
            feature_std=feature_std,
            amp_enabled=amp_enabled,
        )
        calibrated_test_probabilities = _calibrate_probabilities(
            test_probabilities_raw,
            temperature=best_temperature,
            source_pos_prior=float(calibration_source_prior),
            target_pos_prior=float(calibration_target_prior),
        )
        calibrated_test_metrics = _compute_binary_metrics(
            test_labels_raw,
            calibrated_test_probabilities,
            threshold=best_threshold,
        )
        test_metrics = calibrated_test_metrics
    LOGGER.info(
        "Test metrics | loss=%.4f acc=%.4f prec1=%.4f bal_acc=%.4f rec1=%.4f rec0=%.4f f1_macro=%.4f threshold=%.2f",
        test_loss,
        test_metrics["accuracy"],
        test_metrics["precision_pos"],
        test_metrics["balanced_accuracy"],
        test_metrics["recall_pos"],
        test_metrics["recall_neg"],
        test_metrics["f1_macro"],
        best_threshold,
    )

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "best_val_precision_pos": best_val_precision_pos,
                "best_val_balanced_accuracy": best_val_balanced_accuracy,
                "best_val_f1_macro": best_val_f1_macro,
                "best_threshold": best_threshold,
                "best_temperature": best_temperature,
                "threshold_objective": threshold_objective,
                "min_recall_pos": min_recall_pos,
                "loss_name": loss_name,
                "criterion_config": criterion_config,
                "sampler_strategy": sampler_strategy,
                "normalize_features": normalize_features,
                "device": str(device),
                "amp_enabled": bool(amp_enabled and device.type == "cuda"),
                "cpu_threads": int(resolved_threads),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "scheduler_name": scheduler_name,
                "freeze_feature_epochs": int(freeze_feature_epochs),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "patience": int(patience),
                "min_epochs": int(min_epochs),
                "last_checkpoint_path": str(last_checkpoint_path),
                "resume_from": str(resume_from) if resume_from is not None else None,
                "best_objective_score": best_objective_score,
                "test_loss": test_loss,
                "test_accuracy": test_metrics["accuracy"],
                "test_precision_pos": test_metrics["precision_pos"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_recall_pos": test_metrics["recall_pos"],
                "test_recall_neg": test_metrics["recall_neg"],
                "test_f1_macro": test_metrics["f1_macro"],
                "prior_shift_calibration": {
                    "enabled": bool(calibration_enabled),
                    "source_pos_prior": float(calibration_source_prior),
                    "target_pos_prior": float(calibration_target_prior),
                    "train_pos_prior": float(train_pos_prior),
                    "validation_pos_prior": float(val_pos_prior),
                    "test_pos_prior_observed": float(test_pos_prior),
                },
                "history": history,
            },
            indent=2,
        )
    )
    LOGGER.info("Training complete | checkpoint=%s | summary=%s", output_path, summary_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train engagement sequence models on GPU/CPU, then export for CPU inference.")
    parser.add_argument("--manifest", type=Path, default=FEATURE_MANIFEST_CSV, help="Feature manifest CSV")
    parser.add_argument("--output", type=Path, default=MODEL_CHECKPOINT_PATH, help="Model checkpoint path")
    parser.add_argument("--sample", action="store_true", help="Run a tiny 2-epoch test training loop")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from checkpoint (.last.pt or .pt)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--log-every", type=int, default=1, help="Log progress every N epochs")
    parser.add_argument("--run-id", type=str, default=None, help="Run id used to resolve run-scoped manifests")
    parser.add_argument(
        "--threshold-objective",
        type=str,
        default="balanced_accuracy",
        choices=["f1_pos", "focused_recall", "balanced_accuracy", "accuracy", "f1_macro"],
        help="Validation criterion for selecting decision threshold",
    )
    parser.add_argument(
        "--min-recall-pos",
        type=float,
        default=0.75,
        help="Minimum target recall for class=1 when threshold objective is focused_recall",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="bce_weighted",
        choices=["bce_weighted", "focal"],
        help="Training loss strategy",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Positive-class alpha for focal loss; defaults to auto class-ratio if omitted",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma for focal loss",
    )
    parser.add_argument(
        "--train-sampler",
        type=str,
        default="weighted",
        choices=["weighted", "shuffle"],
        help="Sampling strategy for training loader",
    )
    parser.add_argument(
        "--no-normalize-features",
        action="store_true",
        help="Disable train-split feature normalization",
    )
    parser.add_argument(
        "--disable-prior-shift-calibration",
        action="store_true",
        help="Disable prior-shift + temperature probability calibration for thresholding/eval",
    )
    parser.add_argument(
        "--target-pos-prior",
        type=float,
        default=None,
        help="Target positive prior for calibration; defaults to train split prior",
    )
    parser.add_argument(
        "--calibration-temperature-grid",
        type=str,
        default="0.75,0.9,1.0,1.1,1.25,1.5",
        help="Comma-separated temperature grid searched on validation split",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gru",
        choices=[
            "gru",
            "gru_basic",
            "simple_gru",
            "bilstm",
            "bi_lstm",
            "copur_bilstm",
            "tcn",
            "1dcnn",
            "temporal_cnn",
            "stgcn",
            "st-gcn",
            "graph_tcn",
            "hybrid",
            "hybrid_attn",
            "tcn_gru_attn",
            "multiscale_gru_attn",
            "transformer",
            "tiny_transformer",
        ],
        help="Sequence model architecture",
    )
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE, help="Sequence model hidden dimension")
    parser.add_argument("--num-layers", type=int, default=MODEL_NUM_LAYERS, help="GRU/Transformer layer count")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout probability")
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Attention heads used by transformer model",
    )
    parser.add_argument(
        "--tcn-kernel-size",
        type=int,
        default=3,
        help="Kernel size used by TCN model",
    )
    parser.add_argument(
        "--tcn-blocks",
        type=int,
        default=3,
        help="Number of dilated residual blocks used by TCN model",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training and validation batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early-stopping patience")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "none"],
        help="Learning-rate scheduler",
    )
    parser.add_argument(
        "--freeze-feature-epochs",
        type=int,
        default=0,
        help="Freeze feature/frontend projection layers for the first N epochs, then fine-tune end-to-end",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="Minimum epochs before early stopping can trigger",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help="Maximum CPU compute threads for PyTorch (recommended: 1-2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto | cpu | cuda | cuda:0 ...",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision when using CUDA",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Force disable mixed precision",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging()
    amp_enabled = (not args.no_amp) and (args.amp or args.device == "auto" or args.device.startswith("cuda"))
    checkpoint_path = train(
        args.manifest,
        args.output,
        sample=args.sample,
        seed=args.seed,
        log_every=args.log_every,
        run_id=args.run_id,
        threshold_objective=args.threshold_objective,
        min_recall_pos=args.min_recall_pos,
        loss_name=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sampler_strategy=args.train_sampler,
        normalize_features=not args.no_normalize_features,
        model_name=args.model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=args.num_heads,
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_blocks=args.tcn_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        cpu_threads=args.cpu_threads,
        device_arg=args.device,
        amp_enabled=amp_enabled,
        resume_from=args.resume_from,
        enable_prior_shift_calibration=not args.disable_prior_shift_calibration,
        target_pos_prior=args.target_pos_prior,
        calibration_temperature_grid=args.calibration_temperature_grid,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        freeze_feature_epochs=args.freeze_feature_epochs,
    )
    LOGGER.info("Saved checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
