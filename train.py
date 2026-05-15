import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from config import (
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
from dataset import FeatureSequenceDataset
from model import EngagementGRU


LOGGER = logging.getLogger("train")
MODEL_NUM_LAYERS = 2
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

    def step(self, current_score: float) -> bool:
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def _find_best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_metrics = _compute_binary_metrics(labels, probabilities, threshold=0.5)

    for threshold in thresholds:
        metrics = _compute_binary_metrics(labels, probabilities, threshold=float(threshold))
        
        # Tiêu chí 1: Tìm Threshold cho Balanced Accuracy cao nhất (Ép cân bằng 2 Recall)
        if metrics["balanced_accuracy"] > best_metrics["balanced_accuracy"] + 1e-6:
            best_threshold = float(threshold)
            best_metrics = metrics
        # Tiêu chí 2: Nếu Balanced Accuracy ngang nhau, chọn Threshold nào cho Recall 1 (Tập trung) cao hơn
        elif abs(metrics["balanced_accuracy"] - best_metrics["balanced_accuracy"]) <= 1e-6:
            if metrics["recall_pos"] > best_metrics["recall_pos"]:
                best_threshold = float(threshold)
                best_metrics = metrics

    return best_threshold, best_metrics

def _make_loader(dataset: FeatureSequenceDataset, indices: list[int], batch_size: int, shuffle: bool) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        pin_memory=False,
        drop_last=False,
    )


def _make_weighted_train_loader(
    dataset: FeatureSequenceDataset,
    manifest_df: pd.DataFrame,
    indices: list[int],
    batch_size: int,
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
        pin_memory=False,
        drop_last=False,
    )


def _build_model(input_size: int, hidden_size: int, dropout: float, num_layers: int = MODEL_NUM_LAYERS) -> EngagementGRU:
    return EngagementGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


def _train_one_epoch(model, loader, criterion, optimizer, device, threshold: float = 0.5):
    model.train()
    running_loss = 0.0
    running_total = 0
    all_probabilities = []
    all_labels = []

    for features, labels in tqdm(loader, desc="Training", unit="batch"):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
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
def _evaluate(model, loader, criterion, device, threshold: float):
    model.eval()
    running_loss = 0.0
    running_total = 0
    all_probabilities = []
    all_labels = []

    for features, labels in tqdm(loader, desc="Evaluating", unit="batch"):
        features = features.to(device)
        labels = labels.to(device)
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


def train(
    manifest_csv: Path,
    output_path: Path,
    sample: bool = False,
    seed: int = RANDOM_SEED,
    log_every: int = 1,
    run_id: str | None = None,
) -> Path:
    _set_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    manifest_csv = _resolve_manifest_path(manifest_csv, run_id)

    LOGGER.info(
        "Starting training | manifest=%s | output=%s | sample=%s | seed=%d",
        manifest_csv,
        output_path,
        sample,
        seed,
    )

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

    batch_size = BATCH_SIZE
    epochs = SAMPLE_EPOCHS if sample else EPOCHS
    patience = 2 if sample else EARLY_STOPPING_PATIENCE

    train_loader = _make_weighted_train_loader(dataset, manifest_df, train_indices, batch_size=batch_size)
    val_loader = _make_loader(dataset, val_indices, batch_size=batch_size, shuffle=False)
    test_loader = _make_loader(dataset, test_indices, batch_size=batch_size, shuffle=False)

    LOGGER.info(
        "Data splits | train=%d | validation=%d | test=%d | batch_size=%d",
        len(train_indices),
        len(val_indices),
        len(test_indices),
        batch_size,
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

    sample_features, _ = dataset[0]
    model_kwargs = {
        "input_size": sample_features.shape[-1],
        "hidden_size": HIDDEN_SIZE,
        "num_layers": MODEL_NUM_LAYERS,
        "dropout": DROPOUT,
    }
    model = _build_model(**model_kwargs).to(DEVICE)
    criterion = FocalLoss(alpha=0.5, gamma=2.0).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )
    early_stopping = EarlyStopping(patience=patience)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_precision_pos = -math.inf
    best_val_balanced_accuracy = -math.inf
    best_val_f1_macro = -math.inf
    best_threshold = 0.5
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = _train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, threshold=0.5)
        val_loss, _, val_labels, val_probabilities = _evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
            threshold=0.5,
        )
        epoch_threshold, val_metrics = _find_best_threshold(val_labels, val_probabilities)

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
            }
        )

        if epoch % max(1, log_every) == 0 or epoch == 1 or epoch == epochs:
            LOGGER.info(
                "Epoch %02d/%02d | "
                "train_loss=%.4f train_acc=%.4f train_prec1=%.4f train_bal_acc=%.4f train_rec1=%.4f train_rec0=%.4f | "
                "val_loss=%.4f val_acc=%.4f val_prec1=%.4f val_bal_acc=%.4f val_rec1=%.4f val_rec0=%.4f val_f1_macro=%.4f | "
                "th=%.2f lr=%.6f",
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
            )

        scheduler.step(val_loss)

        if val_metrics["balanced_accuracy"] > best_val_balanced_accuracy + 1e-8:
            best_val_precision_pos = val_metrics["precision_pos"]
            best_val_balanced_accuracy = val_metrics["balanced_accuracy"]
            best_val_f1_macro = val_metrics["f1_macro"]
            best_threshold = epoch_threshold
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "best_threshold": best_threshold,
                    "best_val_precision_pos": best_val_precision_pos,
                    "best_val_balanced_accuracy": best_val_balanced_accuracy,
                    "best_val_f1_macro": best_val_f1_macro,
                    "history": history,
                },
                output_path,
            )

        if early_stopping.step(val_metrics["balanced_accuracy"]):
            LOGGER.info("Early stopping triggered at epoch %d", epoch)
            break

    checkpoint = torch.load(output_path, map_location=DEVICE)
    checkpoint_model_kwargs = checkpoint.get("model_kwargs", model_kwargs)
    if checkpoint_model_kwargs != model_kwargs:
        LOGGER.info("Reloading model with checkpoint architecture: %s", checkpoint_model_kwargs)
        model = _build_model(**checkpoint_model_kwargs).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = float(checkpoint.get("best_threshold", best_threshold))

    test_loss, test_metrics, _, _ = _evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
        threshold=best_threshold,
    )
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
                "test_loss": test_loss,
                "test_accuracy": test_metrics["accuracy"],
                "test_precision_pos": test_metrics["precision_pos"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_recall_pos": test_metrics["recall_pos"],
                "test_recall_neg": test_metrics["recall_neg"],
                "test_f1_macro": test_metrics["f1_macro"],
                "history": history,
            },
            indent=2,
        )
    )
    LOGGER.info("Training complete | checkpoint=%s | summary=%s", output_path, summary_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CPU-only GRU engagement classifier.")
    parser.add_argument("--manifest", type=Path, default=FEATURE_MANIFEST_CSV, help="Feature manifest CSV")
    parser.add_argument("--output", type=Path, default=MODEL_CHECKPOINT_PATH, help="Model checkpoint path")
    parser.add_argument("--sample", action="store_true", help="Run a tiny 2-epoch test training loop")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--log-every", type=int, default=1, help="Log progress every N epochs")
    parser.add_argument("--run-id", type=str, default=None, help="Run id used to resolve run-scoped manifests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging()
    checkpoint_path = train(
        args.manifest,
        args.output,
        sample=args.sample,
        seed=args.seed,
        log_every=args.log_every,
        run_id=args.run_id,
    )
    LOGGER.info("Saved checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()