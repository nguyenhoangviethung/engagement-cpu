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
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from engagement_daisee.cnn.dataset import DAiSEECNNFrameDataset
from engagement_daisee.cnn.model import build_cnn_model
from engagement_daisee.common.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DEVICE as DEFAULT_DEVICE,
    EPOCHS,
    NUM_WORKERS,
    RANDOM_SEED,
    SAMPLE_EPOCHS,
)
from engagement_daisee.common.manifest import normalize_manifest_columns


LOGGER = logging.getLogger("train_cnn")
DEFAULT_MANIFEST = Path("data/processed/cnn_frame_manifest.csv")
DEFAULT_OUTPUT = CHECKPOINT_DIR / "engagement_cnn.pt"
RUNS_PROCESSED_DIR = Path("data/processed/runs")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class EarlyStopping:
    patience: int = 6
    min_delta: float = 1e-4
    best_score: float = field(default=-math.inf)
    bad_epochs: int = field(default=0)

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    rec_pos = _safe_div(tp, tp + fn)
    rec_neg = _safe_div(tn, tn + fp)
    precision_pos = _safe_div(tp, tp + fp)
    f1_pos = _safe_div(2 * precision_pos * rec_pos, precision_pos + rec_pos)
    precision_neg = _safe_div(tn, tn + fn)
    f1_neg = _safe_div(2 * precision_neg * rec_neg, precision_neg + rec_neg)
    bal_acc = (rec_pos + rec_neg) / 2.0
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    f1_macro = (f1_pos + f1_neg) / 2.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_pos": precision_pos,
        "recall_pos": rec_pos,
        "recall_neg": rec_neg,
        "f1_macro": f1_macro,
    }


def _is_better_for_objective(candidate: dict[str, float], incumbent: dict[str, float], objective: str) -> bool:
    candidate_score = float(candidate[objective])
    incumbent_score = float(incumbent[objective])
    if candidate_score > incumbent_score + 1e-8:
        return True
    if abs(candidate_score - incumbent_score) <= 1e-8:
        return candidate["f1_macro"] > incumbent["f1_macro"] + 1e-8
    return False


def _find_best_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    objective: str = "balanced_accuracy",
) -> tuple[float, dict[str, float]]:
    thresholds = np.arange(0.1, 0.901, 0.01)
    best_t = 0.5
    best_m = _compute_metrics(labels, probs, threshold=0.5)
    for t in thresholds:
        m = _compute_metrics(labels, probs, threshold=float(t))
        if _is_better_for_objective(m, best_m, objective):
            best_t = float(t)
            best_m = m
    return best_t, best_m


def _resolve_manifest_path(manifest_csv: Path, run_id: str | None) -> Path:
    if run_id is None and manifest_csv.exists():
        return manifest_csv

    candidates: list[Path] = []
    if run_id:
        candidates.extend(
            [
                RUNS_PROCESSED_DIR / f"train_{run_id}" / "cnn_frame_manifest.csv",
                RUNS_PROCESSED_DIR / f"pipeline_{run_id}" / "cnn_frame_manifest.csv",
                RUNS_PROCESSED_DIR / f"extract_{run_id}" / "cnn_frame_manifest.csv",
            ]
        )
    candidates.append(manifest_csv)
    candidates.append(DEFAULT_MANIFEST)

    for path in candidates:
        if path.exists():
            LOGGER.info("Resolved manifest path: %s", path)
            return path
    raise FileNotFoundError(f"Could not find CNN frame manifest. Tried: {candidates}")


def _split_indices(manifest_df: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    manifest_df = normalize_manifest_columns(manifest_df)
    if "split" not in manifest_df.columns:
        raise ValueError("Manifest must include split column.")
    split_series = manifest_df["split"].astype(str).str.strip().str.lower()
    train_indices = split_series[split_series == "train"].index.tolist()
    val_indices = split_series[split_series == "validation"].index.tolist()
    test_indices = split_series[split_series == "test"].index.tolist()
    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            f"Invalid split sizes | train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}"
        )
    return train_indices, val_indices, test_indices


def _build_transform(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 12, image_size + 12)),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _make_loaders(
    dataset: DAiSEECNNFrameDataset,
    manifest_df: pd.DataFrame,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    batch_size: int,
    sampler_strategy: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    labels = manifest_df.iloc[train_indices]["label"].astype(int).to_numpy()
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError(f"Train split must contain both classes; got {class_counts}.")

    if sampler_strategy == "weighted":
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            persistent_workers=True if NUM_WORKERS > 0 else False,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            persistent_workers=True if NUM_WORKERS > 0 else False,
        )

    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    return train_loader, val_loader, test_loader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    all_probs = []
    all_labels = []
    running_loss = 0.0
    total = 0

    for images, labels in tqdm(loader, desc="Train" if is_train else "Eval", unit="batch"):
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images).view(-1)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        probs = torch.sigmoid(logits)
        bs = labels.size(0)
        running_loss += float(loss.item()) * bs
        total += bs
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = running_loss / max(1, total)
    labels_np = np.concatenate(all_labels).astype(np.int64)
    probs_np = np.concatenate(all_probs).astype(np.float32)
    return avg_loss, labels_np, probs_np


def train_cnn(
    manifest_csv: Path,
    output_path: Path,
    model_name: str = "mobilenet_v3_small",
    image_size: int = 112,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    epochs: int = EPOCHS,
    patience: int = 6,
    sample: bool = False,
    seed: int = RANDOM_SEED,
    run_id: str | None = None,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    sampler_strategy: str = "weighted",
    device: str = DEFAULT_DEVICE,
    threshold_objective: str = "balanced_accuracy",
) -> Path:
    _set_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"

    manifest_csv = _resolve_manifest_path(manifest_csv, run_id)
    manifest_df = normalize_manifest_columns(pd.read_csv(manifest_csv))
    train_indices, val_indices, test_indices = _split_indices(manifest_df)

    if sample:
        rng = random.Random(seed)
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)
        train_indices = train_indices[: max(1, min(len(train_indices), batch_size * 6))]
        val_indices = val_indices[: max(1, min(len(val_indices), batch_size * 3))]
        test_indices = test_indices[: max(1, min(len(test_indices), batch_size * 3))]
        epochs = SAMPLE_EPOCHS
        patience = min(patience, 2)

    LOGGER.info(
        "CNN training | model=%s | manifest=%s | train=%d val=%d test=%d | sample=%s",
        model_name,
        manifest_csv,
        len(train_indices),
        len(val_indices),
        len(test_indices),
        sample,
    )
    LOGGER.info("Using device: %s", device)

    train_dataset = DAiSEECNNFrameDataset(manifest_csv=manifest_csv, transform=_build_transform(image_size, train=True))
    eval_dataset = DAiSEECNNFrameDataset(manifest_csv=manifest_csv, transform=_build_transform(image_size, train=False))

    train_loader, _, _ = _make_loaders(
        train_dataset,
        manifest_df,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=batch_size,
        sampler_strategy=sampler_strategy,
    )
    _, val_loader, test_loader = _make_loaders(
        eval_dataset,
        manifest_df,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=batch_size,
        sampler_strategy=sampler_strategy,
    )

    train_labels = manifest_df.iloc[train_indices]["label"].astype(float).to_numpy()
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    pos_weight = negatives / max(1.0, positives)

    model = build_cnn_model(model_name=model_name, pretrained=pretrained, freeze_backbone=freeze_backbone).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    stopper = EarlyStopping(patience=patience)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_score = -math.inf
    best_threshold = 0.5
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_labels_np, train_probs_np = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        train_metrics = _compute_metrics(train_labels_np, train_probs_np, threshold=0.5)

        val_loss, val_labels_np, val_probs_np = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        val_threshold, val_metrics = _find_best_threshold(
            val_labels_np,
            val_probs_np,
            objective=threshold_objective,
        )
        val_score = val_metrics[threshold_objective]
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_recall_pos": train_metrics["recall_pos"],
                "train_recall_neg": train_metrics["recall_neg"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_recall_pos": val_metrics["recall_pos"],
                "val_recall_neg": val_metrics["recall_neg"],
                "val_threshold": val_threshold,
            }
        )

        LOGGER.info(
            "Epoch %02d/%02d | train_loss=%.4f train_bal_acc=%.4f | val_loss=%.4f val_bal_acc=%.4f val_rec1=%.4f val_rec0=%.4f th=%.2f",
            epoch,
            epochs,
            train_loss,
            train_metrics["balanced_accuracy"],
            val_loss,
            val_metrics["balanced_accuracy"],
            val_metrics["recall_pos"],
            val_metrics["recall_neg"],
            val_threshold,
        )

        if val_score > best_val_score + 1e-8:
            best_val_score = val_score
            best_threshold = val_threshold
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "image_size": image_size,
                    "best_threshold": best_threshold,
                    "best_val_metric": best_val_score,
                    "best_val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "threshold_objective": threshold_objective,
                    "history": history,
                },
                output_path,
            )

        if stopper.step(val_score):
            LOGGER.info("Early stopping at epoch %d", epoch)
            break

    ckpt = torch.load(output_path, map_location=device)
    model = build_cnn_model(model_name=ckpt["model_name"], pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_threshold = float(ckpt.get("best_threshold", best_threshold))

    test_loss, test_labels_np, test_probs_np = _run_epoch(model, test_loader, criterion, device, optimizer=None)
    test_metrics = _compute_metrics(test_labels_np, test_probs_np, threshold=best_threshold)

    LOGGER.info(
        "Test | loss=%.4f acc=%.4f bal_acc=%.4f rec1=%.4f rec0=%.4f th=%.2f",
        test_loss,
        test_metrics["accuracy"],
        test_metrics["balanced_accuracy"],
        test_metrics["recall_pos"],
        test_metrics["recall_neg"],
        best_threshold,
    )

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "image_size": image_size,
                "device": device,
                "best_threshold": best_threshold,
                "best_val_metric": best_val_score,
                "best_val_balanced_accuracy": float(ckpt.get("best_val_balanced_accuracy", best_val_score)),
                "threshold_objective": threshold_objective,
                "test_loss": test_loss,
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_recall_pos": test_metrics["recall_pos"],
                "test_recall_neg": test_metrics["recall_neg"],
                "history": history,
            },
            indent=2,
        )
    )
    LOGGER.info("Saved CNN checkpoint: %s", output_path)
    LOGGER.info("Saved CNN summary: %s", summary_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight CNN for DAiSEE engagement classification.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Frame manifest CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Checkpoint output path")
    parser.add_argument("--model", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "efficientnet_b0", "tinycnn"])
    parser.add_argument("--image-size", type=int, default=112, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights if available")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze CNN backbone and train classifier head only")
    parser.add_argument("--train-sampler", type=str, default="weighted", choices=["weighted", "shuffle"])
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    parser.add_argument(
        "--threshold-objective",
        type=str,
        default="balanced_accuracy",
        choices=["accuracy", "balanced_accuracy", "f1_macro"],
        help="Validation metric used for checkpoint/threshold selection.",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    train_cnn(
        manifest_csv=args.manifest,
        output_path=args.output,
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        sample=args.sample,
        seed=args.seed,
        run_id=args.run_id,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        sampler_strategy=args.train_sampler,
        device=args.device,
        threshold_objective=args.threshold_objective,
    )


if __name__ == "__main__":
    main()
