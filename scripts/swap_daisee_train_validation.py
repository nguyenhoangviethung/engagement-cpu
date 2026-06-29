#!/usr/bin/env python3
"""Swap half of DAiSEE validation videos with label-matched train videos."""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def video_index(root: Path) -> dict[str, Path]:
    paths = list(root.rglob("*.avi"))
    result = {path.name: path for path in paths}
    if len(result) != len(paths):
        raise RuntimeError(f"Duplicate clip names found below {root}")
    return result


def proportional_counts(counts: Counter[str], total: int) -> dict[str, int]:
    population = sum(counts.values())
    exact = {key: total * value / population for key, value in counts.items()}
    result = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(result.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def resolve_daisee_root() -> Path:
    env_candidates = [
        os.getenv("ENGAGEMENT_DAISEE_ROOT"),
        os.getenv("DAISEE_ROOT"),
        os.getenv("RAW_DAISEE_ROOT"),
        os.getenv("RAW_DATA_DIR"),
    ]
    candidates = [Path(value).expanduser() for value in env_candidates if value]
    candidates.extend(
        [
            Path("data/raw/daisee/DAiSEE"),
            Path("data/raw/DAiSEE"),
            Path.home() / "engagement-cpu/data/raw/daisee/DAiSEE",
            Path.home() / "engagement-cpu/data/raw/DAiSEE",
            Path("/mnt/data/raw/daisee/DAiSEE"),
            Path("/mnt/data/raw/DAiSEE"),
        ]
    )
    for candidate in candidates:
        if candidate.name == "DataSet":
            candidate = candidate.parent
        if (candidate / "DataSet").is_dir() and (candidate / "Labels").is_dir():
            return candidate
    return Path("data/raw/daisee/DAiSEE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="DAiSEE directory containing DataSet and Labels",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = (args.root or resolve_daisee_root()).resolve()
    dataset = root / "DataSet"
    labels_dir = root / "Labels"
    train_root = dataset / "Train"
    val_root = dataset / "Validation"
    audit_path = labels_dir / f"TrainValidationSwap_seed{args.seed}.csv"
    if audit_path.exists():
        raise RuntimeError(
            f"Swap audit already exists; refusing to apply the same operation again: {audit_path}"
        )

    with (labels_dir / "AllLabels.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError("AllLabels.csv has no header")
        rows = list(reader)
    rows_by_id = {row["ClipID"]: row for row in rows}

    train = video_index(train_root)
    validation = video_index(val_root)
    swap_count = len(validation) // 2
    if swap_count == 0:
        raise RuntimeError("Validation contains no swappable videos")

    val_groups: dict[str, list[str]] = defaultdict(list)
    train_groups: dict[str, list[str]] = defaultdict(list)
    for clip_id in validation:
        if clip_id in rows_by_id:
            val_groups[rows_by_id[clip_id]["Engagement"]].append(clip_id)
    for clip_id in train:
        if clip_id in rows_by_id:
            train_groups[rows_by_id[clip_id]["Engagement"]].append(clip_id)

    allocation = proportional_counts(
        Counter({key: len(value) for key, value in val_groups.items()}), swap_count
    )
    rng = random.Random(args.seed)
    val_selected: list[str] = []
    train_selected: list[str] = []
    for label in sorted(allocation):
        needed = allocation[label]
        if len(train_groups[label]) < needed:
            raise RuntimeError(f"Train label {label} has fewer than {needed} videos")
        val_selected.extend(rng.sample(sorted(val_groups[label]), needed))
        train_selected.extend(rng.sample(sorted(train_groups[label]), needed))

    for clip_id in val_selected:
        source = validation[clip_id]
        target = train_root / source.relative_to(val_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source.parent), str(target.parent))
    for clip_id in train_selected:
        source = train[clip_id]
        target = val_root / source.relative_to(train_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source.parent), str(target.parent))

    # Remove participant directories left empty after moving their clip directories.
    for split_root in (train_root, val_root):
        for directory in sorted(split_root.iterdir()):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()

    train_after = video_index(train_root)
    val_after = video_index(val_root)
    if len(train_after) != len(train) or len(val_after) != len(validation):
        raise RuntimeError("Split sizes changed unexpectedly during swap")
    if set(train_after) & set(val_after):
        raise RuntimeError("A clip appears in both train and validation")

    train_ids = sorted(train_after)
    val_ids = sorted(val_after)
    write_lines(dataset / "Train.txt", train_ids)
    write_lines(dataset / "Validation.txt", val_ids)

    for output_name, ids in (
        ("TrainLabels.csv", set(train_ids)),
        ("ValidationLabels.csv", set(val_ids)),
    ):
        with (labels_dir / output_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row for row in rows if row["ClipID"] in ids)

    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ClipID", "Engagement", "From", "To"])
        for clip_id in sorted(val_selected):
            writer.writerow([clip_id, rows_by_id[clip_id]["Engagement"], "Validation", "Train"])
        for clip_id in sorted(train_selected):
            writer.writerow([clip_id, rows_by_id[clip_id]["Engagement"], "Train", "Validation"])

    print(f"Swapped {swap_count} videos in each direction (seed={args.seed}).")
    print(f"Engagement allocation: {dict(sorted(allocation.items()))}")
    print(f"Train: {len(train_after)} videos; Validation: {len(val_after)} videos")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
