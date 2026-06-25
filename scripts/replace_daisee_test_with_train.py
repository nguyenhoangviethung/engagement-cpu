#!/usr/bin/env python3
"""Replace half of the DAiSEE test split with label-matched train videos.

This keeps the train split unchanged. The selected train clips are copied into
test, the selected test clips are removed from test, and the latest manifest
snapshot is rewritten so the model pipeline can see the updated test split.
"""

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
    if population <= 0:
        raise RuntimeError("Cannot allocate counts from an empty population")
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
            Path("data/raw/DAISEE"),
            Path.home() / "engagement-cpu/data/raw/daisee/DAiSEE",
            Path.home() / "engagement-cpu/data/raw/DAISEE",
            Path("/mnt/data/raw/daisee/DAiSEE"),
            Path("/mnt/data/raw/DAISEE"),
        ]
    )
    for candidate in candidates:
        if candidate.name == "DataSet":
            candidate = candidate.parent
        if (candidate / "DataSet").is_dir() and (candidate / "Labels").is_dir():
            return candidate
    return Path("data/raw/daisee/DAiSEE")


def load_labels(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError(f"{path} has no header")
        rows = list(reader)
    rows_by_id = {row["ClipID"]: row for row in rows}
    return fieldnames, rows_by_id


def cleanup_empty_parents(root: Path) -> None:
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="DAiSEE directory containing DataSet and Labels",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("checkpoints/runs/train_all_4class_depth_robust_20260620_160616/manifest_snapshot.csv"),
        help="Manifest snapshot to rewrite",
    )
    args = parser.parse_args()

    root = (args.root or resolve_daisee_root()).resolve()
    dataset = root / "DataSet"
    labels_dir = root / "Labels"
    train_root = dataset / "Train"
    test_root = dataset / "Test"
    audit_path = labels_dir / f"TestReplaceFromTrain_seed{args.seed}.csv"
    manifest_path = args.manifest.resolve()
    manifest_backup = manifest_path.with_name(f"{manifest_path.stem}.before_test_replace_seed{args.seed}{manifest_path.suffix}")

    if audit_path.exists():
        raise RuntimeError(
            f"Audit already exists; refusing to apply the same replace operation again: {audit_path}"
        )
    if manifest_backup.exists():
        raise RuntimeError(
            f"Manifest backup already exists; refusing to overwrite it: {manifest_backup}"
        )

    fieldnames, rows_by_id = load_labels(labels_dir / "AllLabels.csv")

    train = video_index(train_root)
    test = video_index(test_root)
    replace_count = len(test) // 2
    if replace_count == 0:
        raise RuntimeError("Test contains no videos to replace")

    labeled_test = [clip_id for clip_id in test if clip_id in rows_by_id]
    if len(labeled_test) < replace_count:
        raise RuntimeError(
            f"Not enough labeled test videos to replace {replace_count}; only {len(labeled_test)} available"
        )

    test_groups: dict[str, list[str]] = defaultdict(list)
    train_groups: dict[str, list[str]] = defaultdict(list)
    for clip_id in labeled_test:
        test_groups[rows_by_id[clip_id]["Engagement"]].append(clip_id)
    for clip_id in train:
        if clip_id in rows_by_id:
            train_groups[rows_by_id[clip_id]["Engagement"]].append(clip_id)

    allocation = proportional_counts(
        Counter({key: len(value) for key, value in test_groups.items()}), replace_count
    )
    rng = random.Random(args.seed)
    test_selected: list[str] = []
    train_selected: list[str] = []
    for label in sorted(allocation):
        needed = allocation[label]
        if len(train_groups[label]) < needed:
            raise RuntimeError(f"Train label {label} has fewer than {needed} videos")
        test_selected.extend(rng.sample(sorted(test_groups[label]), needed))
        train_selected.extend(rng.sample(sorted(train_groups[label]), needed))

    test_selected_set = set(test_selected)
    train_selected_set = set(train_selected)

    for clip_id in test_selected:
        source = test[clip_id]
        shutil.rmtree(source.parent)

    for clip_id in train_selected:
        source = train[clip_id]
        target = test_root / source.relative_to(train_root)
        if target.exists():
            raise RuntimeError(f"Target already exists in test: {target}")
        target.parent.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source.parent, target.parent)

    cleanup_empty_parents(test_root)

    test_after = video_index(test_root)
    if len(test_after) != len(test):
        raise RuntimeError("Test size changed unexpectedly during replace")
    if set(test_after) & (set(train) - train_selected_set):
        # Existing overlap would be a pre-existing issue, but the new selection
        # should not introduce one outside of the intended copied train clips.
        raise RuntimeError("Unexpected train/test overlap detected after replace")

    test_ids = sorted(test_after)
    write_lines(dataset / "Test.txt", test_ids)

    with (labels_dir / "TestLabels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows_by_id.values() if row["ClipID"] in set(test_ids))

    manifest_backup.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        manifest_fields = reader.fieldnames
        if not manifest_fields:
            raise RuntimeError("Manifest has no header")
        manifest_rows = list(reader)

    if "split" not in manifest_fields or "video_id" not in manifest_fields or "video_path" not in manifest_fields:
        raise RuntimeError("Manifest is missing required columns: split, video_id, video_path")

    train_path_lookup = {clip_id: train[clip_id] for clip_id in train_selected_set}
    retained_rows: list[dict[str, str]] = []
    for row in manifest_rows:
        if row["split"] == "test" and row["video_id"] in test_selected_set:
            continue
        retained_rows.append(row)

    for row in manifest_rows:
        if row["split"] != "train" or row["video_id"] not in train_selected_set:
            continue
        copied = dict(row)
        copied["split"] = "test"
        copied["video_path"] = str(test_root / train_path_lookup[row["video_id"]].relative_to(train_root))
        retained_rows.append(copied)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(retained_rows)

    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ClipID", "Engagement", "From", "To"])
        for clip_id in sorted(test_selected):
            writer.writerow([clip_id, rows_by_id[clip_id]["Engagement"], "Test", "Removed"])
        for clip_id in sorted(train_selected):
            writer.writerow([clip_id, rows_by_id[clip_id]["Engagement"], "Train", "Test"])

    print(f"Replaced {replace_count} test videos with train videos (seed={args.seed}).")
    print(f"Engagement allocation: {dict(sorted(allocation.items()))}")
    print(f"Test: {len(test_after)} videos; Train unchanged: {len(train)} videos")
    print(f"Audit: {audit_path}")
    print(f"Manifest backup: {manifest_backup}")


if __name__ == "__main__":
    main()
