from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import kagglehub

from config import RAW_DATA_DIR

DATASET_ID = "olgaparfenova/daisee"


def _check_kaggle_credentials() -> Path:
    credentials_path = Path.home() / ".kaggle" / "kaggle.json"
    if not credentials_path.exists():
        raise FileNotFoundError(
            "Kaggle credentials not found at ~/.kaggle/kaggle.json. "
            "Create that file before running the download script."
        )
    return credentials_path


def _print_tree(root: Path, max_depth: int = 3) -> None:
    root = root.resolve()
    print(f"\nDataset tree for: {root}")
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        depth = len(relative.parts)
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        suffix = "/" if path.is_dir() else ""
        print(f"{indent}{relative.name}{suffix}")


def _copy_dataset(source: Path, destination: Path) -> None:
    if destination.exists():
        print(f"Mirror destination already exists, skipping copy: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"Mirrored dataset to: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the DAiSEE Kaggle dataset with kagglehub.")
    parser.add_argument(
        "--mirror-to",
        type=Path,
        default=RAW_DATA_DIR,
        help="Local folder to mirror the downloaded dataset into (default: data/raw/daisee/DAiSEE)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    credentials_path = _check_kaggle_credentials()
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(credentials_path.parent))
    print(f"Downloading dataset: {DATASET_ID}")
    path = Path(kagglehub.dataset_download(DATASET_ID))
    print(f"Path to dataset files: {path}")
    _print_tree(path)

    if args.mirror_to is not None:
        _copy_dataset(path, args.mirror_to)


if __name__ == "__main__":
    main()
