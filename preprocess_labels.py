import argparse
from pathlib import Path

import pandas as pd

from config import (
    PROCESSED_LABELS_CSV,
    RAW_LABELS_CSV,
    RAW_TEST_LABELS_CSV,
    RAW_TRAIN_LABELS_CSV,
    RAW_VALIDATION_LABELS_CSV,
)


def _normalise(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {_normalise(column): column for column in df.columns}
    for candidate in candidates:
        key = _normalise(candidate)
        if key in lookup:
            return lookup[key]
    return None


def _build_clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    engagement_column = _find_column(
        df,
        [
            "Engagement",
            "engagement",
            "engagement_level",
            "engagementscore",
            "engagement_score",
            "engagementlabel",
        ],
    )
    if engagement_column is None:
        raise ValueError("Could not find an Engagement column in the labels CSV.")

    boredom_column = _find_column(df, ["Boredom", "boredom"])
    confusion_column = _find_column(df, ["Confusion", "confusion"])
    frustration_column = _find_column(df, ["Frustration", "frustration"])
    if boredom_column is None or confusion_column is None or frustration_column is None:
        raise ValueError(
            "Could not find all required columns: Boredom, Confusion, Frustration. "
            f"Detected columns: {list(df.columns)}"
        )

    id_column = _find_column(
        df,
        ["ClipID", "clip_id", "video_id", "video", "video_path", "path", "filename", "file_name"],
    )

    clean = pd.DataFrame()
    if id_column is not None:
        clean["video_id"] = df[id_column].astype(str)
    else:
        clean["video_id"] = df.index.astype(str)

    if "video_path" in df.columns:
        clean["video_path"] = df["video_path"].astype(str)
    elif id_column is not None and _normalise(id_column) in {"video_path", "path", "filename", "file_name"}:
        clean["video_path"] = df[id_column].astype(str)

    engagement_raw = pd.to_numeric(df[engagement_column], errors="coerce")
    boredom_raw = pd.to_numeric(df[boredom_column], errors="coerce")
    confusion_raw = pd.to_numeric(df[confusion_column], errors="coerce")
    frustration_raw = pd.to_numeric(df[frustration_column], errors="coerce")

    valid_mask = (
        engagement_raw.notna()
        & boredom_raw.notna()
        & confusion_raw.notna()
        & frustration_raw.notna()
    )
    clean = clean.loc[valid_mask].copy()

    engagement_raw = engagement_raw.loc[valid_mask].astype(int)
    boredom_raw = boredom_raw.loc[valid_mask].astype(int)
    confusion_raw = confusion_raw.loc[valid_mask].astype(int)
    frustration_raw = frustration_raw.loc[valid_mask].astype(int)

    clean["engagement_raw"] = engagement_raw.to_numpy()
    clean["boredom_raw"] = boredom_raw.to_numpy()
    clean["confusion_raw"] = confusion_raw.to_numpy()
    clean["frustration_raw"] = frustration_raw.to_numpy()

    positive_mask = (
        (engagement_raw >= 2)
        & (boredom_raw < 2)
        & (confusion_raw < 2)
        & (frustration_raw < 2)
    )
    clean["engagement_binary"] = positive_mask.astype(int).to_numpy()
    clean = clean.drop_duplicates(subset=["video_id"]).reset_index(drop=True)
    return clean


def _load_official_split_frames() -> pd.DataFrame:
    split_sources = {
        "train": RAW_TRAIN_LABELS_CSV,
        "validation": RAW_VALIDATION_LABELS_CSV,
        "test": RAW_TEST_LABELS_CSV,
    }

    frames: list[pd.DataFrame] = []
    missing = [str(path) for path in split_sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing one or more official DAiSEE split label files: "
            f"{missing}"
        )

    for split_name, split_path in split_sources.items():
        split_df = pd.read_csv(split_path)
        clean_split_df = _build_clean_frame(split_df)
        clean_split_df["split"] = split_name
        frames.append(clean_split_df)

    return pd.concat(frames, axis=0, ignore_index=True)


def preprocess_labels(input_csv: Path, output_csv: Path, use_official_splits: bool = True) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if use_official_splits:
        clean_df = _load_official_split_frames()
    else:
        if not input_csv.exists():
            parent = input_csv.parent
            available = sorted([path.name for path in parent.glob("*.csv")]) if parent.exists() else []
            raise FileNotFoundError(
                f"Labels CSV not found: {input_csv}. "
                f"Available CSV files in labels folder: {available}"
            )
        raw_df = pd.read_csv(input_csv)
        clean_df = _build_clean_frame(raw_df)
        clean_df["split"] = "all"

    clean_df.to_csv(output_csv, index=False)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter DAiSEE labels to Engagement only.")
    parser.add_argument("--input", type=Path, default=RAW_LABELS_CSV, help="Path to DAiSEE Labels.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_LABELS_CSV,
        help="Where to save the cleaned engagement-only CSV",
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Use only --input instead of combining official Train/Validation/Test labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = preprocess_labels(args.input, args.output, use_official_splits=not args.single_file)
    print(f"Saved cleaned labels to {output_path}")


if __name__ == "__main__":
    main()