import argparse
import logging
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from engagement_daisee.common.config import (
    PROCESSED_LABELS_CSV,
    PROCESSED_DIR,
    RAW_VIDEO_DIR,
    SAMPLE_VIDEO_COUNT,
    VIDEO_EXTENSIONS,
)


LOGGER = logging.getLogger("cnn_extract_frames")
DEFAULT_CNN_FRAMES_DIR = PROCESSED_DIR / "cnn_frames"
DEFAULT_CNN_MANIFEST_CSV = PROCESSED_DIR / "cnn_frame_manifest.csv"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _collect_rows(labels_csv: Path) -> pd.DataFrame:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    df = pd.read_csv(labels_csv)
    required = {"video_id", "engagement_binary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Labels file missing columns: {sorted(missing)}")
    return df


def _choose_rows(labels_df: pd.DataFrame, sample: bool, seed: int) -> pd.DataFrame:
    unique_videos = labels_df.drop_duplicates(subset=["video_id"]).copy()
    if not sample or len(unique_videos) <= SAMPLE_VIDEO_COUNT:
        return unique_videos.reset_index(drop=True)

    rng = random.Random(seed)
    if "split" not in unique_videos.columns:
        return unique_videos.sample(n=SAMPLE_VIDEO_COUNT, random_state=seed).reset_index(drop=True)

    sampled_frames = []
    split_names = list(unique_videos["split"].astype(str).str.lower().unique())
    per_split = max(1, SAMPLE_VIDEO_COUNT // max(1, len(split_names)))

    for split_name, split_df in unique_videos.groupby("split", sort=False):
        take_n = min(len(split_df), per_split)
        sampled_frames.append(split_df.sample(n=take_n, random_state=seed))

    sampled_df = pd.concat(sampled_frames, axis=0, ignore_index=True).drop_duplicates(subset=["video_id"])
    remaining = SAMPLE_VIDEO_COUNT - len(sampled_df)
    if remaining > 0:
        remaining_pool = unique_videos[~unique_videos["video_id"].isin(sampled_df["video_id"])].copy()
        if len(remaining_pool) > 0:
            sampled_df = pd.concat(
                [sampled_df, remaining_pool.sample(n=min(remaining, len(remaining_pool)), random_state=seed)],
                axis=0,
                ignore_index=True,
            )

    sampled_rows = sampled_df.to_dict(orient="records")
    rng.shuffle(sampled_rows)
    return pd.DataFrame(sampled_rows)


def _build_video_index(raw_video_dir: Path) -> dict[str, Path]:
    video_index: dict[str, Path] = {}
    for extension in VIDEO_EXTENSIONS:
        for video_path in raw_video_dir.rglob(f"*{extension}"):
            stem = video_path.stem
            if stem not in video_index:
                video_index[stem] = video_path
    return video_index


def _resolve_video_path(video_id: str, raw_video_dir: Path, video_index: dict[str, Path]) -> Path | None:
    candidate = Path(str(video_id))
    if candidate.is_file():
        return candidate

    candidate_path = raw_video_dir / candidate
    if candidate.suffix and candidate_path.is_file():
        return candidate_path

    if candidate.stem in video_index:
        return video_index[candidate.stem]
    if candidate.name in video_index:
        return video_index[candidate.name]

    return None


def _uniform_indices(total_frames: int, frames_per_video: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= frames_per_video:
        return list(range(total_frames))
    raw = np.linspace(0, total_frames - 1, num=frames_per_video, dtype=np.int64)
    return sorted(set(int(i) for i in raw))


def _extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    video_stem: str,
    frame_size: int,
    frames_per_video: int,
) -> tuple[list[dict], int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], 0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _uniform_indices(total_frames, frames_per_video)
    if not indices:
        capture.release()
        return [], total_frames

    selected_set = set(indices)
    frame_rows: list[dict] = []
    current_index = 0
    saved_idx = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if current_index in selected_set:
            resized = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
            frame_name = f"{video_stem}_f{current_index:05d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), resized)
            frame_rows.append(
                {
                    "frame_path": str(frame_path),
                    "frame_index": int(current_index),
                    "frame_order": int(saved_idx),
                    "num_frames": int(total_frames),
                }
            )
            saved_idx += 1
            if saved_idx >= len(indices):
                break
        current_index += 1

    capture.release()
    return frame_rows, total_frames


def extract_cnn_frames(
    labels_csv: Path,
    raw_video_dir: Path,
    output_frames_dir: Path,
    manifest_csv: Path,
    frame_size: int = 112,
    frames_per_video: int = 8,
    sample: bool = False,
    seed: int = 42,
) -> Path:
    start = time.time()
    labels_df = _collect_rows(labels_csv)
    labels_df = _choose_rows(labels_df, sample=sample, seed=seed)
    video_index = _build_video_index(raw_video_dir)

    output_frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Starting CNN frame extraction | rows=%d | indexed_videos=%d | frame_size=%d | frames_per_video=%d",
        len(labels_df),
        len(video_index),
        frame_size,
        frames_per_video,
    )

    manifest_rows: list[dict] = []
    unresolved = 0
    unreadable = 0
    processed_videos = 0

    for row in labels_df.to_dict(orient="records"):
        video_id = str(row["video_id"])
        split = str(row.get("split", "unknown")).strip().lower()
        label = int(row["engagement_binary"])

        video_path = _resolve_video_path(video_id, raw_video_dir, video_index)
        if video_path is None:
            unresolved += 1
            continue

        split_dir = output_frames_dir / split / Path(video_id).stem
        split_dir.mkdir(parents=True, exist_ok=True)
        extracted, total_frames = _extract_frames_from_video(
            video_path=video_path,
            output_dir=split_dir,
            video_stem=Path(video_id).stem,
            frame_size=frame_size,
            frames_per_video=frames_per_video,
        )
        if not extracted:
            unreadable += 1
            continue

        for item in extracted:
            manifest_rows.append(
                {
                    "frame_path": item["frame_path"],
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "split": split,
                    "label": label,
                    "frame_index": item["frame_index"],
                    "frame_order": item["frame_order"],
                    "num_frames": int(total_frames),
                }
            )
        processed_videos += 1

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_csv, index=False)
    LOGGER.info(
        "Done | videos=%d | frames=%d | unresolved=%d | unreadable=%d | manifest=%s | elapsed=%.1fs",
        processed_videos,
        len(manifest_df),
        unresolved,
        unreadable,
        manifest_csv,
        time.time() - start,
    )
    return manifest_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract sampled frames for CNN training from DAiSEE videos.")
    parser.add_argument("--labels", type=Path, default=PROCESSED_LABELS_CSV, help="Processed labels CSV")
    parser.add_argument("--videos", type=Path, default=RAW_VIDEO_DIR, help="Root directory containing raw videos")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CNN_FRAMES_DIR, help="Directory to store extracted frames")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CNN_MANIFEST_CSV, help="Output frame manifest CSV")
    parser.add_argument("--frame-size", type=int, default=112, help="Square frame size")
    parser.add_argument("--frames-per-video", type=int, default=8, help="Uniform frames sampled from each video")
    parser.add_argument("--sample", action="store_true", help="Use only a small random subset of videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    extract_cnn_frames(
        labels_csv=args.labels,
        raw_video_dir=args.videos,
        output_frames_dir=args.output_dir,
        manifest_csv=args.manifest,
        frame_size=args.frame_size,
        frames_per_video=args.frames_per_video,
        sample=args.sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

