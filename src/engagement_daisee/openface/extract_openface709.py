from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engagement_daisee.common.config import (
    BASE_DIR,
    PROCESSED_LABELS_CSV,
    RAW_VIDEO_DIR,
    SEQUENCE_LENGTH,
    VIDEO_EXTENSIONS,
)

LOGGER = logging.getLogger(__name__)
META_COLUMNS = {"frame", "face_id", "timestamp", "confidence", "success"}
EXPECTED_FEATURE_DIM = 709
MANIFEST_COLUMNS = [
    "feature_path",
    "video_id",
    "video_path",
    "label",
    "split",
    "segment_index",
    "num_frames_raw",
    "num_frames_used",
    "sequence_length",
    "max_frames",
    "feature_set",
    "feature_dim",
    "openface_csv_path",
]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_video_index(raw_video_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ext in VIDEO_EXTENSIONS:
        for path in raw_video_dir.rglob(f"*{ext}"):
            index.setdefault(path.name.lower(), path)
            index.setdefault(path.stem.lower(), path)
    return index


def _resolve_video_path(video_id: str, raw_video_dir: Path, video_index: dict[str, Path]) -> Path | None:
    candidate = Path(video_id)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    key = str(video_id).lower()
    if key in video_index:
        return video_index[key]
    stem_key = Path(video_id).stem.lower()
    return video_index.get(stem_key)


def _read_openface_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [str(c).strip() for c in df.columns]
    return [c for c in cols if c not in META_COLUMNS]


def _validate_labels(labels_df: pd.DataFrame) -> None:
    required = {"video_id", "engagement_binary", "split"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"labels CSV is missing required columns: {sorted(missing)}")
    labels = set(pd.to_numeric(labels_df["engagement_binary"], errors="raise").astype(int).unique())
    if not labels <= {0, 1}:
        raise ValueError(f"engagement_binary must be binary 0/1, got: {sorted(labels)}")


def _uniform_sample(features: np.ndarray, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or len(features) <= max_frames:
        return features
    indices = np.linspace(0, len(features) - 1, num=max_frames).round().astype(int)
    return features[indices]


def _window_features(features: np.ndarray, sequence_length: int) -> list[np.ndarray]:
    if features.size == 0:
        return []
    if len(features) < sequence_length:
        pad_count = sequence_length - len(features)
        pad = np.repeat(features[-1:], pad_count, axis=0)
        return [np.concatenate([features, pad], axis=0)]
    windows = []
    for start in range(0, len(features) - sequence_length + 1, sequence_length):
        windows.append(features[start : start + sequence_length])
    if not windows:
        windows.append(features[:sequence_length])
    return windows


def _run_openface(
    binary: Path,
    build_dir: Path,
    video_path: Path,
    csv_dir: Path,
    log_dir: Path,
    force: bool,
) -> tuple[Path | None, int]:
    csv_path = csv_dir / f"{video_path.stem}.csv"
    log_path = log_dir / f"{video_path.stem}.log"
    if csv_path.exists() and not force:
        return csv_path, 0

    for stale in csv_dir.glob(f"{video_path.stem}*"):
        if stale.is_file():
            stale.unlink()
    cmd = [
        str(binary),
        "-f",
        str(video_path),
        "-out_dir",
        str(csv_dir),
        "-2Dfp",
        "-3Dfp",
        "-pdmparams",
        "-pose",
        "-aus",
        "-gaze",
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, cwd=str(build_dir), stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0 or not csv_path.exists():
        return None, proc.returncode
    return csv_path, proc.returncode


def _load_resume_manifest(manifest_path: Path) -> tuple[set[str], list[dict[str, object]]]:
    if not manifest_path.exists():
        return set(), []
    try:
        frame = pd.read_csv(manifest_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read existing manifest for resume: {manifest_path}") from exc
    if frame.empty:
        return set(), []
    if "video_id" not in frame.columns:
        raise ValueError(f"Resume manifest missing video_id column: {manifest_path}")
    processed_ids = set(frame["video_id"].astype(str).tolist())
    rows = frame.to_dict(orient="records")
    return processed_ids, rows


def _append_manifest_rows(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=MANIFEST_COLUMNS)
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    frame.to_csv(manifest_path, mode="a", header=write_header, index=False)


def _write_state(state_path: Path, payload: dict[str, object]) -> None:
    tmp_path = state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(state_path)


def extract_openface709(
    labels_csv: Path,
    raw_video_dir: Path,
    output_root: Path,
    openface_binary: Path,
    sample: bool = False,
    sample_videos: int = 10,
    seed: int = 42,
    sequence_length: int = SEQUENCE_LENGTH,
    max_frames: int = 120,
    min_confidence: float = 0.0,
    force: bool = False,
    log_every: int = 25,
    resume: bool = True,
    save_every: int = 20,
) -> Path:
    start_time = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    csv_dir = output_root / "openface_csv"
    features_dir = output_root / "features"
    logs_dir = output_root / "openface_logs"
    for directory in (csv_dir, features_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "feature_manifest.csv"
    state_path = output_root / "extract_state.json"

    build_dir = openface_binary.resolve().parents[1]
    if not openface_binary.exists():
        raise FileNotFoundError(f"OpenFace FeatureExtraction not found: {openface_binary}")

    labels_df = pd.read_csv(labels_csv)
    _validate_labels(labels_df)
    labels_df = labels_df.drop_duplicates(subset=["video_id"]).reset_index(drop=True)
    rows = labels_df.to_dict(orient="records")
    if sample:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:sample_videos]

    video_index = _build_video_index(raw_video_dir)
    manifest_rows: list[dict[str, object]] = []
    pending_rows: list[dict[str, object]] = []
    processed_video_ids: set[str] = set()
    if resume and not force:
        processed_video_ids, manifest_rows = _load_resume_manifest(manifest_path)
        if processed_video_ids:
            LOGGER.info(
                "Resume detected | processed_videos=%d | existing_manifest_rows=%d | manifest=%s",
                len(processed_video_ids),
                len(manifest_rows),
                manifest_path,
            )
    schema_columns: list[str] | None = None
    unresolved = failed = empty = schema_mismatch = skipped_already_done = 0
    split_counts: dict[str, int] = {}
    label_counts = {0: 0, 1: 0}
    if manifest_rows:
        manifest_df_resume = pd.DataFrame(manifest_rows)
        if "split" in manifest_df_resume.columns:
            split_counts = (
                manifest_df_resume["split"].astype(str).value_counts().to_dict()
            )
        if "label" in manifest_df_resume.columns:
            label_counts_resume = manifest_df_resume["label"].astype(int).value_counts().to_dict()
            label_counts[0] = int(label_counts_resume.get(0, 0))
            label_counts[1] = int(label_counts_resume.get(1, 0))

    LOGGER.info(
        "Starting OpenFace709 extraction | rows=%d | videos_indexed=%d | max_frames=%d | seq_len=%d",
        len(rows),
        len(video_index),
        max_frames,
        sequence_length,
    )

    for idx, row in enumerate(rows, start=1):
        video_id = str(row["video_id"])
        if video_id in processed_video_ids and not force:
            skipped_already_done += 1
            continue
        video_path = _resolve_video_path(video_id, raw_video_dir, video_index)
        if video_path is None:
            unresolved += 1
            LOGGER.warning("Skipping unresolved video_id=%s", video_id)
            continue

        csv_path, status = _run_openface(openface_binary, build_dir, video_path, csv_dir, logs_dir, force=force)
        if csv_path is None:
            failed += 1
            LOGGER.warning("OpenFace failed | status=%s | video=%s", status, video_path)
            continue

        df = _read_openface_csv(csv_path)
        columns = _feature_columns(df)
        if schema_columns is None:
            schema_columns = columns
            if len(schema_columns) != EXPECTED_FEATURE_DIM:
                raise ValueError(
                    f"OpenFace schema check failed: expected {EXPECTED_FEATURE_DIM} feature columns after metadata, "
                    f"got {len(schema_columns)} from {csv_path}"
                )
        elif columns != schema_columns:
            schema_mismatch += 1
            LOGGER.warning("Schema mismatch; skipping video=%s", video_path)
            continue

        if "confidence" in df.columns and min_confidence > 0:
            df = df[pd.to_numeric(df["confidence"], errors="coerce").fillna(0) >= min_confidence]
        if df.empty:
            empty += 1
            LOGGER.warning("OpenFace CSV is empty after filtering | video=%s", video_path)
            continue

        feature_df = df[schema_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        features = feature_df.to_numpy(dtype=np.float32)
        features = _uniform_sample(features, max_frames=max_frames)
        windows = _window_features(features, sequence_length=sequence_length)
        label = int(row["engagement_binary"])
        split = str(row.get("split", "unknown")).lower()
        split_counts[split] = split_counts.get(split, 0) + len(windows)
        label_counts[label] = label_counts.get(label, 0) + len(windows)

        for segment_index, window in enumerate(windows):
            feature_path = features_dir / f"{Path(video_id).stem}_seg{segment_index:04d}.npy"
            np.save(feature_path, window.astype(np.float32))
            new_row = {
                "feature_path": str(feature_path),
                "video_id": video_id,
                "video_path": str(video_path),
                "label": label,
                "split": split,
                "segment_index": segment_index,
                "num_frames_raw": int(len(df)),
                "num_frames_used": int(len(features)),
                "sequence_length": int(sequence_length),
                "max_frames": int(max_frames),
                "feature_set": "openface709",
                "feature_dim": EXPECTED_FEATURE_DIM,
                "openface_csv_path": str(csv_path),
            }
            manifest_rows.append(new_row)
            pending_rows.append(new_row)

        processed_video_ids.add(video_id)
        if len(pending_rows) >= max(1, save_every):
            _append_manifest_rows(manifest_path, pending_rows)
            pending_rows.clear()
            _write_state(
                state_path,
                {
                    "manifest": str(manifest_path),
                    "processed_videos": len(processed_video_ids),
                    "manifest_rows": len(manifest_rows),
                    "last_video_id": video_id,
                    "updated_at": time.time(),
                    "resume_enabled": resume,
                }
            )

        if idx % max(1, log_every) == 0 or idx == len(rows):
            LOGGER.info(
                "Progress %d/%d | manifest_rows=%d | unresolved=%d failed=%d empty=%d schema_mismatch=%d skipped_done=%d",
                idx,
                len(rows),
                len(manifest_rows),
                unresolved,
                failed,
                empty,
                schema_mismatch,
                skipped_already_done,
            )

    if schema_columns is None:
        if (output_root / "openface709_schema.json").exists():
            schema_payload = json.loads((output_root / "openface709_schema.json").read_text(encoding="utf-8"))
            schema_columns = list(schema_payload.get("feature_columns", []))
        if not schema_columns:
            raise RuntimeError("No OpenFace CSV was successfully processed; cannot write schema/manifest.")

    if pending_rows:
        _append_manifest_rows(manifest_path, pending_rows)
        pending_rows.clear()

    schema_path = output_root / "openface709_schema.json"
    schema_payload = {
        "feature_set": "openface709",
        "expected_feature_dim": EXPECTED_FEATURE_DIM,
        "metadata_columns_removed": sorted(META_COLUMNS),
        "feature_columns": schema_columns,
        "label_mapping": {
            "0": "not engaged: Engagement < 2 or any Boredom/Confusion/Frustration >= 2",
            "1": "engaged: Engagement >= 2 and Boredom/Confusion/Frustration < 2",
        },
        "labels_csv": str(labels_csv),
        "openface_binary": str(openface_binary),
        "openface_flags": ["-2Dfp", "-3Dfp", "-pdmparams", "-pose", "-aus", "-gaze"],
        "sequence_length": int(sequence_length),
        "max_frames": int(max_frames),
        "min_confidence": float(min_confidence),
    }
    schema_path.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")

    manifest_df = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame(manifest_rows)

    summary = {
        "manifest": str(manifest_path),
        "schema": str(schema_path),
        "rows": len(manifest_rows),
        "feature_dim": EXPECTED_FEATURE_DIM,
        "split_counts": split_counts,
        "label_counts": label_counts,
        "unresolved": unresolved,
        "failed": failed,
        "empty": empty,
        "schema_mismatch": schema_mismatch,
        "skipped_already_done": skipped_already_done,
        "elapsed_sec": time.time() - start_time,
    }
    (output_root / "extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_state(
        state_path,
        {
            "manifest": str(manifest_path),
            "processed_videos": len(processed_video_ids),
            "manifest_rows": len(manifest_df),
            "finished": True,
            "updated_at": time.time(),
        },
    )
    LOGGER.info("Finished OpenFace709 extraction | %s", json.dumps(summary, indent=2))
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract exact OpenFace 709-D frame features for DAiSEE binary engagement.")
    parser.add_argument("--labels", type=Path, default=PROCESSED_LABELS_CSV)
    parser.add_argument("--videos", type=Path, default=RAW_VIDEO_DIR)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--openface-binary",
        type=Path,
        default=BASE_DIR / "external" / "openface" / "OpenFace" / "build" / "bin" / "FeatureExtraction",
    )
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sample-videos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=20, help="Flush manifest every N newly processed videos")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing manifest if present")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume behavior")
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    extract_openface709(
        labels_csv=args.labels,
        raw_video_dir=args.videos,
        output_root=args.output_root,
        openface_binary=args.openface_binary,
        sample=args.sample,
        sample_videos=args.sample_videos,
        seed=args.seed,
        sequence_length=args.sequence_length,
        max_frames=args.max_frames,
        min_confidence=args.min_confidence,
        force=args.force,
        log_every=args.log_every,
        resume=(args.resume and not args.no_resume),
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
