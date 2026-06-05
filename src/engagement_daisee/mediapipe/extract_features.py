from __future__ import annotations

import argparse
import json
import logging
import random
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from engagement_daisee.common.config import PROCESSED_LABELS_CSV, RAW_VIDEO_DIR, VIDEO_EXTENSIONS
from engagement_daisee.app.focus_monitor import (
    CHIN,
    LEFT_EYE,
    LEFT_IRIS,
    LOWER_LIP,
    NOSE_TIP,
    RIGHT_EYE,
    RIGHT_IRIS,
    UPPER_LIP,
    _dist,
    _eye_aspect_ratio,
    _normed,
)

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    raise RuntimeError("mediapipe is required for MediaPipe feature extraction") from exc


LOGGER = logging.getLogger("mediapipe_extract")
DEFAULT_OUTPUT_ROOT = Path("data/processed/runs/mediapipe_product/features")
DEFAULT_FACE_LANDMARKER = Path("external/mediapipe/face_landmarker.task")
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FEATURE_COLUMNS = [
    "face_present",
    "ear",
    "left_ear",
    "right_ear",
    "gaze_offset",
    "gaze_x",
    "gaze_y",
    "head_tilt",
    "head_yaw_proxy",
    "head_pitch_proxy",
    "mouth_open",
    "face_center_x",
    "face_center_y",
    "face_width",
    "face_height",
    "face_area",
    "nose_x",
    "nose_y",
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
            index.setdefault(path.stem.lower(), path)
            index.setdefault(path.name.lower(), path)
    return index


def _resolve_video_path(video_id: str, raw_video_dir: Path, index: dict[str, Path]) -> Path | None:
    candidate = Path(video_id)
    if candidate.is_file():
        return candidate
    nested = raw_video_dir / candidate
    if nested.is_file():
        return nested
    return index.get(candidate.stem.lower()) or index.get(candidate.name.lower()) or index.get(str(video_id).lower())


def _uniform_indices(total_frames: int, frames_per_video: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= frames_per_video:
        return list(range(total_frames))
    return sorted(set(int(i) for i in np.linspace(0, total_frames - 1, frames_per_video).round()))


def _ensure_face_landmarker_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading MediaPipe FaceLandmarker model to %s", model_path)
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, model_path)
    return model_path


def _create_face_landmarker(model_path: Path):
    model_path = _ensure_face_landmarker_model(model_path)
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _select_rows(labels: pd.DataFrame, sample_videos: int, seed: int) -> pd.DataFrame:
    rows = labels.drop_duplicates("video_id").copy()
    if sample_videos <= 0 or len(rows) <= sample_videos:
        return rows.reset_index(drop=True)
    parts = []
    per_split = max(1, sample_videos // max(1, rows["split"].nunique())) if "split" in rows.columns else sample_videos
    for _, split_rows in rows.groupby("split", sort=False):
        parts.append(split_rows.sample(n=min(per_split, len(split_rows)), random_state=seed))
    sampled = pd.concat(parts, ignore_index=True).drop_duplicates("video_id")
    remaining = sample_videos - len(sampled)
    if remaining > 0:
        pool = rows[~rows["video_id"].isin(sampled["video_id"])]
        if len(pool):
            sampled = pd.concat([sampled, pool.sample(n=min(remaining, len(pool)), random_state=seed)], ignore_index=True)
    records = sampled.to_dict("records")
    random.Random(seed).shuffle(records)
    return pd.DataFrame(records)


def _landmark_features(landmarks, width: int, height: int) -> np.ndarray:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    px = _normed(pts, width, height)
    left_ear = _eye_aspect_ratio(px, LEFT_EYE)
    right_ear = _eye_aspect_ratio(px, RIGHT_EYE)
    ear = 0.5 * (left_ear + right_ear)

    l_iris = np.mean(px[LEFT_IRIS, :2], axis=0)
    l_eye_center = 0.5 * (px[33, :2] + px[133, :2])
    r_iris = np.mean(px[RIGHT_IRIS, :2], axis=0)
    r_eye_center = 0.5 * (px[362, :2] + px[263, :2])
    l_eye_width = _dist(px[33, :2], px[133, :2])
    r_eye_width = _dist(px[362, :2], px[263, :2])
    l_offset = (l_iris - l_eye_center) / l_eye_width
    r_offset = (r_iris - r_eye_center) / r_eye_width
    gaze_xy = 0.5 * (l_offset + r_offset)
    gaze_offset = float(np.linalg.norm(gaze_xy))

    nose = px[NOSE_TIP, :2]
    chin = px[CHIN, :2]
    head_vec = chin - nose
    head_tilt = abs(float(np.arctan2(head_vec[0], head_vec[1] + 1e-6)))
    face_min = px[:, :2].min(axis=0)
    face_max = px[:, :2].max(axis=0)
    face_size = np.maximum(face_max - face_min, 1e-6)
    face_center = 0.5 * (face_min + face_max)
    head_yaw_proxy = float((nose[0] - face_center[0]) / face_size[0])
    head_pitch_proxy = float((nose[1] - face_center[1]) / face_size[1])
    mouth_open = _dist(px[UPPER_LIP, :2], px[LOWER_LIP, :2]) / _dist(px[33, :2], px[263, :2])

    return np.array(
        [
            1.0,
            ear,
            left_ear,
            right_ear,
            gaze_offset,
            float(gaze_xy[0]),
            float(gaze_xy[1]),
            head_tilt,
            head_yaw_proxy,
            head_pitch_proxy,
            mouth_open,
            float(face_center[0] / max(1, width)),
            float(face_center[1] / max(1, height)),
            float(face_size[0] / max(1, width)),
            float(face_size[1] / max(1, height)),
            float((face_size[0] * face_size[1]) / max(1, width * height)),
            float(nose[0] / max(1, width)),
            float(nose[1] / max(1, height)),
        ],
        dtype=np.float32,
    )


def _detect_landmarks(face_landmarker, rgb: np.ndarray):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect(image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def _extract_video_sequence(video_path: Path, frames_per_video: int, face_landmarker) -> tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32), 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _uniform_indices(total_frames, frames_per_video)
    wanted = set(indices)
    sequence: list[np.ndarray] = []
    current = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current in wanted:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = _detect_landmarks(face_landmarker, rgb)
            if landmarks:
                features = _landmark_features(landmarks, w, h)
            else:
                features = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)
            sequence.append(features)
            if len(sequence) >= len(indices):
                break
        current += 1
    cap.release()
    if not sequence:
        return np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32), total_frames
    return np.stack(sequence, axis=0).astype(np.float32), total_frames


def extract_mediapipe_features(
    labels_csv: Path,
    raw_video_dir: Path,
    output_root: Path,
    frames_per_video: int,
    sample_videos: int,
    seed: int,
    resume: bool,
    log_every: int,
    face_landmarker_model: Path,
) -> Path:
    start = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "mediapipe_feature_manifest.csv"
    state_path = output_root / "extract_state.json"

    labels = pd.read_csv(labels_csv)
    required = {"video_id", "engagement_binary", "split"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {sorted(missing)}")
    labels = _select_rows(labels, sample_videos=sample_videos, seed=seed)
    labels["video_id"] = labels["video_id"].astype(str)

    done_ids: set[str] = set()
    rows: list[dict[str, object]] = []
    if resume and manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        rows = existing.to_dict("records")
        done_ids = set(existing["video_id"].astype(str).tolist())
        LOGGER.info("Resume | existing_videos=%d manifest=%s", len(done_ids), manifest_path)

    index = _build_video_index(raw_video_dir)
    face_landmarker = _create_face_landmarker(face_landmarker_model)

    unresolved = unreadable = empty = 0
    pending = []
    for i, item in enumerate(labels.to_dict("records"), start=1):
        video_id = str(item["video_id"])
        if video_id in done_ids:
            continue
        video_path = _resolve_video_path(video_id, raw_video_dir, index)
        if video_path is None:
            unresolved += 1
            continue
        sequence, total_frames = _extract_video_sequence(video_path, frames_per_video, face_landmarker)
        if total_frames <= 0:
            unreadable += 1
            continue
        if len(sequence) == 0:
            empty += 1
            continue
        feature_path = features_dir / f"{Path(video_id).stem}.npy"
        np.save(feature_path, sequence)
        row = {
            "feature_path": str(feature_path),
            "video_id": video_id,
            "video_path": str(video_path),
            "split": str(item["split"]).lower(),
            "label": int(item["engagement_binary"]),
            "num_frames_raw": int(total_frames),
            "num_frames_used": int(len(sequence)),
            "face_present_ratio": float(sequence[:, 0].mean()),
            "feature_set": "mediapipe_facemesh_v1",
            "feature_dim": len(FEATURE_COLUMNS),
        }
        rows.append(row)
        pending.append(row)
        if len(pending) >= 50:
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            pending.clear()
        if i % max(1, log_every) == 0:
            LOGGER.info("Progress %d/%d | rows=%d unresolved=%d unreadable=%d empty=%d", i, len(labels), len(rows), unresolved, unreadable, empty)

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    summary = {
        "manifest": str(manifest_path),
        "videos": len(rows),
        "frames_per_video": frames_per_video,
        "feature_columns": FEATURE_COLUMNS,
        "unresolved": unresolved,
        "unreadable": unreadable,
        "empty": empty,
        "elapsed_sec": time.time() - start,
    }
    state_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Finished MediaPipe extraction | %s", json.dumps(summary, indent=2))
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CPU-friendly MediaPipe FaceMesh engagement features.")
    parser.add_argument("--labels", type=Path, default=PROCESSED_LABELS_CSV)
    parser.add_argument("--videos", type=Path, default=RAW_VIDEO_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frames-per-video", type=int, default=30)
    parser.add_argument("--sample-videos", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--face-landmarker-model", type=Path, default=DEFAULT_FACE_LANDMARKER)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()
    extract_mediapipe_features(
        labels_csv=args.labels,
        raw_video_dir=args.videos,
        output_root=args.output_root,
        frames_per_video=args.frames_per_video,
        sample_videos=args.sample_videos,
        seed=args.seed,
        resume=not args.no_resume,
        log_every=args.log_every,
        face_landmarker_model=args.face_landmarker_model,
    )


if __name__ == "__main__":
    main()
