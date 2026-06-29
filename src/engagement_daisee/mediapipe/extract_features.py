from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from engagement_daisee.common.config import RAW_VIDEO_DIR, VIDEO_EXTENSIONS

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    raise RuntimeError("mediapipe is required for 504-feature extraction") from exc


LOGGER = logging.getLogger("extract_504_features")

DEFAULT_FACE_LANDMARKER = Path("external/mediapipe/face_landmarker.task")
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# 56 stable facial landmarks. 56 * (x, y, z) = 168 dims per frame.
# The final model consumes velocity/std enrichment: 168 * 3 = 504 dims.
SELECTED_LANDMARKS = [
    # face contour / pose anchors
    10, 152, 234, 454, 127, 356, 93, 323,
    # eyebrows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # eyes + eyelids
    33, 133, 160, 159, 158, 144, 145, 153,
    362, 263, 387, 386, 385, 373, 374, 380,
    # iris proxies / eye centers
    468, 473,
    # nose
    1, 2, 98, 327, 168, 197,
    # mouth
    61, 291, 13, 14, 78, 308, 81, 311, 178, 402,
    # cheeks / jaw support
    50, 280, 205, 425, 172, 397,
]
FRAME_DIM = len(SELECTED_LANDMARKS) * 3
FEATURE_DIM = FRAME_DIM * 3
WINDOW_SIZE = 30


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def _build_video_index(raw_video_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ext in VIDEO_EXTENSIONS:
        for path in raw_video_dir.rglob(f"*{ext}"):
            index.setdefault(path.stem.lower(), path)
            index.setdefault(path.name.lower(), path)
    return index


def _resolve_video_path(video_id: str, raw_video_dir: Path, index: dict[str, Path]) -> Path | None:
    candidate = Path(str(video_id))
    if candidate.is_file():
        return candidate
    nested = raw_video_dir / candidate
    if nested.is_file():
        return nested
    return index.get(candidate.stem.lower()) or index.get(candidate.name.lower()) or index.get(str(video_id).lower())


def _frame_feature_168(landmarks, image_width: int, image_height: int) -> np.ndarray:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    selected = pts[SELECTED_LANDMARKS].copy()

    # Normalize translation and scale so distance-to-camera changes are less harmful.
    face_min = pts[:, :2].min(axis=0)
    face_max = pts[:, :2].max(axis=0)
    center = 0.5 * (face_min + face_max)
    scale = float(max(face_max[0] - face_min[0], face_max[1] - face_min[1], 1e-6))
    aspect = float(image_width / max(1, image_height))

    selected[:, 0] = (selected[:, 0] - center[0]) / scale
    selected[:, 1] = ((selected[:, 1] - center[1]) * aspect) / scale
    selected[:, 2] = selected[:, 2] / scale
    return np.nan_to_num(selected.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _detect_frame(face_landmarker, frame_bgr: np.ndarray) -> np.ndarray | None:
    height, width = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect(image)
    if not result.face_landmarks:
        return None
    return _frame_feature_168(result.face_landmarks[0], width, height)


def _interpolate_missing(sequence: list[np.ndarray | None]) -> np.ndarray:
    frame = pd.DataFrame(
        [np.full(FRAME_DIM, np.nan, dtype=np.float32) if item is None else item for item in sequence]
    )
    frame = frame.interpolate(limit_direction="both").fillna(0.0)
    return frame.to_numpy(dtype=np.float32)


def _enrich_window(raw_window: np.ndarray) -> np.ndarray:
    raw_window = np.asarray(raw_window, dtype=np.float32)
    velocity = np.diff(raw_window, axis=0, prepend=raw_window[:1])
    std = np.repeat(raw_window.std(axis=0, keepdims=True), raw_window.shape[0], axis=0)
    return np.concatenate([raw_window, velocity, std], axis=1).astype(np.float32)


def _windows_504(sequence_168: np.ndarray, window_size: int = WINDOW_SIZE) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    for start in range(0, len(sequence_168) - window_size + 1, window_size):
        windows.append(_enrich_window(sequence_168[start : start + window_size]))
    return windows


def extract_video_504(video_path: Path, face_landmarker, *, frame_stride: int = 1, max_frames: int = 0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    sequence: list[np.ndarray | None] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(1, frame_stride) == 0:
            sequence.append(_detect_frame(face_landmarker, frame))
            if max_frames > 0 and len(sequence) >= max_frames:
                break
        frame_index += 1
    cap.release()
    if not sequence:
        return []
    return _windows_504(_interpolate_missing(sequence), window_size=WINDOW_SIZE)


def run_extract(args: argparse.Namespace) -> dict[str, object]:
    labels = pd.read_csv(args.labels_csv)
    if "video_id" not in labels.columns:
        raise ValueError("labels csv must contain video_id")
    if "label" not in labels.columns:
        raise ValueError("labels csv must contain label")
    if "split" not in labels.columns:
        labels["split"] = "train"

    raw_video_dir = args.raw_video_dir
    output_root = args.output_dir
    feature_dir = output_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    video_index = _build_video_index(raw_video_dir)

    rows = []
    face_landmarker = _create_face_landmarker(args.face_landmarker)
    try:
        for row_idx, row in labels.iterrows():
            if args.limit_videos > 0 and row_idx >= args.limit_videos:
                break
            video_id = str(row["video_id"])
            video_path = _resolve_video_path(video_id, raw_video_dir, video_index)
            if video_path is None:
                LOGGER.warning("Missing video for %s", video_id)
                continue
            LOGGER.info("Extracting %s", video_path)
            windows = extract_video_504(
                video_path,
                face_landmarker,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
            )
            for segment_index, window in enumerate(windows):
                out_path = feature_dir / f"{Path(video_id).stem}_seg{segment_index:04d}.npy"
                np.save(out_path, window.astype(np.float32))
                rows.append(
                    {
                        "feature_path": out_path.as_posix(),
                        "video_id": Path(video_id).name,
                        "video_path": video_path.as_posix(),
                        "label": int(row["label"]),
                        "split": str(row["split"]).strip().lower(),
                        "segment_index": segment_index,
                        "num_frames": int(window.shape[0]),
                        "feature_set": "depth_robust_v2",
                        "temporal_enrichment": "velocity_std",
                        "feature_dim": FEATURE_DIM,
                        "storage_dtype": "float32",
                    }
                )
    finally:
        face_landmarker.close()

    manifest = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "feature_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "status": "success",
        "manifest": manifest_path.as_posix(),
        "rows": len(manifest),
        "videos": int(manifest["video_id"].nunique()) if len(manifest) else 0,
        "feature_dim": FEATURE_DIM,
        "window_size": WINDOW_SIZE,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 30x504 depth-robust MediaPipe feature windows.")
    parser.add_argument("--labels-csv", type=Path, default=Path("data/processed/engagement_only_labels.csv"))
    parser.add_argument("--raw-video-dir", type=Path, default=RAW_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/runs/triple_xgb_504_features"))
    parser.add_argument("--face-landmarker", type=Path, default=DEFAULT_FACE_LANDMARKER)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--limit-videos", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    print(json.dumps(run_extract(parse_args()), indent=2))


if __name__ == "__main__":
    main()
