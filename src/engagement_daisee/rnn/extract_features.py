import argparse
import logging
import math
import random
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from engagement_daisee.common.config import (
    FEATURE_MANIFEST_CSV,
    FEATURES_DIR,
    PROCESSED_LABELS_CSV,
    RAW_VIDEO_DIR,
    SAMPLE_VIDEO_COUNT,
    SEQUENCE_LENGTH,
    VIDEO_EXTENSIONS,
)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
BASE_LANDMARKS = [33, 133, 362, 263, 61, 291, 13, 14]
ENHANCED_LANDMARKS = [
    1,
    4,
    10,
    33,
    46,
    52,
    61,
    78,
    93,
    133,
    152,
    159,
    168,
    172,
    197,
    234,
    263,
    276,
    282,
    291,
    308,
    323,
    362,
    386,
    454,
]


LOGGER = logging.getLogger("extract_features")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = Path.home() / ".cache" / "engagement_daisee" / "face_landmarker.task"


class _TaskLandmarkList:
    def __init__(self, landmarks) -> None:
        self.landmark = landmarks


class _TaskFaceLandmarker:
    def __init__(self, model_path: Path) -> None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_frame):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(image)
        faces = [_TaskLandmarkList(face) for face in result.face_landmarks]
        return type("FaceMeshResult", (), {"multi_face_landmarks": faces})()

    def close(self) -> None:
        self._detector.close()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _ensure_face_landmarker_model(model_path: Path = FACE_LANDMARKER_MODEL_PATH) -> Path:
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading MediaPipe FaceLandmarker model to %s", model_path)
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, model_path)
    return model_path


def _create_face_landmarker():
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "face_mesh"):
        return solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    model_path = _ensure_face_landmarker_model()
    return _TaskFaceLandmarker(model_path)


def _lm_xy(landmarks, index: int) -> np.ndarray:
    point = landmarks.landmark[index]
    return np.array([point.x, point.y], dtype=np.float32)


def _lm_xyz(landmarks, index: int) -> np.ndarray:
    point = landmarks.landmark[index]
    return np.array([point.x, point.y, point.z], dtype=np.float32)


def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def _eye_aspect_ratio(landmarks, eye_indices: list[int]) -> float:
    p1 = _lm_xy(landmarks, eye_indices[0])
    p2 = _lm_xy(landmarks, eye_indices[1])
    p3 = _lm_xy(landmarks, eye_indices[2])
    p4 = _lm_xy(landmarks, eye_indices[3])
    p5 = _lm_xy(landmarks, eye_indices[4])
    p6 = _lm_xy(landmarks, eye_indices[5])
    vertical = _distance(p2, p6) + _distance(p3, p5)
    horizontal = 2.0 * _distance(p1, p4) + 1e-6
    return vertical / horizontal


def _mouth_aspect_ratio(landmarks) -> float:
    top = _lm_xy(landmarks, 13)
    bottom = _lm_xy(landmarks, 14)
    left = _lm_xy(landmarks, 61)
    right = _lm_xy(landmarks, 291)
    vertical = _distance(top, bottom)
    horizontal = _distance(left, right) + 1e-6
    return vertical / horizontal


def _face_geometry_proxy(landmarks) -> np.ndarray:
    nose = _lm_xy(landmarks, 1)
    chin = _lm_xy(landmarks, 152)
    left_eye = _lm_xy(landmarks, 33)
    right_eye = _lm_xy(landmarks, 263)
    mouth_left = _lm_xy(landmarks, 61)
    mouth_right = _lm_xy(landmarks, 291)

    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (mouth_left + mouth_right) / 2.0
    face_width = _distance(left_eye, right_eye) + 1e-6
    face_height = _distance(eye_center, chin) + 1e-6

    yaw = (nose[0] - eye_center[0]) / face_width
    pitch = (nose[1] - mouth_center[1]) / face_height
    roll = math.atan2(float(right_eye[1] - left_eye[1]), float(right_eye[0] - left_eye[0]) + 1e-6)
    mouth_width = _distance(mouth_left, mouth_right) / face_width
    eye_mouth_distance = _distance(eye_center, mouth_center) / face_height
    nose_chin_distance = _distance(nose, chin) / face_height
    return np.array(
        [pitch, yaw, roll, face_width, face_height, mouth_width, eye_mouth_distance, nose_chin_distance],
        dtype=np.float32,
    )


def _build_frame_feature(landmarks, feature_set: str) -> np.ndarray:
    features = [
        _eye_aspect_ratio(landmarks, LEFT_EYE),
        _eye_aspect_ratio(landmarks, RIGHT_EYE),
        _mouth_aspect_ratio(landmarks),
    ]
    geometry = _face_geometry_proxy(landmarks)
    if feature_set == "base":
        features.extend(geometry[:3].tolist())
        landmark_indices = BASE_LANDMARKS
    elif feature_set == "enhanced":
        features.extend(geometry.tolist())
        left_eye_center = (_lm_xy(landmarks, 33) + _lm_xy(landmarks, 133)) / 2.0
        right_eye_center = (_lm_xy(landmarks, 362) + _lm_xy(landmarks, 263)) / 2.0
        mouth_center = (_lm_xy(landmarks, 61) + _lm_xy(landmarks, 291)) / 2.0
        nose = _lm_xy(landmarks, 1)
        face_width = float(geometry[3]) + 1e-6
        face_height = float(geometry[4]) + 1e-6
        features.extend(((nose - left_eye_center) / face_width).tolist())
        features.extend(((nose - right_eye_center) / face_width).tolist())
        features.extend(((mouth_center - nose) / face_height).tolist())
        landmark_indices = ENHANCED_LANDMARKS
    else:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    for landmark_index in landmark_indices:
        features.extend(_lm_xyz(landmarks, landmark_index).tolist())
    return np.asarray(features, dtype=np.float32)


def _frame_feature_dim(feature_set: str) -> int:
    if feature_set == "base":
        return 3 + 3 + len(BASE_LANDMARKS) * 3
    if feature_set == "enhanced":
        return 3 + 8 + 6 + len(ENHANCED_LANDMARKS) * 3
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def _empty_feature(feature_set: str) -> np.ndarray:
    return np.zeros(_frame_feature_dim(feature_set), dtype=np.float32)


def _pad_sequence(sequence: list[np.ndarray], feature_set: str) -> np.ndarray:
    if not sequence:
        return np.zeros((SEQUENCE_LENGTH, _frame_feature_dim(feature_set)), dtype=np.float32)

    stacked_sequence = np.stack(sequence, axis=0).astype(np.float32)
    if len(sequence) >= SEQUENCE_LENGTH:
        return stacked_sequence[:SEQUENCE_LENGTH]

    pad_count = SEQUENCE_LENGTH - len(sequence)
    edge_padding = np.repeat(stacked_sequence[-1:], pad_count, axis=0)
    return np.concatenate([stacked_sequence, edge_padding], axis=0)


def _window_frames(frame_features: list[np.ndarray], feature_set: str) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    for start_index in range(0, len(frame_features), SEQUENCE_LENGTH):
        chunk = frame_features[start_index : start_index + SEQUENCE_LENGTH]
        padded_chunk = _pad_sequence(chunk, feature_set=feature_set)

        velocity = np.zeros_like(padded_chunk, dtype=np.float32)
        velocity[1:] = padded_chunk[1:] - padded_chunk[:-1]

        window_std = np.std(padded_chunk, axis=0).astype(np.float32)
        std_matrix = np.tile(window_std, (SEQUENCE_LENGTH, 1))

        enriched_chunk = np.concatenate([padded_chunk, velocity, std_matrix], axis=-1).astype(np.float32, copy=False)
        windows.append(enriched_chunk)
    return windows


def _collect_video_records(labels_csv: Path) -> pd.DataFrame:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Processed labels CSV not found: {labels_csv}")
    labels_df = pd.read_csv(labels_csv)
    if "video_id" not in labels_df.columns:
        raise ValueError("The processed labels CSV must contain a video_id column.")
    return labels_df


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


def _selected_frame_indices(total_frames: int, frame_stride: int, max_frames: int) -> set[int] | None:
    if max_frames > 0 and total_frames > max_frames:
        return set(int(i) for i in np.linspace(0, total_frames - 1, num=max_frames, dtype=np.int64))
    if frame_stride > 1 and total_frames > 0:
        return set(range(0, total_frames, frame_stride))
    return None


def _resize_for_landmarks(frame, resize_width: int):
    if resize_width <= 0 or frame.shape[1] <= resize_width:
        return frame
    scale = resize_width / float(frame.shape[1])
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA)


def _extract_video_features(
    video_path: Path,
    frame_stride: int = 1,
    max_frames: int = 0,
    resize_width: int = 0,
    feature_set: str = "base",
    face_mesh=None,
) -> list[np.ndarray]:
    frame_features: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return frame_features
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    selected_indices = _selected_frame_indices(
        total_frames=total_frames,
        frame_stride=max(1, frame_stride),
        max_frames=max(0, max_frames),
    )

    close_face_mesh = False
    if face_mesh is None:
        face_mesh = _create_face_landmarker()
        close_face_mesh = True
    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if selected_indices is not None and frame_index not in selected_indices:
                frame_index += 1
                continue
            frame = _resize_for_landmarks(frame, resize_width=resize_width)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                frame_features.append(_build_frame_feature(results.multi_face_landmarks[0], feature_set=feature_set))
            else:
                frame_features.append(_empty_feature(feature_set=feature_set))
            frame_index += 1
    finally:
        capture.release()
        if close_face_mesh:
            face_mesh.close()

    return frame_features


def _choose_rows(labels_df: pd.DataFrame, sample: bool, seed: int) -> pd.DataFrame:
    if not sample:
        return labels_df
    unique_videos = labels_df.drop_duplicates(subset=["video_id"]).copy()
    if len(unique_videos) <= SAMPLE_VIDEO_COUNT:
        return unique_videos

    if "split" in unique_videos.columns:
        sampled_frames = []
        per_split = max(1, SAMPLE_VIDEO_COUNT // max(1, unique_videos["split"].nunique()))
        for split_name, split_frame in unique_videos.groupby("split", sort=False):
            take_n = min(len(split_frame), per_split)
            sampled_frames.append(split_frame.sample(n=take_n, random_state=seed))

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
        return sampled_df.reset_index(drop=True)

    return unique_videos.sample(n=SAMPLE_VIDEO_COUNT, random_state=seed).reset_index(drop=True)


def extract_features(
    processed_labels_csv: Path,
    raw_video_dir: Path,
    features_dir: Path,
    manifest_csv: Path,
    sample: bool = False,
    seed: int = 42,
    log_every: int = 10,
    frame_stride: int = 1,
    max_frames: int = 0,
    resize_width: int = 0,
    feature_set: str = "base",
) -> Path:
    start_time = time.time()
    LOGGER.info("Starting feature extraction")
    LOGGER.info(
        "Config | labels=%s | videos=%s | features=%s | manifest=%s | sample=%s | seed=%d | frame_stride=%d | max_frames=%d | resize_width=%d | feature_set=%s | output_dim=%d",
        processed_labels_csv,
        raw_video_dir,
        features_dir,
        manifest_csv,
        sample,
        seed,
        frame_stride,
        max_frames,
        resize_width,
        feature_set,
        _frame_feature_dim(feature_set) * 3,
    )

    labels_df = _collect_video_records(processed_labels_csv)
    labels_df = _choose_rows(labels_df, sample=sample, seed=seed)
    video_index = _build_video_index(raw_video_dir)

    LOGGER.info("Loaded %d label rows after sampling", len(labels_df))
    LOGGER.info("Indexed %d unique videos from raw directory", len(video_index))

    features_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    rng = random.Random(seed)
    rows = labels_df.to_dict(orient="records")
    if sample:
        rng.shuffle(rows)

    total_rows = len(rows)
    unresolved_count = 0
    unreadable_count = 0
    saved_sequences = 0
    saved_videos = 0

    LOGGER.info("Processing %d videos", total_rows)

    face_mesh = _create_face_landmarker()
    try:
        for row_index, row in enumerate(rows, start=1):
            video_id = str(row["video_id"])
            video_path = _resolve_video_path(video_id, raw_video_dir, video_index)
            if video_path is None:
                unresolved_count += 1
                LOGGER.warning("Skipping unresolved video | video_id=%s", video_id)
                continue

            raw_features = _extract_video_features(
                video_path,
                frame_stride=frame_stride,
                max_frames=max_frames,
                resize_width=resize_width,
                feature_set=feature_set,
                face_mesh=face_mesh,
            )
            if not raw_features:
                unreadable_count += 1
                LOGGER.warning("Skipping unreadable or empty video | video_path=%s", video_path)
                continue

            sequences = _window_frames(raw_features, feature_set=feature_set)
            saved_videos += 1
            for segment_index, sequence in enumerate(sequences):
                sequence_file = features_dir / f"{Path(video_id).stem}_seg{segment_index:04d}.npy"
                np.save(sequence_file, sequence.astype(np.float32))
                saved_sequences += 1
                manifest_rows.append(
                    {
                        "feature_path": str(sequence_file),
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "label": int(row["engagement_binary"]),
                        "split": str(row.get("split", "unknown")).lower(),
                        "segment_index": segment_index,
                        "num_frames": int(len(raw_features)),
                        "frame_stride": int(max(1, frame_stride)),
                        "max_frames": int(max(0, max_frames)),
                        "resize_width": int(max(0, resize_width)),
                        "feature_set": feature_set,
                        "feature_dim": int(sequences[0].shape[-1]),
                    }
                )

            if row_index % max(1, log_every) == 0 or row_index == total_rows:
                LOGGER.info(
                    "Progress %d/%d | saved_videos=%d | saved_sequences=%d | unresolved=%d | unreadable=%d",
                    row_index,
                    total_rows,
                    saved_videos,
                    saved_sequences,
                    unresolved_count,
                    unreadable_count,
                )
    finally:
        face_mesh.close()

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_csv, index=False)

    elapsed = time.time() - start_time
    LOGGER.info(
        "Finished extraction in %.2fs | manifest_rows=%d | output=%s",
        elapsed,
        len(manifest_df),
        manifest_csv,
    )
    return manifest_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract lightweight MediaPipe features for engagement detection.")
    parser.add_argument("--labels", type=Path, default=PROCESSED_LABELS_CSV, help="Clean engagement-only labels CSV")
    parser.add_argument("--videos", type=Path, default=RAW_VIDEO_DIR, help="Directory containing DAiSEE videos")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR, help="Directory to save .npy feature files")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FEATURE_MANIFEST_CSV,
        help="CSV file mapping saved features to labels",
    )
    parser.add_argument("--sample", action="store_true", help="Process only 10 random videos for a quick test run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=10, help="Log progress every N videos")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame before windowing")
    parser.add_argument("--max-frames", type=int, default=0, help="Uniformly sample at most N frames per video; 0 keeps all")
    parser.add_argument("--resize-width", type=int, default=0, help="Resize frames to this width before landmark detection; 0 disables")
    parser.add_argument("--feature-set", type=str, default="base", choices=["base", "enhanced"], help="Facial feature recipe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging()
    manifest_path = extract_features(
        processed_labels_csv=args.labels,
        raw_video_dir=args.videos,
        features_dir=args.features_dir,
        manifest_csv=args.manifest,
        sample=args.sample,
        seed=args.seed,
        log_every=args.log_every,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        resize_width=args.resize_width,
        feature_set=args.feature_set,
    )
    LOGGER.info("Saved feature manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
