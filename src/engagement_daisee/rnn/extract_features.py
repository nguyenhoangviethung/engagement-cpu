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
DENSE709_LANDMARK_COUNT = 229
DENSE709_SCALAR_DIM = 22
DEPTH_ROBUST_SCALAR_DIM = 25
DEPTH_ROBUST_LANDMARK_CLIP = 10.0
DEPTH_ROBUST_V2_SCALAR_DIM = 25
DEPTH_ROBUST_V2_BLENDSHAPE_DIM = 52
DEPTH_ROBUST_V2_TRANSFORM_DIM = 16
DEPTH_ROBUST_V2_VALIDITY_INDEX = 24
DEPTH_ROBUST_V2_CLIP = 100.0


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
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self._warm_start = True

    def start_video(self) -> None:
        self._warm_start = True

    def _detect(self, image):
        result = self._detector.detect_for_video(image, self._timestamp_ms)
        self._timestamp_ms += 1
        return result

    def process(self, rgb_frame):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self._warm_start:
            self._detect(image)
            self._warm_start = False
        result = self._detect(image)
        faces = [_TaskLandmarkList(face) for face in result.face_landmarks]
        return type(
            "FaceMeshResult",
            (),
            {
                "multi_face_landmarks": faces,
                "face_blendshapes": result.face_blendshapes,
                "facial_transformation_matrixes": result.facial_transformation_matrixes,
            },
        )()

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


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + 1e-6))


def _dense_scalar_features(landmarks) -> np.ndarray:
    left_eye_outer = _lm_xy(landmarks, 33)
    left_eye_inner = _lm_xy(landmarks, 133)
    right_eye_inner = _lm_xy(landmarks, 362)
    right_eye_outer = _lm_xy(landmarks, 263)
    left_mouth = _lm_xy(landmarks, 61)
    right_mouth = _lm_xy(landmarks, 291)
    upper_lip = _lm_xy(landmarks, 13)
    lower_lip = _lm_xy(landmarks, 14)
    nose = _lm_xy(landmarks, 1)
    nose_tip = _lm_xy(landmarks, 4)
    forehead = _lm_xy(landmarks, 10)
    chin = _lm_xy(landmarks, 152)
    left_cheek = _lm_xy(landmarks, 234)
    right_cheek = _lm_xy(landmarks, 454)
    face_center = (left_cheek + right_cheek + forehead + chin) / 4.0
    left_eye_center = (left_eye_outer + left_eye_inner) / 2.0
    right_eye_center = (right_eye_outer + right_eye_inner) / 2.0
    eye_center = (left_eye_center + right_eye_center) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0

    face_width = _distance(left_cheek, right_cheek) + 1e-6
    face_height = _distance(forehead, chin) + 1e-6
    inter_eye = _distance(left_eye_center, right_eye_center) + 1e-6
    mouth_width = _distance(left_mouth, right_mouth)
    mouth_open = _distance(upper_lip, lower_lip)
    roll = math.atan2(float(right_eye_center[1] - left_eye_center[1]), float(right_eye_center[0] - left_eye_center[0]) + 1e-6)

    return np.asarray(
        [
            _eye_aspect_ratio(landmarks, LEFT_EYE),
            _eye_aspect_ratio(landmarks, RIGHT_EYE),
            _mouth_aspect_ratio(landmarks),
            (nose[0] - eye_center[0]) / inter_eye,
            (nose[1] - eye_center[1]) / face_height,
            roll,
            face_width,
            face_height,
            _safe_ratio(inter_eye, face_width),
            _safe_ratio(mouth_width, face_width),
            _safe_ratio(mouth_open, mouth_width),
            _safe_ratio(_distance(eye_center, mouth_center), face_height),
            _safe_ratio(_distance(nose, chin), face_height),
            _safe_ratio(_distance(forehead, nose), face_height),
            _safe_ratio(_distance(left_eye_center, mouth_center), face_height),
            _safe_ratio(_distance(right_eye_center, mouth_center), face_height),
            (mouth_center[0] - nose[0]) / face_width,
            (mouth_center[1] - nose[1]) / face_height,
            (nose_tip[0] - face_center[0]) / face_width,
            (nose_tip[1] - face_center[1]) / face_height,
            _safe_ratio(_distance(left_cheek, nose), face_width),
            _safe_ratio(_distance(right_cheek, nose), face_width),
        ],
        dtype=np.float32,
    )


def _dense_landmark_indices(landmarks) -> np.ndarray:
    landmark_count = len(landmarks.landmark)
    return np.linspace(0, landmark_count - 1, num=DENSE709_LANDMARK_COUNT, dtype=np.int64)


def _depth_robust_features(landmarks) -> np.ndarray:
    """Build camera-distance-aware features in a canonical face coordinate system.

    MediaPipe's x/y/z values are monocular and do not provide metric depth.  We
    therefore expose inverse face-scale proxies to the model, while normalizing
    landmark shape by inter-eye distance and in-plane roll.  This makes facial
    shape substantially less sensitive to camera distance without discarding
    the distance signal entirely.
    """
    geometry = _face_geometry_proxy(landmarks)
    left_eye_center = (_lm_xy(landmarks, 33) + _lm_xy(landmarks, 133)) / 2.0
    right_eye_center = (_lm_xy(landmarks, 362) + _lm_xy(landmarks, 263)) / 2.0
    eye_center = (left_eye_center + right_eye_center) / 2.0
    mouth_center = (_lm_xy(landmarks, 61) + _lm_xy(landmarks, 291)) / 2.0
    nose_xy = _lm_xy(landmarks, 1)
    nose_z = float(_lm_xyz(landmarks, 1)[2])

    inter_eye = _distance(left_eye_center, right_eye_center) + 1e-6
    forehead = _lm_xy(landmarks, 10)
    chin = _lm_xy(landmarks, 152)
    left_cheek = _lm_xy(landmarks, 234)
    right_cheek = _lm_xy(landmarks, 454)
    bbox_width = _distance(left_cheek, right_cheek) + 1e-6
    bbox_height = _distance(forehead, chin) + 1e-6

    selected_xyz = np.stack([_lm_xyz(landmarks, index) for index in ENHANCED_LANDMARKS], axis=0)
    z_span = float(np.ptp(selected_xyz[:, 2]))
    depth_proxies = [
        inter_eye,
        -math.log(inter_eye),
        bbox_width,
        bbox_height,
        math.sqrt(bbox_width * bbox_height),
        nose_z,
        z_span,
        bbox_width / bbox_height,
    ]

    # Rotate x/y by -roll, center on the eye midpoint and scale by inter-eye
    # distance. z is relative to the nose and uses the same scale as x/y.
    roll = float(geometry[2])
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    rotation = np.asarray([[cos_roll, sin_roll], [-sin_roll, cos_roll]], dtype=np.float32)
    canonical: list[float] = []
    for point in selected_xyz:
        xy = rotation @ ((point[:2] - eye_center) / inter_eye)
        relative_z = (float(point[2]) - nose_z) / inter_eye
        canonical.extend([float(xy[0]), float(xy[1]), relative_z])

    relational = []
    face_width = float(geometry[3]) + 1e-6
    face_height = float(geometry[4]) + 1e-6
    relational.extend(((nose_xy - left_eye_center) / face_width).tolist())
    relational.extend(((nose_xy - right_eye_center) / face_width).tolist())
    relational.extend(((mouth_center - nose_xy) / face_height).tolist())

    scalar_features = [
        _eye_aspect_ratio(landmarks, LEFT_EYE),
        _eye_aspect_ratio(landmarks, RIGHT_EYE),
        _mouth_aspect_ratio(landmarks),
        *geometry.tolist(),
        *relational,
        *depth_proxies,
    ]
    if len(scalar_features) != DEPTH_ROBUST_SCALAR_DIM:
        raise RuntimeError(f"Unexpected depth-robust scalar dimension: {len(scalar_features)}")
    result = np.asarray([*scalar_features, *canonical], dtype=np.float32)
    return np.nan_to_num(
        np.clip(result, -DEPTH_ROBUST_LANDMARK_CLIP, DEPTH_ROBUST_LANDMARK_CLIP),
        nan=0.0,
        posinf=DEPTH_ROBUST_LANDMARK_CLIP,
        neginf=-DEPTH_ROBUST_LANDMARK_CLIP,
    )


def _lm_xy_aspect(landmarks, index: int, image_aspect_ratio: float) -> np.ndarray:
    """Return x/y in image-height units so Euclidean geometry is isotropic."""
    point = landmarks.landmark[index]
    return np.asarray([point.x * image_aspect_ratio, point.y], dtype=np.float32)


def _lm_xyz_aspect(landmarks, index: int, image_aspect_ratio: float) -> np.ndarray:
    """Aspect-correct x/y and z, whose MediaPipe scale is approximately x."""
    point = landmarks.landmark[index]
    return np.asarray(
        [point.x * image_aspect_ratio, point.y, point.z * image_aspect_ratio],
        dtype=np.float32,
    )


def _eye_aspect_ratio_corrected(landmarks, indices: list[int], image_aspect_ratio: float) -> float:
    points = [_lm_xy_aspect(landmarks, index, image_aspect_ratio) for index in indices]
    vertical = _distance(points[1], points[5]) + _distance(points[2], points[4])
    horizontal = 2.0 * _distance(points[0], points[3]) + 1e-6
    return vertical / horizontal


def _mouth_aspect_ratio_corrected(landmarks, image_aspect_ratio: float) -> float:
    top = _lm_xy_aspect(landmarks, 13, image_aspect_ratio)
    bottom = _lm_xy_aspect(landmarks, 14, image_aspect_ratio)
    left = _lm_xy_aspect(landmarks, 61, image_aspect_ratio)
    right = _lm_xy_aspect(landmarks, 291, image_aspect_ratio)
    return _distance(top, bottom) / (_distance(left, right) + 1e-6)


def _iris_diameter(landmarks, indices: tuple[int, int, int, int], image_aspect_ratio: float) -> float:
    if len(landmarks.landmark) <= max(indices):
        return 0.0
    horizontal = _distance(
        _lm_xy_aspect(landmarks, indices[0], image_aspect_ratio),
        _lm_xy_aspect(landmarks, indices[1], image_aspect_ratio),
    )
    vertical = _distance(
        _lm_xy_aspect(landmarks, indices[2], image_aspect_ratio),
        _lm_xy_aspect(landmarks, indices[3], image_aspect_ratio),
    )
    return 0.5 * (horizontal + vertical)


def _blendshape_vector(face_blendshapes) -> np.ndarray:
    if not face_blendshapes:
        return np.zeros(DEPTH_ROBUST_V2_BLENDSHAPE_DIM, dtype=np.float32)
    categories = face_blendshapes[0] if isinstance(face_blendshapes, list) else face_blendshapes
    ordered = sorted(categories, key=lambda category: str(getattr(category, "category_name", "")))
    scores = [float(getattr(category, "score", 0.0)) for category in ordered]
    scores = (scores + [0.0] * DEPTH_ROBUST_V2_BLENDSHAPE_DIM)[:DEPTH_ROBUST_V2_BLENDSHAPE_DIM]
    return np.asarray(scores, dtype=np.float32)


def _transformation_vector(facial_transformation_matrixes) -> np.ndarray:
    if not facial_transformation_matrixes:
        return np.zeros(DEPTH_ROBUST_V2_TRANSFORM_DIM, dtype=np.float32)
    values = np.asarray(facial_transformation_matrixes[0], dtype=np.float32).reshape(-1)
    result = np.zeros(DEPTH_ROBUST_V2_TRANSFORM_DIM, dtype=np.float32)
    result[: min(len(values), len(result))] = values[: len(result)]
    return result


def _depth_robust_v2_features(
    landmarks,
    image_aspect_ratio: float,
    face_blendshapes=None,
    facial_transformation_matrixes=None,
) -> np.ndarray:
    """Distance-aware, aspect-correct and canonically aligned face features."""
    aspect_ratio = max(float(image_aspect_ratio), 1e-6)
    left_eye_center = (
        _lm_xy_aspect(landmarks, 33, aspect_ratio) + _lm_xy_aspect(landmarks, 133, aspect_ratio)
    ) / 2.0
    right_eye_center = (
        _lm_xy_aspect(landmarks, 362, aspect_ratio) + _lm_xy_aspect(landmarks, 263, aspect_ratio)
    ) / 2.0
    eye_center = (left_eye_center + right_eye_center) / 2.0
    left_mouth = _lm_xy_aspect(landmarks, 61, aspect_ratio)
    right_mouth = _lm_xy_aspect(landmarks, 291, aspect_ratio)
    mouth_center = (left_mouth + right_mouth) / 2.0
    nose = _lm_xy_aspect(landmarks, 1, aspect_ratio)
    forehead = _lm_xy_aspect(landmarks, 10, aspect_ratio)
    chin = _lm_xy_aspect(landmarks, 152, aspect_ratio)
    left_cheek = _lm_xy_aspect(landmarks, 234, aspect_ratio)
    right_cheek = _lm_xy_aspect(landmarks, 454, aspect_ratio)

    inter_eye = _distance(left_eye_center, right_eye_center) + 1e-6
    face_width = _distance(left_cheek, right_cheek) + 1e-6
    face_height = _distance(forehead, chin) + 1e-6
    roll = math.atan2(
        float(right_eye_center[1] - left_eye_center[1]),
        float(right_eye_center[0] - left_eye_center[0]) + 1e-6,
    )
    yaw = float((nose[0] - eye_center[0]) / inter_eye)
    pitch = float((nose[1] - mouth_center[1]) / face_height)

    selected_xyz = np.stack(
        [_lm_xyz_aspect(landmarks, index, aspect_ratio) for index in ENHANCED_LANDMARKS], axis=0
    )
    nose_z = float(_lm_xyz_aspect(landmarks, 1, aspect_ratio)[2])
    raw_z_span = float(np.ptp(selected_xyz[:, 2]))
    normalized_z_span = raw_z_span / inter_eye

    left_iris = _iris_diameter(landmarks, (469, 471, 470, 472), aspect_ratio)
    right_iris = _iris_diameter(landmarks, (474, 476, 475, 477), aspect_ratio)
    valid_irises = [diameter for diameter in (left_iris, right_iris) if diameter > 1e-6]
    mean_iris = float(np.mean(valid_irises)) if valid_irises else 0.0
    log_inverse_iris = -math.log(mean_iris + 1e-6) if mean_iris > 0.0 else 0.0

    relational = [
        (nose[0] - left_eye_center[0]) / face_width,
        (nose[1] - left_eye_center[1]) / face_height,
        (nose[0] - right_eye_center[0]) / face_width,
        (nose[1] - right_eye_center[1]) / face_height,
        (mouth_center[0] - nose[0]) / face_width,
        (mouth_center[1] - nose[1]) / face_height,
    ]
    scalar_features = np.asarray(
        [
            _eye_aspect_ratio_corrected(landmarks, LEFT_EYE, aspect_ratio),
            _eye_aspect_ratio_corrected(landmarks, RIGHT_EYE, aspect_ratio),
            _mouth_aspect_ratio_corrected(landmarks, aspect_ratio),
            pitch,
            yaw,
            roll,
            face_width,
            face_height,
            inter_eye,
            -math.log(inter_eye),
            math.sqrt(face_width * face_height),
            face_width / face_height,
            normalized_z_span,
            raw_z_span,
            left_iris,
            right_iris,
            mean_iris,
            log_inverse_iris,
            *relational,
            1.0,
        ],
        dtype=np.float32,
    )
    if len(scalar_features) != DEPTH_ROBUST_V2_SCALAR_DIM:
        raise RuntimeError(f"Unexpected depth-robust-v2 scalar dimension: {len(scalar_features)}")

    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    rotation = np.asarray([[cos_roll, sin_roll], [-sin_roll, cos_roll]], dtype=np.float32)
    canonical: list[float] = []
    for point in selected_xyz:
        xy = rotation @ ((point[:2] - eye_center) / inter_eye)
        relative_z = (float(point[2]) - nose_z) / inter_eye
        canonical.extend([float(xy[0]), float(xy[1]), relative_z])

    result = np.concatenate(
        [
            scalar_features,
            np.asarray(canonical, dtype=np.float32),
            _blendshape_vector(face_blendshapes),
            _transformation_vector(facial_transformation_matrixes),
        ]
    )
    return np.nan_to_num(
        np.clip(result, -DEPTH_ROBUST_V2_CLIP, DEPTH_ROBUST_V2_CLIP),
        nan=0.0,
        posinf=DEPTH_ROBUST_V2_CLIP,
        neginf=-DEPTH_ROBUST_V2_CLIP,
    ).astype(np.float32, copy=False)


def _build_frame_feature(
    landmarks,
    feature_set: str,
    image_aspect_ratio: float = 1.0,
    face_blendshapes=None,
    facial_transformation_matrixes=None,
) -> np.ndarray:
    if feature_set == "dense709":
        features = _dense_scalar_features(landmarks).tolist()
        for landmark_index in _dense_landmark_indices(landmarks):
            features.extend(_lm_xyz(landmarks, int(landmark_index)).tolist())
        return np.asarray(features, dtype=np.float32)
    if feature_set == "depth_robust":
        return _depth_robust_features(landmarks)
    if feature_set == "depth_robust_v2":
        return _depth_robust_v2_features(
            landmarks,
            image_aspect_ratio=image_aspect_ratio,
            face_blendshapes=face_blendshapes,
            facial_transformation_matrixes=facial_transformation_matrixes,
        )

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
    if feature_set == "dense709":
        return DENSE709_SCALAR_DIM + DENSE709_LANDMARK_COUNT * 3
    if feature_set == "depth_robust":
        return DEPTH_ROBUST_SCALAR_DIM + len(ENHANCED_LANDMARKS) * 3
    if feature_set == "depth_robust_v2":
        return (
            DEPTH_ROBUST_V2_SCALAR_DIM
            + len(ENHANCED_LANDMARKS) * 3
            + DEPTH_ROBUST_V2_BLENDSHAPE_DIM
            + DEPTH_ROBUST_V2_TRANSFORM_DIM
        )
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def _empty_feature(feature_set: str) -> np.ndarray:
    return np.zeros(_frame_feature_dim(feature_set), dtype=np.float32)


def _interpolate_missing_v2(sequence: list[np.ndarray | None]) -> list[np.ndarray]:
    """Interpolate missing detections while retaining a zero validity mask."""
    if not sequence:
        return []
    valid_indices = [index for index, value in enumerate(sequence) if value is not None]
    if not valid_indices:
        return [_empty_feature("depth_robust_v2") for _ in sequence]

    output: list[np.ndarray | None] = [None] * len(sequence)
    for index in valid_indices:
        output[index] = np.asarray(sequence[index], dtype=np.float32).copy()

    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    for index in range(first_valid):
        output[index] = output[first_valid].copy()
    for index in range(last_valid + 1, len(output)):
        output[index] = output[last_valid].copy()

    for left_index, right_index in zip(valid_indices[:-1], valid_indices[1:]):
        gap = right_index - left_index
        if gap <= 1:
            continue
        left_value = output[left_index]
        right_value = output[right_index]
        for offset in range(1, gap):
            alpha = offset / gap
            output[left_index + offset] = ((1.0 - alpha) * left_value + alpha * right_value).astype(np.float32)

    valid_set = set(valid_indices)
    final: list[np.ndarray] = []
    for index, value in enumerate(output):
        if value is None:
            value = _empty_feature("depth_robust_v2")
        if index not in valid_set:
            value[DEPTH_ROBUST_V2_VALIDITY_INDEX] = 0.0
        final.append(value)
    return final


def _pad_sequence(sequence: list[np.ndarray], feature_set: str) -> np.ndarray:
    if not sequence:
        return np.zeros((SEQUENCE_LENGTH, _frame_feature_dim(feature_set)), dtype=np.float32)

    stacked_sequence = np.stack(sequence, axis=0).astype(np.float32)
    if len(sequence) >= SEQUENCE_LENGTH:
        return stacked_sequence[:SEQUENCE_LENGTH]

    pad_count = SEQUENCE_LENGTH - len(sequence)
    edge_padding = np.repeat(stacked_sequence[-1:], pad_count, axis=0)
    return np.concatenate([stacked_sequence, edge_padding], axis=0)


def _window_frames(
    frame_features: list[np.ndarray],
    feature_set: str,
    temporal_enrichment: str = "velocity_std",
) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    for start_index in range(0, len(frame_features), SEQUENCE_LENGTH):
        chunk = frame_features[start_index : start_index + SEQUENCE_LENGTH]
        padded_chunk = _pad_sequence(chunk, feature_set=feature_set)
        if temporal_enrichment == "none":
            windows.append(padded_chunk.astype(np.float32, copy=False))
            continue
        if temporal_enrichment != "velocity_std":
            raise ValueError(f"Unsupported temporal_enrichment: {temporal_enrichment}")

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


def _resolve_label_column(labels_df: pd.DataFrame, label_mode: str) -> tuple[str, int]:
    if label_mode == "four_class":
        label_column, num_classes = "engagement_raw", 4
    elif label_mode == "binary":
        label_column, num_classes = "engagement_binary", 2
    else:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    if label_column not in labels_df.columns:
        raise ValueError(
            f"Label mode '{label_mode}' requires column '{label_column}' in the labels CSV. "
            f"Available columns: {sorted(labels_df.columns)}"
        )
    numeric_labels = pd.to_numeric(labels_df[label_column], errors="coerce")
    invalid = numeric_labels.isna() | (numeric_labels % 1 != 0) | ~numeric_labels.between(0, num_classes - 1)
    if invalid.any():
        invalid_values = labels_df.loc[invalid, label_column].head(10).tolist()
        raise ValueError(
            f"Column '{label_column}' contains labels outside 0..{num_classes - 1}: {invalid_values}"
        )
    labels_df[label_column] = numeric_labels.astype(int)
    return label_column, num_classes


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
    frame_features: list[np.ndarray | None] = []
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
    start_video = getattr(face_mesh, "start_video", None)
    if callable(start_video):
        start_video()
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
                frame_features.append(
                    _build_frame_feature(
                        results.multi_face_landmarks[0],
                        feature_set=feature_set,
                        image_aspect_ratio=frame.shape[1] / max(1.0, float(frame.shape[0])),
                        face_blendshapes=getattr(results, "face_blendshapes", None),
                        facial_transformation_matrixes=getattr(
                            results, "facial_transformation_matrixes", None
                        ),
                    )
                )
            else:
                frame_features.append(None if feature_set == "depth_robust_v2" else _empty_feature(feature_set))
            frame_index += 1
    finally:
        capture.release()
        if close_face_mesh:
            face_mesh.close()

    if feature_set == "depth_robust_v2":
        return _interpolate_missing_v2(frame_features)
    return [feature for feature in frame_features if feature is not None]


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
    temporal_enrichment: str = "velocity_std",
    label_mode: str = "four_class",
    num_shards: int = 1,
    shard_index: int = 0,
    storage_dtype: str = "float32",
) -> Path:
    start_time = time.time()
    LOGGER.info("Starting feature extraction")
    LOGGER.info(
        "Config | labels=%s | videos=%s | features=%s | manifest=%s | sample=%s | seed=%d | frame_stride=%d | max_frames=%d | resize_width=%d | feature_set=%s | temporal_enrichment=%s | label_mode=%s | shard=%d/%d | output_dim=%d | storage_dtype=%s",
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
        temporal_enrichment,
        label_mode,
        shard_index,
        num_shards,
        _frame_feature_dim(feature_set) * (1 if temporal_enrichment == "none" else 3),
        storage_dtype,
    )

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError(f"storage_dtype must be float16 or float32, got {storage_dtype}")
    output_dtype = np.float16 if storage_dtype == "float16" else np.float32

    labels_df = _collect_video_records(processed_labels_csv)
    label_column, num_classes = _resolve_label_column(labels_df, label_mode=label_mode)
    labels_df = _choose_rows(labels_df, sample=sample, seed=seed)
    total_label_rows = len(labels_df)
    labels_df = labels_df.iloc[shard_index::num_shards].reset_index(drop=True)
    video_index = _build_video_index(raw_video_dir)

    LOGGER.info(
        "Loaded shard %d/%d | rows=%d of total=%d",
        shard_index,
        num_shards,
        len(labels_df),
        total_label_rows,
    )
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

            sequences = _window_frames(
                raw_features,
                feature_set=feature_set,
                temporal_enrichment=temporal_enrichment,
            )
            saved_videos += 1
            for segment_index, sequence in enumerate(sequences):
                sequence_file = features_dir / f"{Path(video_id).stem}_seg{segment_index:04d}.npy"
                np.save(sequence_file, sequence.astype(output_dtype))
                saved_sequences += 1
                manifest_rows.append(
                    {
                        "feature_path": str(sequence_file),
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "label": int(row[label_column]),
                        "label_mode": label_mode,
                        "label_source_column": label_column,
                        "num_classes": num_classes,
                        "split": str(
                            row.get("split", row.get("official_split", row.get("partition", "unknown")))
                        ).lower(),
                        "segment_index": segment_index,
                        "num_frames": int(len(raw_features)),
                        "frame_stride": int(max(1, frame_stride)),
                        "max_frames": int(max(0, max_frames)),
                        "resize_width": int(max(0, resize_width)),
                        "feature_set": feature_set,
                        "temporal_enrichment": temporal_enrichment,
                        "feature_dim": int(sequences[0].shape[-1]),
                        "shard_index": int(shard_index),
                        "num_shards": int(num_shards),
                        "storage_dtype": storage_dtype,
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
    parser.add_argument(
        "--feature-set",
        type=str,
        default="base",
        choices=["base", "enhanced", "dense709", "depth_robust", "depth_robust_v2"],
        help="Facial feature recipe",
    )
    parser.add_argument(
        "--temporal-enrichment",
        type=str,
        default="velocity_std",
        choices=["none", "velocity_std"],
        help="Append per-frame velocity and per-window std copies. Use none for exact frame feature dimension.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="four_class",
        choices=["four_class", "binary"],
        help="Target label recipe. four_class uses engagement_raw (0..3).",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Split label rows across this many workers")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based worker shard index")
    parser.add_argument(
        "--storage-dtype",
        choices=["float16", "float32"],
        default="float32",
        help="On-disk dtype; loaders cast to float32 during training",
    )
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
        temporal_enrichment=args.temporal_enrichment,
        label_mode=args.label_mode,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        storage_dtype=args.storage_dtype,
    )
    LOGGER.info("Saved feature manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
