import os
from pathlib import Path


def _resolve_daisee_root() -> Path:
    env_candidates = [
        os.getenv("ENGAGEMENT_DAISEE_ROOT"),
        os.getenv("DAISEE_ROOT"),
        os.getenv("RAW_DAISEE_ROOT"),
        os.getenv("RAW_DATA_DIR"),
    ]
    path_candidates = []
    for value in env_candidates:
        if value:
            path_candidates.append(Path(value).expanduser())

    path_candidates.extend(
        [
            BASE_DIR / "data" / "raw" / "daisee" / "DAiSEE",
            BASE_DIR / "data" / "raw" / "DAiSEE",
            BASE_DIR.parent / "data" / "raw" / "daisee" / "DAiSEE",
            BASE_DIR.parent / "data" / "raw" / "DAiSEE",
            Path.home() / "engagement-cpu" / "data" / "raw" / "daisee" / "DAiSEE",
            Path.home() / "engagement-cpu" / "data" / "raw" / "DAiSEE",
            Path("/mnt") / "data" / "raw" / "daisee" / "DAiSEE",
            Path("/mnt") / "data" / "raw" / "DAiSEE",
        ]
    )

    for candidate in path_candidates:
        resolved = candidate.expanduser()
        if resolved.name == "DataSet":
            resolved = resolved.parent
        if (resolved / "DataSet").is_dir() and (resolved / "Labels").is_dir():
            return resolved
    return BASE_DIR / "data" / "raw" / "daisee" / "DAiSEE"


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = _resolve_daisee_root()
RAW_DATASET_DIR = RAW_DATA_DIR / "DataSet"
RAW_LABELS_DIR = RAW_DATA_DIR / "Labels"
RAW_LABELS_CSV = RAW_LABELS_DIR / "AllLabels.csv"
RAW_TRAIN_LABELS_CSV = RAW_LABELS_DIR / "TrainLabels.csv"
RAW_VALIDATION_LABELS_CSV = RAW_LABELS_DIR / "ValidationLabels.csv"
RAW_TEST_LABELS_CSV = RAW_LABELS_DIR / "TestLabels.csv"
RAW_VIDEO_DIR = RAW_DATASET_DIR
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
PROCESSED_LABELS_CSV = PROCESSED_DIR / "engagement_only_labels.csv"
FEATURE_MANIFEST_CSV = PROCESSED_DIR / "feature_manifest.csv"
FOUR_CLASS_DEPTH_ROBUST_RUN_DIR = (
    PROCESSED_DIR / "runs" / "extract_depth_robust_5w_20260620_130850"
)
FOUR_CLASS_FEATURE_MANIFEST_CSV = FEATURE_MANIFEST_CSV

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
MODEL_CHECKPOINT_PATH = CHECKPOINT_DIR / "engagement_gru.pt"
MODEL_CHECKPOINT_ML = CHECKPOINT_DIR / "engagement_gru_ml.pt"
SEQUENCE_LENGTH = 30
FEATURE_DIM = 90
DEPTH_ROBUST_V2_FRAME_DIM = 168
DEPTH_ROBUST_V2_FEATURE_DIM = DEPTH_ROBUST_V2_FRAME_DIM * 3
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 3e-4
HIDDEN_SIZE = 128
DROPOUT = 0.3
VAL_RATIO = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 0
DEVICE = "cpu"
SAMPLE_VIDEO_COUNT = 10
SAMPLE_EPOCHS = 2
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 6

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")
