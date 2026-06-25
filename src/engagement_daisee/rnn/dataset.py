from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from engagement_daisee.common.manifest import normalize_manifest_columns

class FeatureSequenceDataset(Dataset):
    def __init__(self, manifest_csv: str | Path):
        self.manifest_csv = Path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_csv}")

        manifest = normalize_manifest_columns(pd.read_csv(self.manifest_csv))
        required_columns = {"feature_path", "label"}
        missing_columns = required_columns - set(manifest.columns)
        if missing_columns:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

        self.feature_paths = [Path(value) for value in manifest["feature_path"].tolist()]
        self.label_values = manifest["label"].astype(float).tolist()
        declared_dims = set()
        if "feature_dim" in manifest.columns:
            declared_dims = set(pd.to_numeric(manifest["feature_dim"], errors="coerce").dropna().astype(int))
            if len(declared_dims) > 1:
                raise ValueError(f"Manifest contains inconsistent feature_dim values: {sorted(declared_dims)}")

        if not self.feature_paths:
            raise RuntimeError(f"Manifest contains no feature rows: {self.manifest_csv}")
        first = self._load_array(0, expected_shape=None)
        self.sequence_length, self.feature_dim = (int(first.shape[0]), int(first.shape[1]))
        if declared_dims and self.feature_dim != next(iter(declared_dims)):
            raise ValueError(
                f"Manifest declares feature_dim={next(iter(declared_dims))}, first file has {self.feature_dim}"
            )
        self.manifest = manifest.reset_index(drop=True)

    def _load_array(self, index: int, expected_shape: tuple[int, int] | None) -> np.ndarray:
        feature_path = self.feature_paths[index]
        try:
            features = np.load(feature_path, allow_pickle=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load feature file {feature_path}: {exc}") from exc
        if features.ndim != 2:
            raise ValueError(f"Expected {feature_path} to have shape (sequence, feature), got {features.shape}")
        if expected_shape is not None and tuple(features.shape) != expected_shape:
            raise ValueError(f"Expected {feature_path} to have shape {expected_shape}, got {features.shape}")
        return features

    def __len__(self) -> int:
        return len(self.label_values)

    def __getitem__(self, index: int):
        expected_shape = (self.sequence_length, self.feature_dim)
        features = self._load_array(index, expected_shape=expected_shape)
        features_tensor = torch.as_tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(self.label_values[index], dtype=torch.float32)
        return features_tensor, label_tensor
