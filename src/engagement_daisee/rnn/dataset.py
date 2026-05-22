from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class FeatureSequenceDataset(Dataset):
    def __init__(self, manifest_csv: str | Path):
        self.manifest_csv = Path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_csv}")

        manifest = pd.read_csv(self.manifest_csv)
        required_columns = {"feature_path", "label"}
        missing_columns = required_columns - set(manifest.columns)
        if missing_columns:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

        self.data_cache: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        valid_rows: list[dict] = []

        iterator = manifest.iterrows()
        for _, row in iterator:
            feature_path = Path(row["feature_path"])
            try:
                features = np.load(feature_path)
                features_tensor = torch.as_tensor(features, dtype=torch.float32)
                label_tensor = torch.tensor(float(row["label"]), dtype=torch.float32)
            except Exception as exc:
                print(f"Warning: failed to load {feature_path}: {exc}")
                continue

            self.data_cache.append(features_tensor)
            self.labels.append(label_tensor)
            valid_rows.append(row.to_dict())

        if not self.labels:
            raise RuntimeError(
                "No valid feature files were loaded into cache. "
                f"Check manifest and feature files: {self.manifest_csv}"
            )

        self.manifest = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.data_cache[index], self.labels[index]