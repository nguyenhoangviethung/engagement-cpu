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

        self.manifest = pd.read_csv(self.manifest_csv)
        required_columns = {"feature_path", "label"}
        missing_columns = required_columns - set(self.manifest.columns)
        if missing_columns:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        features = np.load(row["feature_path"]).astype(np.float32)
        label = np.float32(row["label"])
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.float32)