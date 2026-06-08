from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from engagement_daisee.common.manifest import normalize_manifest_columns


class DAiSEECNNFrameDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, indices: list[int] | None = None, transform=None):
        self.manifest_csv = Path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")

        df = normalize_manifest_columns(pd.read_csv(self.manifest_csv))
        required = {"frame_path", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        self.manifest = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        image_path = Path(row["frame_path"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image, label
