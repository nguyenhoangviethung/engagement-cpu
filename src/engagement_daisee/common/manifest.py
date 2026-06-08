from __future__ import annotations

import pandas as pd


PARTITION_COLUMN = "partition"
LEGACY_PARTITION_COLUMN = "split"


def normalize_manifest_columns(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stable runtime aliases for renamed manifest columns."""
    frame = manifest.copy()
    if PARTITION_COLUMN in frame.columns and LEGACY_PARTITION_COLUMN not in frame.columns:
        frame[LEGACY_PARTITION_COLUMN] = frame[PARTITION_COLUMN]
    return frame
