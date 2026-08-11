"""Minimal image dataset for backbone extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MetadataImageDataset(Dataset):
    """Rows from master_metadata with columns path, label_idx, domain."""

    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[idx]
        path = Path(str(row["path"]))
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label_idx"])
        return image, label
