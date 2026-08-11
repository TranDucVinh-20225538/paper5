"""Tests for preprocessing transform builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.backbone.preprocessing_transform import load_preprocessing_pipeline


def test_load_resnet50_preprocessing(repo_root: Path) -> None:
    transform = load_preprocessing_pipeline(repo_root / "assets" / "preprocessing" / "resnet50.json")
    img = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
    out = transform(img)
    assert out.shape == (3, 224, 224)
