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


def test_load_uni_preprocessing_non_square(repo_root: Path) -> None:
    transform = load_preprocessing_pipeline(repo_root / "assets" / "preprocessing" / "uni.json")
    img = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8))
    out = transform(img)
    assert out.shape == (3, 224, 224)


def test_load_siglip_preprocessing_non_square(repo_root: Path) -> None:
    transform = load_preprocessing_pipeline(repo_root / "assets" / "preprocessing" / "siglip.json")
    img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    out = transform(img)
    assert out.shape == (3, 256, 256)


def test_load_medsam_preprocessing_non_square(repo_root: Path) -> None:
    transform = load_preprocessing_pipeline(repo_root / "assets" / "preprocessing" / "medsam.json")
    img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    out = transform(img)
    assert out.shape == (3, 1024, 1024)


def test_load_dinov3_preprocessing_non_square(repo_root: Path) -> None:
    transform = load_preprocessing_pipeline(repo_root / "assets" / "preprocessing" / "dinov3.json")
    img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    out = transform(img)
    assert out.shape == (3, 224, 224)
