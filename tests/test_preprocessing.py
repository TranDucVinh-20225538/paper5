"""Tests for frozen preprocessing pipelines."""

from __future__ import annotations

from PIL import Image

from src.backbone.preprocessing_transform import load_preprocessing_pipeline


def test_uni_preprocessing_emits_square_tensors(repo_root) -> None:
    pipe = load_preprocessing_pipeline(repo_root / "assets/preprocessing/uni.json")
    wide = Image.new("RGB", (400, 300), color=(128, 64, 32))
    tall = Image.new("RGB", (300, 400), color=(32, 64, 128))
    assert tuple(pipe(wide).shape) == (3, 224, 224)
    assert tuple(pipe(tall).shape) == (3, 224, 224)
