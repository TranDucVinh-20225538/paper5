"""Unit tests for timm loader helpers."""

from __future__ import annotations

import torch

from src.backbone.loaders.timm_loader import _pool_forward_features


def test_pool_gap_4d() -> None:
    x = torch.randn(2, 64, 7, 7)
    out = _pool_forward_features(x, "gap")
    assert out.shape == (2, 64)


def test_pool_cls_3d() -> None:
    x = torch.randn(2, 197, 768)
    out = _pool_forward_features(x, "cls")
    assert out.shape == (2, 768)
