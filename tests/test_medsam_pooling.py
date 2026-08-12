"""Tests for MedSAM masked GAP pooling (D-025, D-026)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.backbone.pooling import (
    masked_global_average_pool,
    medsam_valid_patch_mask,
    sam_resize_dimensions,
)


def test_sam_resize_landscape() -> None:
    h, w = sam_resize_dimensions(600, 800, target_long_side=1024)
    assert w == 1024
    assert h == int(round(600 * 1024 / 800))


def test_sam_resize_portrait() -> None:
    h, w = sam_resize_dimensions(800, 600, target_long_side=1024)
    assert h == 1024
    assert w == int(round(600 * 1024 / 800))


def test_medsam_mask_non_square_excludes_padding_patches() -> None:
    # 800×600 landscape → scaled 1024×768, padded to 1024×1024 bottom/right
    mask = medsam_valid_patch_mask(600, 800)
    assert mask.shape == (64, 64)
    assert mask[47, 63]        # last valid row, last valid col
    assert not mask[48, 63]    # first fully padded row
    assert mask[47, 64 - 1]
    assert mask.sum() == 48 * 64  # 768/16 = 48 valid rows, all 64 cols valid


def test_medsam_mask_square_image_full_grid() -> None:
    mask = medsam_valid_patch_mask(1024, 1024)
    assert mask.all()
    assert mask.shape == (64, 64)


def test_masked_pool_ignores_padding_tokens() -> None:
    grid = 4
    d = 3
    tokens = np.zeros((grid, grid, d), dtype=np.float64)
    tokens[:, :, 0] = 1.0  # padding region would read 1.0 if included

    # valid region: top-left 2×2 only
    mask = np.zeros((grid, grid), dtype=bool)
    mask[:2, :2] = True
    tokens[:2, :2, 0] = 0.0
    tokens[:2, :2, 1] = 2.0

    pooled = masked_global_average_pool(tokens, mask)
    assert pooled[0] == pytest.approx(0.0)
    assert pooled[1] == pytest.approx(2.0)


def test_masked_pool_empty_mask_forbidden() -> None:
    tokens = np.ones((4, 4, 8))
    mask = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="zero tokens"):
        masked_global_average_pool(tokens, mask)


def test_medsam_config_asserts_pre_neck_dimension():
    """D-043: 768 is pre-neck. 256 means the neck ran, which D-025 rejected.

    transformers applies the neck inside SamVisionEncoder.forward, so the obvious
    API call silently returns the wrong tensor at a plausible shape. The config
    carries the assertion so extraction can enforce it.
    """
    import yaml

    from src.utils.config import find_repo_root

    path = find_repo_root(Path(__file__)) / "configs" / "medsam.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["assertions"]["embed_dim_must_be"] == 768
    assert cfg["backbone"]["embed_dim"] == 768, "config must declare pre-neck 768, not post-neck 256"
    assert cfg["backbone"]["revision"], "D-044: MedSAM revision must be pinned"
