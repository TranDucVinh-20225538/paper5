"""Tests for MedSAM masked GAP pooling (D-025, D-026)."""

from __future__ import annotations

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
    assert mask[47, 63] is True   # last valid row, last valid col
    assert mask[48, 63] is False  # first fully padded row
    assert mask[47, 64 - 1] is True
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
