"""
Spatial pooling utilities for backbone extraction (D-025, D-026).

MedSAM: masked global average pool over 16×16 patch tokens on a 64×64 grid.
Unmasked pooling is forbidden for MedSAM.
"""

from __future__ import annotations

import numpy as np


def sam_resize_dimensions(
    orig_height: int,
    orig_width: int,
    target_long_side: int = 1024,
) -> tuple[int, int]:
    """
    SAM/MedSAM step 1: resize longest side to target_long_side, preserve aspect ratio.
    """
    if orig_height <= 0 or orig_width <= 0:
        raise ValueError("orig_height and orig_width must be positive")
    if target_long_side <= 0:
        raise ValueError("target_long_side must be positive")

    if orig_height >= orig_width:
        new_h = target_long_side
        new_w = int(round(orig_width * target_long_side / orig_height))
    else:
        new_w = target_long_side
        new_h = int(round(orig_height * target_long_side / orig_width))
    return new_h, new_w


def medsam_valid_patch_mask(
    orig_height: int,
    orig_width: int,
    *,
    canvas_size: int = 1024,
    patch_size: int = 16,
    target_long_side: int = 1024,
) -> np.ndarray:
    """
    Boolean mask [grid, grid] for patch tokens with non-padded content (D-026).

    Preprocessing assumed:
      1. resize longest side to target_long_side (aspect ratio preserved)
      2. pad bottom/right with zeros to canvas_size × canvas_size

    A patch (i, j) is valid iff its top-left corner lies inside the non-padded
    content region — i.e. it overlaps at least one non-padded pixel.
    """
    new_h, new_w = sam_resize_dimensions(orig_height, orig_width, target_long_side)
    if new_h > canvas_size or new_w > canvas_size:
        raise ValueError(
            f"Scaled size ({new_h}, {new_w}) exceeds canvas {canvas_size}; "
            "check preprocessing parameters"
        )

    grid = canvas_size // patch_size
    mask = np.zeros((grid, grid), dtype=bool)
    for i in range(grid):
        for j in range(grid):
            if i * patch_size < new_h and j * patch_size < new_w:
                mask[i, j] = True
    return mask


def masked_global_average_pool(
    spatial_tokens: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    GAP over spatial tokens using only valid positions.

    Parameters
    ----------
    spatial_tokens :
        [H, W, D] or [H*W, D] feature map before pooling.
    valid_mask :
        [H, W] boolean mask; must align with the token grid.
    """
    tokens = np.asarray(spatial_tokens, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)

    if tokens.ndim == 3:
        if tokens.shape[:2] != mask.shape:
            raise ValueError(
                f"spatial_tokens shape {tokens.shape[:2]} != mask shape {mask.shape}"
            )
        flat = tokens.reshape(-1, tokens.shape[-1])
        flat_mask = mask.reshape(-1)
    elif tokens.ndim == 2:
        if mask.size != tokens.shape[0]:
            raise ValueError(
                f"mask size {mask.size} != token count {tokens.shape[0]}"
            )
        flat = tokens
        flat_mask = mask.reshape(-1)
    else:
        raise ValueError("spatial_tokens must be [H, W, D] or [N, D]")

    if not np.any(flat_mask):
        raise ValueError("valid_mask selects zero tokens — unmasked pooling forbidden")

    return flat[flat_mask].mean(axis=0)
