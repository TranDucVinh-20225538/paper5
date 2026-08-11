"""
Pooling dispatch for backbone extraction (D-018, D-025, D-026).

Maps config representation.pooling to a vector from spatial tokens or passthrough.
"""

from __future__ import annotations

import numpy as np

from src.backbone.pooling import masked_global_average_pool, medsam_valid_patch_mask


def pool_representation(
    pooling: str | None,
    tokens: np.ndarray,
    *,
    embed_dim: int,
    orig_height: int | None = None,
    orig_width: int | None = None,
) -> np.ndarray:
    """
    Reduce spatial tokens to a single embedding vector per config.

    Parameters
    ----------
    pooling :
        Config value: none | cls | gap | gap_masked | attn_pool (passthrough vector).
    tokens :
        [D] already pooled, [H, W, D] spatial map, or [N, D] with external mask for gap_masked batch.
    orig_height, orig_width :
        Required for gap_masked (MedSAM per-image pooling).
    """
    if pooling in (None, "none"):
        vec = np.asarray(tokens, dtype=np.float64)
        if vec.ndim != 1:
            raise ValueError(f"pooling=none expects 1-D vector, got shape {vec.shape}")
        return vec

    if pooling == "gap":
        arr = np.asarray(tokens, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"pooling=gap expects [H, W, D], got shape {arr.shape}")
        return arr.mean(axis=(0, 1))

    if pooling == "gap_masked":
        if orig_height is None or orig_width is None:
            raise ValueError("gap_masked requires orig_height and orig_width (D-026)")
        arr = np.asarray(tokens, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"pooling=gap_masked expects [H, W, D], got shape {arr.shape}")
        mask = medsam_valid_patch_mask(orig_height, orig_width)
        if mask.shape != arr.shape[:2]:
            raise ValueError(
                f"token grid {arr.shape[:2]} != mask grid {mask.shape}"
            )
        return masked_global_average_pool(arr, mask)

    if pooling in ("cls", "attn_pool"):
        vec = np.asarray(tokens, dtype=np.float64)
        if vec.ndim == 1:
            return vec
        if vec.ndim == 2 and vec.shape[0] == 1:
            return vec[0]
        raise ValueError(
            f"pooling={pooling} expects a pre-pooled [D] or [1, D] vector, got {vec.shape}"
        )

    raise ValueError(f"Unsupported pooling mode: {pooling!r}")
