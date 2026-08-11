"""MedSAM extraction — protocol Step 3 (D-025/D-026). Skeleton until checkpoint wired."""

from __future__ import annotations

from pathlib import Path

from src.utils.config import BackboneConfig


def extract_medsam_embeddings(
    cfg: BackboneConfig,
    *,
    output_dir: Path,
) -> None:
    raise NotImplementedError(
        f"{cfg.name}: MedSAM loader skeleton only. "
        "Requires checkpoint path, gap_masked pooling on pre-neck 64×768 tokens, "
        "and assets/preprocessing/medsam.json. See docs/medsam_integration.md."
    )
