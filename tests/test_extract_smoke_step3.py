"""Step 3 smoke test: one fixture image -> one embedding per backbone."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.backbone.smoke_extract import extract_single_embedding
from src.utils.config import load_backbone_config

# (config stem, expected embed_dim) — PanDerm excluded (Paper 4 reuse path).
SMOKE_BACKBONES: tuple[tuple[str, int], ...] = (
    ("resnet50", 2048),
    ("efficientnet_b3", 1536),
    ("uni", 1024),
    ("monet", 1024),
    ("biomedclip", 768),
    ("openclip", 768),
    ("siglip", 1024),
    ("mocov3", 768),
    ("dinov3", 1024),
    ("medsam", 768),
)

SMOKE_IMAGE = Path(__file__).resolve().parent / "fixtures" / "images" / "smoke_rgb.png"


@pytest.fixture(scope="session")
def smoke_image_path() -> Path:
    if not SMOKE_IMAGE.is_file():
        pytest.fail(f"missing fixture image: {SMOKE_IMAGE}")
    return SMOKE_IMAGE


@pytest.mark.smoke
@pytest.mark.checkpoint
@pytest.mark.parametrize(("backbone_name", "expected_dim"), SMOKE_BACKBONES)
def test_smoke_step3_single_image(
    repo_root: Path,
    smoke_image_path: Path,
    backbone_name: str,
    expected_dim: int,
) -> None:
    cfg_path = repo_root / "configs" / f"{backbone_name}.yaml"
    cfg = load_backbone_config(cfg_path, repo_root=repo_root)
    assert cfg.embed_dim == expected_dim

    vec = extract_single_embedding(cfg, smoke_image_path)
    assert vec.shape == (expected_dim,)
    assert np.isfinite(vec).all()

    # User-facing contract: batch-of-one shape.
    batch = vec.reshape(1, -1)
    assert batch.shape == (1, expected_dim)
