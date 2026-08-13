"""Forward-pass unit tests for production Step 3 loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.backbone.loaders.model_factory import create_backbone_model, verify_backbone_forward
from src.utils.config import load_backbone_config

PRODUCTION_BACKBONES = (
    "monet",
    "siglip",
    "mocov3",
    "dinov3",
    "biomedclip",
    "openclip",
    "medsam",
)


@pytest.mark.checkpoint
@pytest.mark.parametrize("backbone_name", PRODUCTION_BACKBONES)
def test_loader_forward_shape(repo_root: Path, backbone_name: str) -> None:
    """One CPU forward pass; shape must match config embed_dim."""
    cfg_path = repo_root / "configs" / f"{backbone_name}.yaml"
    cfg = load_backbone_config(cfg_path, repo_root=repo_root)
    model = create_backbone_model(cfg)
    out = verify_backbone_forward(cfg, model)
    assert out.ndim == 2
    assert out.shape == (1, cfg.embed_dim)


@pytest.mark.parametrize("backbone_name", PRODUCTION_BACKBONES)
def test_extract_dispatch_registered(repo_root: Path, backbone_name: str) -> None:
    """Step 3 dispatch must route to a real loader, not NotImplementedError."""
    from src.backbone.extract import extract_embeddings

    cfg_path = repo_root / "configs" / f"{backbone_name}.yaml"
    cfg = load_backbone_config(cfg_path, repo_root=repo_root)
    assert cfg.loader in {"hf", "open_clip", "medsam"}
    # Dispatch check only — do not run real extraction here.
    loader_name = cfg.loader.lower()
    if loader_name == "hf":
        from src.backbone.loaders.hf_loader import extract_hf_embeddings

        assert extract_hf_embeddings is not None
    elif loader_name == "open_clip":
        from src.backbone.loaders.open_clip_loader import extract_open_clip_embeddings

        assert extract_open_clip_embeddings is not None
    elif loader_name == "medsam":
        from src.backbone.loaders.medsam_loader import extract_medsam_embeddings

        assert extract_medsam_embeddings is not None
    else:
        pytest.fail(f"unexpected loader {loader_name!r}")
    assert extract_embeddings.__doc__ is not None
