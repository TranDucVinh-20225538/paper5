"""Fixture-only Step 3 extraction (no GPU, no real checkpoints)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.backbone.extract import extract_embeddings
from src.utils.config import load_backbone_config

FIXTURE_CFG = (
    Path(__file__).resolve().parent / "fixtures" / "configs" / "fixture_backbone.yaml"
)


def test_fixture_step3_extract_embeddings(repo_root: Path, tmp_path: Path) -> None:
    cfg = load_backbone_config(FIXTURE_CFG, repo_root=repo_root)
    out = extract_embeddings(
        cfg,
        output_dir=tmp_path / "embeddings",
        loader="fixture",
        train_n=8,
        eval_n=4,
        fixture_seed=0,
    )
    assert out.backbone == "fixture_test"
    assert out.train_n == 8
    assert out.eval_n == 4
    assert not out.skipped
    assert out.embed_dim == cfg.embed_dim

    train = np.load(out.train_path)
    eval_ = np.load(out.eval_path)
    assert train.shape == (8, cfg.embed_dim)
    assert eval_.shape == (4, cfg.embed_dim)
    assert out.train_dir.is_dir()
    assert (out.train_dir / "metadata.csv").is_file()
    assert (out.eval_dir / "metadata.csv").is_file()
