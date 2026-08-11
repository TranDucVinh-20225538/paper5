"""Tests for configs/*.yaml loading and protocol hard stops."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import (
    effective_r_grid,
    load_backbone_config,
    validate_for_pipeline_run,
)


def test_load_panderm_config(panderm_config_path: Path) -> None:
    cfg = load_backbone_config(panderm_config_path)
    assert cfg.name == "panderm"
    assert cfg.embed_dim == 1024
    assert cfg.activation == "gelu"
    assert cfg.is_representation_resolved
    assert cfg.intervention_r == 16
    assert cfg.intervention_lambda_proj == 0.1
    assert cfg.seeds == [42, 52, 62, 72, 82]


def test_load_resnet50_config(resnet50_config_path: Path) -> None:
    cfg = load_backbone_config(resnet50_config_path)
    assert cfg.name == "resnet50"
    assert cfg.embed_dim == 2048
    assert cfg.activation == "relu"
    assert cfg.pooling == "gap"
    assert cfg.intervention_r is None


def test_load_medsam_config(medsam_config_path: Path) -> None:
    cfg = load_backbone_config(medsam_config_path)
    assert cfg.name == "medsam"
    assert cfg.embed_dim == 768
    assert cfg.pooling == "gap_masked"
    assert cfg.is_probe
    assert cfg.is_representation_resolved


def test_effective_r_grid_d021_high_dim() -> None:
    base = [16, 32, 64, 128]
    assert 256 in effective_r_grid(2048, base)
    assert 256 not in effective_r_grid(1024, base)


def test_validate_for_pipeline_run_rejects_unresolved(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        """
backbone:
  name: bad
  family: cnn
  variant: x
  loader: fixture
  checkpoint: x
  embed_dim: 128
  activation: relu
  representation:
    status: UNRESOLVED
    pooling: none
preprocessing:
  asset: assets/preprocessing/x.json
  sha256: null
intervention:
  r: null
  lambda_proj: null
  grid:
    r: [16]
    lambda_proj: [0.1]
seeds: [42]
""",
        encoding="utf-8",
    )
    cfg = load_backbone_config(cfg_path, repo_root=tmp_path)
    with pytest.raises(ValueError, match="UNRESOLVED"):
        validate_for_pipeline_run(cfg)
