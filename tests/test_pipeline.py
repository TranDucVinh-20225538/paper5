"""Pipeline hard stops and Step 3 integration (protocol Steps 0–3)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.backbone.pooling_dispatch import pool_representation
from src.pipeline.hard_stops import run_step0_hard_stops
from src.pipeline.steps import run_steps_0_through_3
from src.utils.config import find_repo_root, load_backbone_config

FIXTURE_CFG = (
    Path(__file__).resolve().parent / "fixtures" / "configs" / "fixture_backbone.yaml"
)


def test_step0_fails_null_checkpoint(repo_root: Path) -> None:
    # Uses a dedicated fixture, not MedSAM: MedSAM was pinned to
    # wanglab/medsam-vit-base @ de8488bc (D-043), so it no longer exercises this path.
    # The behaviour under test — Step 0 refuses a null checkpoint — still matters.
    path = repo_root / "tests" / "fixtures" / "configs" / "fixture_null_checkpoint.yaml"
    cfg = load_backbone_config(path, repo_root=repo_root)
    with pytest.raises(ValueError, match="checkpoint is null"):
        run_step0_hard_stops(
            cfg,
            repo_root,
            require_preprocessing_hash=False,
            require_split_checksum=False,
        )


def test_run_all_exits_nonzero_on_null_checkpoint(repo_root: Path) -> None:
    path = repo_root / "tests" / "fixtures" / "configs" / "fixture_null_checkpoint.yaml"
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline.cli", str(path), "--through-step", "0"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "checkpoint is null" in result.stderr or "checkpoint is null" in result.stdout


def test_fixture_pipeline_steps_0_through_3(repo_root: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    out_dir = tmp_path / "embeddings"
    record = run_steps_0_through_3(
        FIXTURE_CFG,
        repo_root=repo_root,
        output_dir=out_dir,
        manifest_path=manifest,
        require_split_checksum=True,
        loader_override="fixture",
        fixture_train_n=8,
        fixture_eval_n=4,
    )
    assert record["backbone"] == "fixture_test"
    assert record["train_n"] == 8
    assert record["eval_n"] == 4
    assert not record["skipped"]
    assert Path(record["train_dir"]).is_dir()
    assert (Path(record["train_dir"]) / "embeddings.npy").is_file()
    assert (Path(record["eval_dir"]) / "embeddings.npy").is_file()

    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["step"] == "3_extract_embeddings"
    assert parsed["train_sha256"] == record["train_sha256"]


def test_pooling_dispatch_gap_masked() -> None:
    grid = 64
    d = 768
    tokens = np.random.randn(grid, grid, d)
    vec = pool_representation(
        "gap_masked",
        tokens,
        embed_dim=d,
        orig_height=600,
        orig_width=800,
    )
    assert vec.shape == (d,)


def test_unresolved_representation_rejected(repo_root: Path, tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        """
backbone:
  name: bad
  family: cnn
  variant: x
  loader: fixture
  checkpoint: x
  embed_dim: 8
  activation: relu
  representation:
    status: UNRESOLVED
    pooling: none
preprocessing:
  asset: tests/fixtures/preprocessing/test_backbone.json
  sha256: null
intervention:
  r: null
  lambda_proj: null
  grid: {r: [16], lambda_proj: [0.1]}
  nuisance_direction: {formula: unit(mu_ISIC - mu_PAD), learned: false}
  alpha_ladder: [0.0]
seeds: [42]
""",
        encoding="utf-8",
    )
    cfg = load_backbone_config(cfg_path, repo_root=repo_root)
    with pytest.raises(ValueError, match="UNRESOLVED"):
        run_step0_hard_stops(
            cfg,
            repo_root,
            require_preprocessing_hash=False,
            require_split_checksum=False,
        )
