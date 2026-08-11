"""Integration test for Steps 4–6 on fixture backbone."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.steps import run_steps_0_through_6
from src.utils.config import load_backbone_config

FIXTURE_CFG = (
    Path(__file__).resolve().parent / "fixtures" / "configs" / "fixture_backbone.yaml"
)


@pytest.mark.slow
def test_fixture_steps_0_through_6(repo_root: Path, tmp_path: Path) -> None:
    """Full M3 path on synthetic embeddings — grid uses 2 epochs for speed."""
    manifest = tmp_path / "manifest.jsonl"
    record = run_steps_0_through_6(
        FIXTURE_CFG,
        repo_root=repo_root,
        manifest_path=manifest,
        require_split_checksum=True,
        fixture_train_n=200,
        fixture_eval_n=120,
        grid_epochs=2,
        loader_override="fixture",
    )
    assert record["step"] == "6_gate0"
    assert record["gate0"]["gate0_pass"] is True
    assert record["selected_r"] in load_backbone_config(FIXTURE_CFG, repo_root=repo_root).grid_r
