"""Tests for scheduler backbone phase detection."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.scheduler_state import (
    backbone_name_from_config,
    backbone_phase,
    filter_queue_by_phase,
    scheduler_status_json,
)


def test_backbone_phase_pending(tmp_path: Path, fixture_config: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    assert backbone_phase(fixture_config, manifest_path=manifest) == "pending"


def test_backbone_phase_gpu_done(tmp_path: Path, fixture_config: Path) -> None:
    import pytest

    name = backbone_name_from_config(fixture_config)
    root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"backbone": name, "step": "7_train_arms"}) + "\n",
        encoding="utf-8",
    )
    arms = root / "experiments" / name / "arms" / "conventional" / "manifest.json"
    if not arms.is_file():
        pytest.skip("fixture arms manifest not on disk")
    assert backbone_phase(fixture_config, manifest_path=manifest) == "gpu_done"


def test_backbone_phase_done(tmp_path: Path, fixture_config: Path) -> None:
    name = backbone_name_from_config(fixture_config)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"backbone": name, "step": "12_record", "gate1": "pass"}) + "\n",
        encoding="utf-8",
    )
    assert backbone_phase(fixture_config, manifest_path=manifest) == "done"


def test_filter_queue_skips_done(tmp_path: Path, fixture_config: Path) -> None:
    name = backbone_name_from_config(fixture_config)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"backbone": name, "step": "12_record", "gate1": "pass"}) + "\n",
        encoding="utf-8",
    )
    parts = filter_queue_by_phase([fixture_config], manifest_path=manifest)
    assert parts["done"] == [str(fixture_config)]
    assert parts["pending"] == []


def test_scheduler_status_json(tmp_path: Path, fixture_config: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    status = scheduler_status_json([fixture_config], manifest_path=manifest)
    assert "queue" in status
    assert fixture_config.as_posix() in status["queue"][0] or str(fixture_config) in status["queue"]
