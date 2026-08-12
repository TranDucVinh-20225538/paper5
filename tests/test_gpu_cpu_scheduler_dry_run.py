"""Dry-run tests for gpu_cpu_scheduler.sh (no GPU consumption)."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "scripts" / "gpu_cpu_scheduler.sh"
FIXTURE_CFG = ROOT / "tests" / "fixtures" / "configs" / "fixture_backbone.yaml"


@pytest.fixture(scope="module")
def scheduler_executable() -> Path:
    SCHEDULER.chmod(SCHEDULER.stat().st_mode | 0o111)
    return SCHEDULER


def test_scheduler_dry_run_two_backbones(scheduler_executable: Path, tmp_path: Path) -> None:
    """Simulate 2-backbone overlap without touching GPU."""
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "DRY_RUN_GPU_SEC": "1",
            "DRY_RUN_CPU_SEC": "1",
            "MAX_CPU": "2",
            "POLL_SEC": "1",
            "RESPECT_EXTERNAL_GPU": "0",
            "LOCK_DIR": str(tmp_path / "locks"),
            "STATE_FILE": str(tmp_path / "state.json"),
            "CSG_DATA_ROOT": env.get("CSG_DATA_ROOT", "/tmp"),
        }
    )
    t0 = time.monotonic()
    proc = subprocess.run(
        [str(scheduler_executable), str(FIXTURE_CFG), str(FIXTURE_CFG)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.monotonic() - t0
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert elapsed < 30, "dry-run should finish quickly"
    combined = proc.stdout + proc.stderr
    assert "scheduler done" in combined.lower()


def test_scheduler_dry_run_single_backbone(scheduler_executable: Path, tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "DRY_RUN_GPU_SEC": "1",
            "DRY_RUN_CPU_SEC": "1",
            "MAX_CPU": "1",
            "POLL_SEC": "1",
            "RESPECT_EXTERNAL_GPU": "0",
            "LOCK_DIR": str(tmp_path / "locks"),
            "STATE_FILE": str(tmp_path / "state2.json"),
            "CSG_DATA_ROOT": "/tmp",
        }
    )
    proc = subprocess.run(
        [str(scheduler_executable), str(FIXTURE_CFG)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
