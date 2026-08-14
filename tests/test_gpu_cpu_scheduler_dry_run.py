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


def _bash_supports_assoc_arrays() -> bool:
    """gpu_cpu_scheduler.sh uses `declare -A`, which needs bash 4+.

    macOS ships bash 3.2 (frozen in 2007 over GPLv3), so these tests cannot run on a
    stock Mac. They are skipped rather than left failing: a permanently red suite
    trains people to ignore failures, which is exactly how a real regression gets
    through. The scheduler itself is fine — the Linux server has bash 5.
    """
    out = subprocess.run(["bash", "-c", "declare -A _t 2>/dev/null && echo ok"],
                         capture_output=True, text=True)
    return out.stdout.strip() == "ok"


requires_bash4 = pytest.mark.skipif(
    not _bash_supports_assoc_arrays(),
    reason="gpu_cpu_scheduler.sh needs bash 4+ (declare -A); macOS ships bash 3.2",
)

@requires_bash4
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


@requires_bash4
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


@requires_bash4
def test_scheduler_continues_after_gpu_failure(scheduler_executable: Path, tmp_path: Path) -> None:
    """First backbone GPU fail must not block the second."""
    cfg_a = tmp_path / "dryrun_a.yaml"
    cfg_b = tmp_path / "dryrun_b.yaml"
    for path, name in ((cfg_a, "dryrun_a"), (cfg_b, "dryrun_b")):
        text = FIXTURE_CFG.read_text(encoding="utf-8").replace("fixture_test", name, 1)
        path.write_text(text, encoding="utf-8")

    env = os.environ.copy()
    failures = tmp_path / "failures.jsonl"
    env.update(
        {
            "DRY_RUN": "1",
            "DRY_RUN_GPU_SEC": "1",
            "DRY_RUN_CPU_SEC": "1",
            "DRY_RUN_FAIL_FIRST_GPU": "1",
            "MAX_CPU": "2",
            "POLL_SEC": "1",
            "RESPECT_EXTERNAL_GPU": "0",
            "LOCK_DIR": str(tmp_path / "locks"),
            "STATE_FILE": str(tmp_path / "state3.json"),
            "FAILURES_LOG": str(failures),
            "CSG_DATA_ROOT": "/tmp",
        }
    )
    proc = subprocess.run(
        [str(scheduler_executable), str(cfg_a), str(cfg_b)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1, proc.stderr + proc.stdout
    assert failures.is_file()
    body = failures.read_text(encoding="utf-8")
    assert "dryrun_a" in body
    combined = proc.stdout + proc.stderr
    assert "continuing queue" in combined.lower()
    assert "CPU phase OK dryrun_b" in combined
