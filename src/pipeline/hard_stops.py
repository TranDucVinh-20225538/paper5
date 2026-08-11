"""Protocol Step 0 — hard stops before any compute."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.utils.config import BackboneConfig, validate_for_pipeline_run
from src.utils.preprocessing import verify_preprocessing_hash


def git_commit_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def verify_split_checksum(repo_root: Path, *, verify_live: bool = True) -> None:
    """
    Verify pinned split spec exists and optionally match live master_metadata.csv.

    When CSG_DATA_ROOT is configured, compares sha256 against the Papers 1–4 artifact.
    """
    from src.utils.paths import (
        load_dataset_paths,
        load_split_checksum_spec,
        verify_master_metadata_checksum,
    )

    spec_path = repo_root / "datasets" / "checksums" / "split_seed42.sha256"
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"Split checksum missing: {spec_path}. "
            "Verify Papers 1–4 split before running (protocol Step 0)."
        )
    spec = load_split_checksum_spec(repo_root)
    if "sha256" not in spec:
        raise ValueError(f"Invalid split spec (missing sha256): {spec_path}")

    if not verify_live:
        return

    try:
        paths = load_dataset_paths(repo_root=repo_root)
        verify_master_metadata_checksum(paths, repo_root=repo_root)
    except FileNotFoundError:
        # Dataset paths not configured — pinned spec in repo is sufficient for CI/fixture runs.
        pass


def run_step0_hard_stops(
    cfg: BackboneConfig,
    repo_root: Path,
    *,
    require_preprocessing_hash: bool = True,
    require_split_checksum: bool = True,
    require_checkpoint: bool = True,
) -> None:
    """
    Refuse to start if protocol Step 0 conditions hold.

    Raises ValueError / FileNotFoundError — caller must exit non-zero.
    """
    from src.utils.config import validate_for_pipeline_run

    validate_for_pipeline_run(cfg)

    if require_checkpoint and cfg.checkpoint is None:
        raise ValueError(
            f"{cfg.name}: backbone.checkpoint is null — pipeline refuses to run"
        )

    if not cfg.preprocessing_asset.is_file():
        raise FileNotFoundError(
            f"{cfg.name}: preprocessing asset missing: {cfg.preprocessing_asset}"
        )

    verify_preprocessing_hash(
        cfg.preprocessing_asset,
        cfg.preprocessing_sha256,
        require_hash=require_preprocessing_hash,
    )

    if require_split_checksum:
        verify_split_checksum(repo_root)
