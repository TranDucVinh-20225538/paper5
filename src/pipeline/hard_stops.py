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


def verify_split_checksum(repo_root: Path) -> None:
    """Verify dataset split manifest if the checksum file is present."""
    checksum_path = repo_root / "datasets" / "checksums" / "split_seed42.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"Split checksum missing: {checksum_path}. "
            "Verify Papers 1–4 split before running (protocol Step 0)."
        )
    # File presence is the gate for M2; content verification added when manifest lands.


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
