"""Pipeline steps 0–3 (protocol): hard stops through embedding extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from src.backbone.extract import extract_embeddings
from src.pipeline.hard_stops import git_commit_sha, run_step0_hard_stops
from src.utils.config import BackboneConfig, find_repo_root, load_backbone_config
from src.utils.manifest import append_manifest


def config_sha256(cfg: BackboneConfig) -> str:
    payload = json.dumps(cfg.raw, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def run_steps_0_through_3(
    config_path: Path | str,
    *,
    repo_root: Path | None = None,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
    require_preprocessing_hash: bool = True,
    require_split_checksum: bool = True,
    require_checkpoint: bool = True,
    loader_override: str | None = None,
    fixture_train_n: int = 16_211,
    fixture_eval_n: int = 7_365,
) -> dict:
    """
    Execute protocol Steps 0–3 for one backbone config.

    Returns the manifest record appended in Step 3.
    """
    cfg = load_backbone_config(config_path, repo_root=repo_root)
    root = repo_root or find_repo_root(Path(config_path))

    run_step0_hard_stops(
        cfg,
        root,
        require_preprocessing_hash=require_preprocessing_hash,
        require_split_checksum=require_split_checksum,
        require_checkpoint=require_checkpoint,
    )

    # Step 1 — preprocessing already frozen; hash verified in Step 0 when required.
    # Step 2 — representation RESOLVED verified in Step 0.

    emb_dir = output_dir or (root / "experiments" / cfg.name / "embeddings")
    extraction = extract_embeddings(
        cfg,
        output_dir=emb_dir,
        loader=loader_override,
        train_n=fixture_train_n,
        eval_n=fixture_eval_n,
    )

    manifest = manifest_path or (root / "results" / "manifest.jsonl")
    record = {
        "step": "3_extract_embeddings",
        "commit": git_commit_sha(root),
        "config_hash": config_sha256(cfg),
        "backbone": cfg.name,
        "family": cfg.family,
        "loader": loader_override or cfg.raw["backbone"].get("loader"),
        "embed_dim": cfg.embed_dim,
        "pooling": cfg.pooling,
        "train_n": extraction.train_n,
        "eval_n": extraction.eval_n,
        "train_path": str(extraction.train_path),
        "eval_path": str(extraction.eval_path),
        "train_sha256": extraction.train_sha256,
        "eval_sha256": extraction.eval_sha256,
        "skipped": extraction.skipped,
        "skip_reason": extraction.skip_reason,
    }
    append_manifest(record, manifest)
    return record
