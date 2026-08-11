"""Pipeline steps 0–12 (protocol): full per-backbone ladder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.backbone.extract import ExtractionOutput, extract_embeddings
from src.intervention.embeddings import build_training_population, load_embedding_artifacts
from src.intervention.grid_search import search_hyperparameters
from src.intervention.gates import compute_gate0
from src.intervention.nuisance import compute_nuisance_direction, save_nuisance_direction
from src.intervention.training import apply_adapter, train_adapter
from src.pipeline.hard_stops import git_commit_sha, run_step0_hard_stops
from src.utils.config import BackboneConfig, find_repo_root, load_backbone_config
from src.utils.manifest import append_manifest
from src.intervention.alpha_ladder import run_alpha_ladder
from src.intervention.arms import ArmCheckpoints, train_all_arms
from src.intervention.gates import gate1_manipulation_pass, score_gate1_ea03
from src.estimators.reliability import run_reliability_ladder
from src.geometry.metrics import run_geometry_completion


def config_sha256(cfg: BackboneConfig) -> str:
    payload = json.dumps(cfg.raw, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _experiment_dir(root: Path, backbone: str) -> Path:
    return root / "experiments" / backbone


def run_step3_extract(
    cfg: BackboneConfig,
    root: Path,
    *,
    output_dir: Path | None = None,
    loader_override: str | None = None,
    fixture_train_n: int = 16_211,
    fixture_eval_n: int = 7_365,
) -> ExtractionOutput:
    emb_dir = output_dir or (_experiment_dir(root, cfg.name) / "embeddings")
    return extract_embeddings(
        cfg,
        output_dir=emb_dir,
        loader=loader_override,
        train_n=fixture_train_n,
        eval_n=fixture_eval_n,
    )


def run_steps_4_through_6(
    cfg: BackboneConfig,
    extraction: ExtractionOutput,
    root: Path,
    *,
    manifest_path: Path | None = None,
    grid_epochs: int | None = None,
) -> dict[str, Any]:
    """Steps 4–6: nuisance direction, grid search, Gate 0 confirmation."""
    exp_dir = _experiment_dir(root, cfg.name)
    artifacts = load_embedding_artifacts(extraction.train_dir, extraction.eval_dir)

    # Step 4 — nuisance direction
    nuisance = compute_nuisance_direction(artifacts)
    nuisance_dir = exp_dir / "nuisance"
    nuisance_path = save_nuisance_direction(nuisance, nuisance_dir)

    # Step 5 — grid search (Gate 0 + Gate 1 EA-02 for selection)
    selection = search_hyperparameters(
        cfg,
        artifacts,
        nuisance.w,
        epochs=grid_epochs,
    )
    if selection is None:
        raise RuntimeError(
            f"{cfg.name}: no (r, lambda_proj) in pre-committed grid passed Gate 0 and Gate 1"
        )

    # Step 6 — Gate 0 on selected hyperparameters (protocol confirmation)
    z_train, y_train = build_training_population(artifacts)
    adapter, _head = train_adapter(
        z_train,
        y_train,
        nuisance.w,
        cfg,
        r=selection.r,
        lambda_proj=selection.lambda_proj,
        epochs=grid_epochs,
    )
    z_adapted = apply_adapter(adapter, artifacts.eval_embeddings)
    gate0_final = compute_gate0(z_adapted, artifacts.eval_metadata, cfg)
    if not gate0_final["gate0_pass"]:
        raise RuntimeError(f"{cfg.name}: Gate 0 failed after grid selection — implementation broken")

    manifest = manifest_path or (root / "results" / "manifest.jsonl")
    record = {
        "step": "6_gate0",
        "commit": git_commit_sha(root),
        "config_hash": config_sha256(cfg),
        "backbone": cfg.name,
        "nuisance_path": str(nuisance_path),
        "w_raw_norm": nuisance.w_raw_norm,
        "selected_r": selection.r,
        "selected_lambda_proj": selection.lambda_proj,
        "gate0": gate0_final,
        "grid_trials": len(selection.trials),
    }
    append_manifest(record, manifest)
    return record


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
) -> dict[str, Any]:
    cfg = load_backbone_config(config_path, repo_root=repo_root)
    root = repo_root or find_repo_root(Path(config_path))

    run_step0_hard_stops(
        cfg,
        root,
        require_preprocessing_hash=require_preprocessing_hash,
        require_split_checksum=require_split_checksum,
        require_checkpoint=require_checkpoint,
    )

    extraction = run_step3_extract(
        cfg,
        root,
        output_dir=output_dir,
        loader_override=loader_override,
        fixture_train_n=fixture_train_n,
        fixture_eval_n=fixture_eval_n,
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
        "train_dir": str(extraction.train_dir),
        "eval_dir": str(extraction.eval_dir),
        "train_sha256": extraction.train_sha256,
        "eval_sha256": extraction.eval_sha256,
        "skipped": extraction.skipped,
        "skip_reason": extraction.skip_reason,
    }
    append_manifest(record, manifest)
    return record


def run_steps_0_through_6(
    config_path: Path | str,
    *,
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
    require_preprocessing_hash: bool = True,
    require_split_checksum: bool = True,
    require_checkpoint: bool = True,
    loader_override: str | None = None,
    fixture_train_n: int = 16_211,
    fixture_eval_n: int = 7_365,
    grid_epochs: int | None = None,
) -> dict[str, Any]:
    cfg = load_backbone_config(config_path, repo_root=repo_root)
    root = repo_root or find_repo_root(Path(config_path))
    exp_emb_dir = _experiment_dir(root, cfg.name) / "embeddings"

    run_step0_hard_stops(
        cfg,
        root,
        require_preprocessing_hash=require_preprocessing_hash,
        require_split_checksum=require_split_checksum,
        require_checkpoint=require_checkpoint,
    )
    extraction = run_step3_extract(
        cfg,
        root,
        output_dir=exp_emb_dir,
        loader_override=loader_override,
        fixture_train_n=fixture_train_n,
        fixture_eval_n=fixture_eval_n,
    )
    manifest = manifest_path or (root / "results" / "manifest.jsonl")
    append_manifest(
        {
            "step": "3_extract_embeddings",
            "commit": git_commit_sha(root),
            "config_hash": config_sha256(cfg),
            "backbone": cfg.name,
            "train_dir": str(extraction.train_dir),
            "eval_dir": str(extraction.eval_dir),
        },
        manifest,
    )
    return run_steps_4_through_6(
        cfg,
        extraction,
        root,
        manifest_path=manifest,
        grid_epochs=grid_epochs,
    )


def run_steps_7_through_12(
    cfg: BackboneConfig,
    extraction: ExtractionOutput,
    root: Path,
    *,
    r: int,
    lambda_proj: float,
    manifest_path: Path | None = None,
    train_epochs: int | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Steps 7–12: arms, alpha ladder, Gate 1 EA-03, geometry, reliability, manifest."""
    exp_dir = _experiment_dir(root, cfg.name)
    ladder_dir = exp_dir / "alpha_ladder"
    artifacts = load_embedding_artifacts(extraction.train_dir, extraction.eval_dir)
    nuisance = compute_nuisance_direction(artifacts)
    nuisance_dir = exp_dir / "nuisance"
    save_nuisance_direction(nuisance, nuisance_dir)

    train_all_arms(
        cfg,
        artifacts,
        nuisance,
        r=r,
        lambda_proj=lambda_proj,
        exp_dir=exp_dir,
        seeds=seeds,
        epochs=train_epochs,
    )
    checkpoints = ArmCheckpoints(backbone=cfg.name, root=exp_dir)
    manifest = manifest_path or (root / "results" / "manifest.jsonl")
    append_manifest(
        {
            "step": "7_train_arms",
            "commit": git_commit_sha(root),
            "config_hash": config_sha256(cfg),
            "backbone": cfg.name,
            "r": r,
            "lambda_proj": lambda_proj,
            "seeds": seeds or cfg.seeds,
        },
        manifest,
    )

    ladder = run_alpha_ladder(cfg, artifacts, checkpoints, r=r, output_dir=ladder_dir)
    ea03 = score_gate1_ea03(ladder.results, ladder.alphas)
    ea03_path = ladder_dir / "gate1_ea03.json"
    ea03_path.write_text(
        json.dumps({"criterion": "EA-03", "alphas": ladder.alphas, "results": ea03}, indent=2),
        encoding="utf-8",
    )
    g1_pass = gate1_manipulation_pass(ea03)
    append_manifest(
        {
            "step": "9_gate1",
            "commit": git_commit_sha(root),
            "config_hash": config_sha256(cfg),
            "backbone": cfg.name,
            "gate1": "pass" if g1_pass else "not_testable",
            "ea03_path": str(ea03_path),
        },
        manifest,
    )
    if not g1_pass:
        return {
            "step": "9_gate1",
            "backbone": cfg.name,
            "gate1": "not_testable",
            "gate1_pass": False,
            "ea03": ea03,
        }

    geometry_path = run_geometry_completion(
        cfg,
        artifacts,
        checkpoints,
        nuisance.w,
        r=r,
        alphas=ladder.alphas,
        output_dir=ladder_dir,
    )
    append_manifest(
        {
            "step": "10_geometry",
            "commit": git_commit_sha(root),
            "config_hash": config_sha256(cfg),
            "backbone": cfg.name,
            "geometry_path": str(geometry_path),
        },
        manifest,
    )

    reliability_path = run_reliability_ladder(
        cfg,
        artifacts,
        checkpoints,
        r=r,
        output_dir=ladder_dir,
    )
    record = {
        "step": "12_record",
        "commit": git_commit_sha(root),
        "config_hash": config_sha256(cfg),
        "backbone": cfg.name,
        "gate1": "pass",
        "gate1_pass": True,
        "selected_r": r,
        "selected_lambda_proj": lambda_proj,
        "alpha_ladder_path": str(ladder.output_path),
        "geometry_path": str(geometry_path),
        "reliability_path": str(reliability_path),
        "ea03": ea03,
        "outcome_data_seen_at_decision_time": "NO",
    }
    append_manifest(record, manifest)
    return record


def run_steps_0_through_12(
    config_path: Path | str,
    *,
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
    require_preprocessing_hash: bool = True,
    require_split_checksum: bool = True,
    require_checkpoint: bool = True,
    loader_override: str | None = None,
    fixture_train_n: int = 16_211,
    fixture_eval_n: int = 7_365,
    grid_epochs: int | None = None,
    train_epochs: int | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    cfg = load_backbone_config(config_path, repo_root=repo_root)
    root = repo_root or find_repo_root(Path(config_path))
    exp_emb_dir = _experiment_dir(root, cfg.name) / "embeddings"

    run_step0_hard_stops(
        cfg,
        root,
        require_preprocessing_hash=require_preprocessing_hash,
        require_split_checksum=require_split_checksum,
        require_checkpoint=require_checkpoint,
    )
    extraction = run_step3_extract(
        cfg,
        root,
        output_dir=exp_emb_dir,
        loader_override=loader_override,
        fixture_train_n=fixture_train_n,
        fixture_eval_n=fixture_eval_n,
    )
    manifest = manifest_path or (root / "results" / "manifest.jsonl")
    append_manifest(
        {
            "step": "3_extract_embeddings",
            "commit": git_commit_sha(root),
            "config_hash": config_sha256(cfg),
            "backbone": cfg.name,
            "train_dir": str(extraction.train_dir),
            "eval_dir": str(extraction.eval_dir),
        },
        manifest,
    )
    step6 = run_steps_4_through_6(
        cfg,
        extraction,
        root,
        manifest_path=manifest,
        grid_epochs=grid_epochs,
    )
    return run_steps_7_through_12(
        cfg,
        extraction,
        root,
        r=int(step6["selected_r"]),
        lambda_proj=float(step6["selected_lambda_proj"]),
        manifest_path=manifest,
        train_epochs=train_epochs or grid_epochs,
        seeds=seeds,
    )
