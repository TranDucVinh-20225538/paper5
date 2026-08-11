"""Three-arm adapter training — protocol Step 7."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.intervention.embeddings import EmbeddingArtifacts, build_training_population
from src.intervention.gates import alpha0_baseline_geometry, compute_gate0, compute_gate1_measurement, gate1_selection_pass
from src.intervention.nuisance import NuisanceDirection
from src.intervention.training import (
    apply_adapter,
    save_adapter_checkpoint,
    train_adapter,
    train_adapter_task_only,
    train_conventional,
    train_linear_probe,
)
from src.utils.config import BackboneConfig

PARTIAL_FT_R = 8


@dataclass(frozen=True)
class ArmCheckpoints:
    backbone: str
    root: Path

    def arm_dir(self, arm: str) -> Path:
        return self.root / "arms" / arm


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enabled_arms(cfg: BackboneConfig) -> dict[str, bool]:
    arms = cfg.raw.get("arms", {})
    return {
        "canonical": bool(arms.get("canonical", True)),
        "conventional": bool(arms.get("conventional", True)),
        "adaptation": bool(arms.get("adaptation", True)),
    }


def train_canonical_arm(
    cfg: BackboneConfig,
    artifacts: EmbeddingArtifacts,
    nuisance: NuisanceDirection,
    *,
    r: int,
    lambda_proj: float,
    seeds: list[int],
    output_dir: Path,
    epochs: int | None = None,
) -> dict[str, Any]:
    z_train, y_train = build_training_population(artifacts)
    z_eval = artifacts.eval_embeddings
    meta_eval = artifacts.eval_metadata
    baseline = alpha0_baseline_geometry(z_eval, meta_eval)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        adapter, head = train_adapter(
            z_train,
            y_train,
            nuisance.w,
            cfg,
            r=r,
            lambda_proj=lambda_proj,
            seed=seed,
            epochs=epochs or 100,
        )
        z_adapted = apply_adapter(adapter, z_eval)
        gate0 = compute_gate0(z_adapted, meta_eval, cfg)
        gate1_meas = compute_gate1_measurement(z_adapted, meta_eval)
        g1 = gate1_selection_pass(gate1_meas, baseline)
        emb_path = output_dir / f"embeddings_seed{seed}.npy"
        ckpt_path = output_dir / f"adapter_seed{seed}.pt"
        np.save(emb_path, z_adapted.astype(np.float32))
        save_adapter_checkpoint(ckpt_path, adapter, head)
        per_seed.append(
            {
                "seed": seed,
                "gate0": gate0,
                "gate1_measurement": gate1_meas,
                "gate1_pass": g1,
                "embeddings_file": emb_path.name,
                "checkpoint_file": ckpt_path.name,
                "embeddings_sha256": _sha256_file(emb_path),
                "checkpoint_sha256": _sha256_file(ckpt_path),
            }
        )

    manifest = {
        "arm": "canonical",
        "hyperparameters": {"r": r, "lambda_proj": lambda_proj, "seeds": seeds},
        "per_seed": per_seed,
        "gate0_pass": all(r["gate0"]["gate0_pass"] for r in per_seed),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_conventional_arm(
    cfg: BackboneConfig,
    artifacts: EmbeddingArtifacts,
    *,
    r: int,
    seeds: list[int],
    output_dir: Path,
    epochs: int | None = None,
) -> dict[str, Any]:
    z_train, y_train = build_training_population(artifacts)
    z_eval = artifacts.eval_embeddings
    meta_eval = artifacts.eval_metadata
    baseline = alpha0_baseline_geometry(z_eval, meta_eval)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        adapter, head = train_conventional(
            z_train,
            y_train,
            cfg,
            r=r,
            seed=seed,
            epochs=epochs or 100,
        )
        z_adapted = apply_adapter(adapter, z_eval)
        gate0 = compute_gate0(z_adapted, meta_eval, cfg)
        gate1_meas = compute_gate1_measurement(z_adapted, meta_eval)
        g1 = gate1_selection_pass(gate1_meas, baseline)
        emb_path = output_dir / f"embeddings_seed{seed}.npy"
        ckpt_path = output_dir / f"adapter_seed{seed}.pt"
        np.save(emb_path, z_adapted.astype(np.float32))
        save_adapter_checkpoint(ckpt_path, adapter, head)
        per_seed.append(
            {
                "seed": seed,
                "gate0": gate0,
                "gate1_measurement": gate1_meas,
                "gate1_pass": g1,
                "embeddings_file": emb_path.name,
                "checkpoint_file": ckpt_path.name,
                "embeddings_sha256": _sha256_file(emb_path),
                "checkpoint_sha256": _sha256_file(ckpt_path),
            }
        )

    manifest = {
        "arm": "conventional",
        "hyperparameters": {"r": r, "seeds": seeds},
        "per_seed": per_seed,
        "gate0_pass": all(r["gate0"]["gate0_pass"] for r in per_seed),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_adaptation_arm(
    cfg: BackboneConfig,
    artifacts: EmbeddingArtifacts,
    nuisance: NuisanceDirection,
    *,
    seeds: list[int],
    output_dir: Path,
    conventional_manifest: dict[str, Any],
    epochs: int | None = None,
) -> dict[str, Any]:
    z_train, y_train = build_training_population(artifacts)
    z_eval = artifacts.eval_embeddings
    meta_eval = artifacts.eval_metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_override = epochs or 100

    linear_probe_rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_linear_probe(z_train, y_train, cfg, seed=seed, epochs=epoch_override)
        gate0 = compute_gate0(z_eval, meta_eval, cfg)
        linear_probe_rows.append({"seed": seed, "gate0": gate0})

    partial_rows: list[dict[str, Any]] = []
    for seed in seeds:
        adapter, head = train_adapter_task_only(
            z_train,
            y_train,
            cfg,
            r=PARTIAL_FT_R,
            seed=seed,
            epochs=epoch_override,
        )
        z_adapted = apply_adapter(adapter, z_eval)
        gate0 = compute_gate0(z_adapted, meta_eval, cfg)
        ckpt_path = output_dir / f"partialFT_adapter_seed{seed}.pt"
        save_adapter_checkpoint(ckpt_path, adapter, head)
        partial_rows.append({"seed": seed, "gate0": gate0, "checkpoint_file": ckpt_path.name})

    full_ft_rows = [
        {
            "seed": int(r["seed"]),
            "gate0": r["gate0"],
            "reused_from": "conventional",
        }
        for r in conventional_manifest["per_seed"]
    ]

    manifest = {
        "arm": "adaptation",
        "rungs": {
            "linear-probe": linear_probe_rows,
            "partial-FT": partial_rows,
            "full-adapter-FT": full_ft_rows,
        },
        "gate0_pass_linear_probe_and_partial_ft": all(
            r["gate0"]["gate0_pass"] for r in linear_probe_rows + partial_rows
        ),
        "nuisance_w_norm": nuisance.w_raw_norm,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_all_arms(
    cfg: BackboneConfig,
    artifacts: EmbeddingArtifacts,
    nuisance: NuisanceDirection,
    *,
    r: int,
    lambda_proj: float,
    exp_dir: Path,
    seeds: list[int] | None = None,
    epochs: int | None = None,
) -> ArmCheckpoints:
    seed_list = seeds or cfg.seeds
    enabled = _enabled_arms(cfg)
    checkpoints = ArmCheckpoints(backbone=cfg.name, root=exp_dir)

    conventional_manifest: dict[str, Any] | None = None
    if enabled["conventional"]:
        conventional_manifest = train_conventional_arm(
            cfg,
            artifacts,
            r=r,
            seeds=seed_list,
            output_dir=checkpoints.arm_dir("conventional"),
            epochs=epochs,
        )
    if enabled["canonical"]:
        train_canonical_arm(
            cfg,
            artifacts,
            nuisance,
            r=r,
            lambda_proj=lambda_proj,
            seeds=seed_list,
            output_dir=checkpoints.arm_dir("canonical"),
            epochs=epochs,
        )
    if enabled["adaptation"]:
        if conventional_manifest is None:
            raise RuntimeError(f"{cfg.name}: adaptation arm requires conventional arm checkpoints")
        train_adaptation_arm(
            cfg,
            artifacts,
            nuisance,
            seeds=seed_list,
            output_dir=checkpoints.arm_dir("adaptation"),
            conventional_manifest=conventional_manifest,
            epochs=epochs,
        )
    return checkpoints
