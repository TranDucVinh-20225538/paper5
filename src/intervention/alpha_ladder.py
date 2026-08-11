"""α-ladder post-hoc interpolation — protocol Step 8."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.intervention.arms import ArmCheckpoints
from src.intervention.embeddings import EmbeddingArtifacts
from src.intervention.gates import (
    alpha0_baseline_geometry,
    compute_gate0,
    compute_gate1_measurement,
    gate1_selection_pass,
)
from src.intervention.training import apply_alpha, compute_delta_z, load_adapter_checkpoint
from src.utils.config import BackboneConfig


@dataclass(frozen=True)
class AlphaLadderResult:
    alphas: list[float]
    baseline: dict[str, float]
    results: dict[str, list[dict[str, Any]]]
    output_path: Path


def _alpha_ladder(cfg: BackboneConfig) -> list[float]:
    ladder = cfg.raw.get("intervention", {}).get("alpha_ladder")
    if not isinstance(ladder, list) or not ladder:
        raise ValueError(f"{cfg.name}: intervention.alpha_ladder must be a non-empty list")
    return [float(a) for a in ladder]


def run_alpha_ladder(
    cfg: BackboneConfig,
    artifacts: EmbeddingArtifacts,
    checkpoints: ArmCheckpoints,
    *,
    r: int,
    output_dir: Path,
) -> AlphaLadderResult:
    """Compute Gate 0 + Gate 1 measurement at each (arm, seed, alpha)."""
    alphas = _alpha_ladder(cfg)
    z_eval = artifacts.eval_embeddings
    meta_eval = artifacts.eval_metadata
    baseline = alpha0_baseline_geometry(z_eval, meta_eval)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict[str, Any]]] = {}
    for arm in ("canonical", "conventional"):
        arm_dir = checkpoints.arm_dir(arm)
        if not arm_dir.is_dir():
            continue
        manifest = json.loads((arm_dir / "manifest.json").read_text(encoding="utf-8"))
        arm_rows: list[dict[str, Any]] = []
        for row in manifest["per_seed"]:
            seed = int(row["seed"])
            adapter = load_adapter_checkpoint(arm_dir / f"adapter_seed{seed}.pt", cfg, r=r)
            dz = compute_delta_z(adapter, z_eval)
            for alpha in alphas:
                z_alpha = apply_alpha(z_eval, dz, alpha)
                gate0 = compute_gate0(z_alpha, meta_eval, cfg)
                gate1_meas = compute_gate1_measurement(z_alpha, meta_eval)
                g1 = gate1_selection_pass(gate1_meas, baseline) if alpha > 0 else False
                arm_rows.append(
                    {
                        "seed": seed,
                        "alpha": alpha,
                        "gate0": gate0,
                        "gate1_measurement": gate1_meas,
                        "gate1_pass": g1,
                    }
                )
        all_results[arm] = arm_rows

    out_path = output_dir / "alpha_ladder_results.json"
    payload = {
        "alphas": alphas,
        "r": r,
        "baseline_alpha0_reference": baseline,
        "results": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return AlphaLadderResult(
        alphas=alphas,
        baseline=baseline,
        results=all_results,
        output_path=out_path,
    )
