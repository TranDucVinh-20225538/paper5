"""Geometry measurement suite — Step 10 (κ_paper4 only; κ_primary blocked by D-029)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.estimators.mahalanobis import compute_mahalanobis_params_from_arrays
from src.geometry.condition_number import condition_number
from src.geometry.lid_spectral_decay import compute_lid_spectral_diagnostics
from src.intervention.arms import ArmCheckpoints
from src.intervention.embeddings import EmbeddingArtifacts
from src.intervention.training import apply_alpha, compute_delta_z, load_adapter_checkpoint
from src.utils.config import BackboneConfig

NUM_CLASSES = 8
REG_EPS = 1e-5


def compute_geometry_metrics(
    z: np.ndarray,
    meta_eval: pd.DataFrame,
    w: np.ndarray,
    *,
    num_classes: int = NUM_CLASSES,
    reg_eps: float = REG_EPS,
) -> dict[str, Any]:
    labels = meta_eval["label_idx"].to_numpy().astype(np.int64)
    _means, precision = compute_mahalanobis_params_from_arrays(
        z, labels, num_classes=num_classes, reg_eps=reg_eps
    )
    lid_diag = compute_lid_spectral_diagnostics(z, labels, num_classes=num_classes)
    proj = z @ w
    return {
        "condition_number": float(condition_number(precision)),
        "per_direction_variance_w": float(np.var(proj)),
        "lid_mean": lid_diag.lid_mean,
        "spectral_decay_slope": lid_diag.spectral_slope,
    }


def run_geometry_completion(
    cfg: BackboneConfig,
    artifacts: EmbeddingArtifacts,
    checkpoints: ArmCheckpoints,
    nuisance_w: np.ndarray,
    *,
    r: int,
    alphas: list[float],
    output_dir: Path,
) -> Path:
    z_eval = artifacts.eval_embeddings
    meta_eval = artifacts.eval_metadata
    ref = compute_geometry_metrics(z_eval, meta_eval, nuisance_w)
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
                metrics = compute_geometry_metrics(z_alpha, meta_eval, nuisance_w)
                arm_rows.append({"seed": seed, "alpha": alpha, **metrics})
        all_results[arm] = arm_rows

    out_path = output_dir / "geometry_metrics.json"
    payload = {
        "alphas": alphas,
        "r": r,
        "reference_alpha0": ref,
        "results": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
