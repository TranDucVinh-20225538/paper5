"""Geometry measurement suite — Step 10 (κ_paper4 and κ_primary; D-029 closed by D-035)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.estimators.mahalanobis import compute_mahalanobis_params_from_arrays
from src.geometry.condition_number import (
    SENSITIVITY_KS,
    condition_number,
    kappa_primary,
    unregularized_pooled_within_class_covariance,
)
from src.geometry.lid_spectral_decay import compute_lid_spectral_diagnostics
from src.intervention.arms import PARTIAL_FT_R, ArmCheckpoints
from src.intervention.embeddings import EmbeddingArtifacts
from src.intervention.training import apply_alpha, compute_delta_z, load_adapter_checkpoint
from src.utils.config import BackboneConfig

NUM_CLASSES = 8
REG_EPS = 1e-5


def score_adaptation_geometry(
    cfg: BackboneConfig,
    checkpoints: ArmCheckpoints,
    z_eval: np.ndarray,
    meta_eval: pd.DataFrame,
    nuisance_w: np.ndarray,
) -> list[dict[str, Any]]:
    """D-034 adaptation rungs: linear-probe (frozen) and partial-FT (r=8, α=1).

    The rungs are capacity points, not dose points: no alpha ladder, one row
    per seed, no "alpha" key. Geometry uses the same compute_geometry_metrics
    as the canonical rows so κ / LID / spectral slope are commensurable.

    full-adapter-FT is deliberately NOT computed. It reuses the conventional
    arm's checkpoints, and D-034 excludes it to avoid conflating the C4
    control with the C2 population — Paper 4's own stated reason.
    """
    adaptation_dir = checkpoints.arm_dir("adaptation")
    adaptation_manifest = adaptation_dir / "manifest.json"
    if not adaptation_manifest.is_file():
        return []

    rungs = json.loads(adaptation_manifest.read_text(encoding="utf-8"))["rungs"]
    adaptation_rows: list[dict[str, Any]] = []

    # linear-probe has no adapter: geometry is on the unmodified frozen embeddings.
    # The representation does not depend on seed, so compute once and copy.
    lp_rows = rungs.get("linear-probe", [])
    if lp_rows:
        lp_metrics = compute_geometry_metrics(z_eval, meta_eval, nuisance_w)
        for row in lp_rows:
            adaptation_rows.append(
                {"seed": int(row["seed"]), "rung": "linear-probe", **lp_metrics}
            )

    # partial-FT checkpoints are at a fixed rank, distinct from the r selected
    # in step 5, and are applied at full strength (no ladder).
    for row in rungs.get("partial-FT", []):
        seed = int(row["seed"])
        ckpt = adaptation_dir / row.get(
            "checkpoint_file", f"partialFT_adapter_seed{seed}.pt"
        )
        if not ckpt.is_file():
            continue
        adapter = load_adapter_checkpoint(ckpt, cfg, r=PARTIAL_FT_R)
        z_a = apply_alpha(z_eval, compute_delta_z(adapter, z_eval), 1.0)
        metrics = compute_geometry_metrics(z_a, meta_eval, nuisance_w)
        adaptation_rows.append({"seed": seed, "rung": "partial-FT", **metrics})

    return adaptation_rows


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
    out: dict[str, Any] = {
        "condition_number": float(condition_number(precision)),
        "per_direction_variance_w": float(np.var(proj)),
        "lid_mean": lid_diag.lid_mean,
        "spectral_decay_slope": lid_diag.spectral_slope,
    }
    out.update(kappa_primary_fields(z, labels, num_classes=num_classes))
    return out


def kappa_primary_fields(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    """κ_primary and preregistered k-sensitivity. Omits a k when d < k."""
    sigma_w = unregularized_pooled_within_class_covariance(z, labels, num_classes=num_classes)
    d = int(sigma_w.shape[0])
    names = {
        256: "condition_number_primary",
        128: "condition_number_primary_k128",
        512: "condition_number_primary_k512",
    }
    fields: dict[str, float] = {}
    for k in SENSITIVITY_KS:
        if d < k:
            continue
        fields[names[k]] = float(kappa_primary(sigma_w, k=k))
    return fields


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

    adaptation_rows = score_adaptation_geometry(
        cfg, checkpoints, z_eval, meta_eval, nuisance_w
    )
    if adaptation_rows:
        all_results["adaptation"] = adaptation_rows

    out_path = output_dir / "geometry_metrics.json"
    payload = {
        "alphas": alphas,
        "r": r,
        "reference_alpha0": ref,
        "results": all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
