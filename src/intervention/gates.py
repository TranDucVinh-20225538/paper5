"""Gate 0 (implementation integrity) and Gate 1 measurement for grid selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.geometry.lid_spectral_decay import compute_lid_spectral_diagnostics
from src.intervention.probes import ece_from_proba, linear_probe
from src.utils.config import BackboneConfig

DEFAULT_PROBE_SEEDS = (42, 52, 62, 72, 82)
MAJORITY_BASELINE_DOMAIN = 0.6879837067209776
CHANCE_BALANCED_ACC = 0.125
NUM_CLASSES = 8


def compute_gate0(
    z_adapted: np.ndarray,
    meta_eval: pd.DataFrame,
    cfg: BackboneConfig,
    *,
    probe_seeds: tuple[int, ...] = DEFAULT_PROBE_SEEDS,
) -> dict[str, Any]:
    gates = cfg.raw.get("gates", {}).get("gate0_implementation_integrity", {})
    majority = MAJORITY_BASELINE_DOMAIN
    probe_margin = float(gates.get("probe_acc_over_majority", 0.05))
    chance = CHANCE_BALANCED_ACC
    balanced_margin = float(gates.get("balanced_acc_over_chance", 0.10))

    domain_binary = (meta_eval["domain"] == "pad_ufes").to_numpy().astype(np.int64)
    probe_accs = []
    for seed in probe_seeds:
        result, _, _, _ = linear_probe(z_adapted, domain_binary, seed=seed)
        probe_accs.append(result["accuracy"])
    probe_acc_mean = float(np.mean(probe_accs))

    id_mask = (meta_eval["domain"] == "isic").to_numpy()
    y_id = meta_eval.loc[id_mask, "label_idx"].to_numpy().astype(np.int64)
    z_id = z_adapted[id_mask]
    bal_accs, eces = [], []
    for seed in probe_seeds:
        result, proba, y_test, _ = linear_probe(z_id, y_id, seed=seed)
        bal_accs.append(result["balanced_accuracy"])
        eces.append(ece_from_proba(proba, y_test))
    balanced_acc_mean = float(np.mean(bal_accs))

    criterion_1 = probe_acc_mean > majority + probe_margin
    criterion_2 = balanced_acc_mean > chance + balanced_margin
    return {
        "domain_probe_accuracy_mean": probe_acc_mean,
        "criterion_1_pass": bool(criterion_1),
        "id_task_balanced_accuracy_mean": balanced_acc_mean,
        "id_task_ece_mean": float(np.mean(eces)),
        "criterion_2_pass": bool(criterion_2),
        "gate0_pass": bool(criterion_1 and criterion_2),
    }


def compute_gate1_measurement(
    z_adapted: np.ndarray,
    meta_eval: pd.DataFrame,
    *,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    labels = meta_eval["label_idx"].to_numpy().astype(np.int64)
    diag = compute_lid_spectral_diagnostics(z_adapted, labels, num_classes=num_classes)
    return {
        "lid_mean": diag.lid_mean,
        "spectral_decay_slope": diag.spectral_slope,
        "composite": diag.composite,
        "n_samples": diag.n_samples,
    }


def gate1_selection_pass(measurement: dict[str, float], baseline: dict[str, float]) -> bool:
    """EA-02 criterion used during Step 5 grid search (Paper 4 stage4_select_lambda.py)."""
    lid_changed = measurement["lid_mean"] < baseline["lid_mean"]
    slope_changed = measurement["spectral_decay_slope"] < baseline["spectral_decay_slope"]
    return bool(lid_changed or slope_changed)


def alpha0_baseline_geometry(z_eval: np.ndarray, meta_eval: pd.DataFrame) -> dict[str, float]:
    """Baseline (alpha=0) LID/slope for grid Gate 1 comparison."""
    m = compute_gate1_measurement(z_eval, meta_eval)
    return {"lid_mean": m["lid_mean"], "spectral_decay_slope": m["spectral_decay_slope"]}


def _metric_reproducible_at_alpha(
    arm_rows: list[dict[str, Any]],
    alpha: float,
    metric_key: str,
) -> bool:
    """EA-03: all seeds deviate from alpha=0 baseline in the same direction."""
    baseline_rows = [r for r in arm_rows if r["alpha"] == 0.0]
    alpha_rows = [r for r in arm_rows if r["alpha"] == alpha]
    if not baseline_rows or not alpha_rows:
        return False
    baseline_val = float(baseline_rows[0]["gate1_measurement"][metric_key])
    deltas = [
        float(r["gate1_measurement"][metric_key]) - baseline_val for r in alpha_rows
    ]
    nonzero = [d for d in deltas if abs(d) > 1e-12]
    if not nonzero:
        return False
    first_sign = np.sign(nonzero[0])
    return all(np.sign(d) == first_sign for d in nonzero)


def score_gate1_ea03(
    ladder_results: dict[str, list[dict[str, Any]]],
    alphas: list[float],
) -> dict[str, list[dict[str, Any]]]:
    """Re-score alpha-ladder outputs under EA-03 manipulation check."""
    scored: dict[str, list[dict[str, Any]]] = {}
    for arm, rows in ladder_results.items():
        arm_scores: list[dict[str, Any]] = []
        for alpha in alphas:
            if alpha == 0.0:
                arm_scores.append(
                    {
                        "alpha": alpha,
                        "lid_reproducible": False,
                        "slope_reproducible": False,
                        "gate1_pass": False,
                    }
                )
                continue
            lid_rep = _metric_reproducible_at_alpha(rows, alpha, "lid_mean")
            slope_rep = _metric_reproducible_at_alpha(rows, alpha, "spectral_decay_slope")
            arm_scores.append(
                {
                    "alpha": alpha,
                    "lid_reproducible": lid_rep,
                    "slope_reproducible": slope_rep,
                    "gate1_pass": bool(lid_rep or slope_rep),
                }
            )
        scored[arm] = arm_scores
    return scored


def gate1_manipulation_pass(ea03_scores: dict[str, list[dict[str, Any]]]) -> bool:
    """Backbone passes EA-03 if canonical or conventional shows dose-dependence at some alpha > 0."""
    for arm in ("canonical", "conventional"):
        for row in ea03_scores.get(arm, []):
            if row["alpha"] > 0 and row["gate1_pass"]:
                return True
    return False
