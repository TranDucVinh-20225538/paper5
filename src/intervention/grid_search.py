"""Pre-committed r × lambda_proj grid search — protocol Step 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.intervention.embeddings import EmbeddingArtifacts, build_training_population
from src.intervention.gates import (
    alpha0_baseline_geometry,
    compute_gate0,
    compute_gate1_measurement,
    gate1_selection_pass,
)
from src.intervention.training import apply_adapter, train_adapter
from src.utils.config import BackboneConfig


@dataclass(frozen=True)
class GridSelection:
    r: int
    lambda_proj: float
    gate0: dict[str, Any]
    gate1_measurement: dict[str, Any]
    gate1_pass: bool
    trials: list[dict[str, Any]]


def search_hyperparameters(
    cfg: BackboneConfig,
    artifacts: EmbeddingArtifacts,
    w: np.ndarray,
    *,
    train_seed: int = 42,
    epochs: int | None = None,
) -> GridSelection | None:
    """
    Smallest (r, lambda_proj) in pre-committed grid order passing Gate 0 and Gate 1 (EA-02).

    Grid order: ascending r, then ascending lambda_proj (Paper 4 stopping rule).
    """
    z_train, y_train = build_training_population(artifacts)
    z_eval = artifacts.eval_embeddings
    meta_eval = artifacts.eval_metadata
    baseline = alpha0_baseline_geometry(z_eval, meta_eval)

    intervention = cfg.raw.get("intervention", {})
    epoch_override = epochs if epochs is not None else int(intervention.get("epochs", 100))

    trials: list[dict[str, Any]] = []
    selected: GridSelection | None = None

    for r in cfg.grid_r:
        for lambda_proj in cfg.grid_lambda_proj:
            adapter, _head = train_adapter(
                z_train,
                y_train,
                w,
                cfg,
                r=r,
                lambda_proj=lambda_proj,
                seed=train_seed,
                epochs=epoch_override,
            )
            z_adapted = apply_adapter(adapter, z_eval)
            gate0 = compute_gate0(z_adapted, meta_eval, cfg)
            gate1_meas = compute_gate1_measurement(z_adapted, meta_eval)
            g1_pass = gate1_selection_pass(gate1_meas, baseline)
            both_pass = gate0["gate0_pass"] and g1_pass
            trial = {
                "r": r,
                "lambda_proj": lambda_proj,
                "gate0": gate0,
                "gate1_measurement": gate1_meas,
                "gate1_pass": g1_pass,
                "both_pass": both_pass,
            }
            trials.append(trial)
            if both_pass and selected is None:
                selected = GridSelection(
                    r=r,
                    lambda_proj=lambda_proj,
                    gate0=gate0,
                    gate1_measurement=gate1_meas,
                    gate1_pass=g1_pass,
                    trials=trials.copy(),
                )
                break
        if selected is not None:
            break

    if selected is None:
        return None
    return selected
