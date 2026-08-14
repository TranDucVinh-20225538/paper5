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
from src.intervention.training import apply_adapter, resolve_training_epochs, train_adapter
from src.utils.config import BackboneConfig


class GridSearchExhausted(RuntimeError):
    """No (r, lambda_proj) passed both gates. Carries every trial for diagnosis."""

    def __init__(self, backbone: str, trials: list[dict[str, Any]]):
        self.backbone = backbone
        self.trials = trials
        g0 = sum(1 for t in trials if t["gate0"]["gate0_pass"])
        g1 = sum(1 for t in trials if t["gate1_pass"])
        super().__init__(
            f"{backbone}: no (r, lambda_proj) in the pre-committed grid passed both gates "
            f"({len(trials)} trials: Gate 0 passed {g0}, Gate 1 passed {g1})"
        )


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
    epoch_override = resolve_training_epochs(cfg, epochs)

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
        # A failed grid is the most informative failure this study can produce -- under
        # D-020/D-033 "no configuration passed" is a declared outcome, not a bug. It
        # cannot be declared without knowing which gate failed and by how much, so the
        # trials are carried out on the exception rather than discarded with the return.
        raise GridSearchExhausted(cfg.name, trials)
    return selected
