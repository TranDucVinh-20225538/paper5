"""Reliability scorers on adapted embeddings — protocol Step 11."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.estimators.mahalanobis import (
    compute_mahalanobis_params_from_arrays,
    mahalanobis_min_squared_distances,
)
from src.estimators.scorers import (
    K_VALUES,
    NUM_CLASSES,
    PRIMARY_K,
    REG_EPS,
    auroc_fpr95,
    compute_knn_scores,
    cosine_centroid_scores,
    density_kde_scores,
)
from src.intervention.arms import ArmCheckpoints
from src.intervention.embeddings import EmbeddingArtifacts
from src.intervention.training import apply_alpha, compute_delta_z, load_adapter_checkpoint
from src.utils.config import BackboneConfig

RELIABILITY_ALPHAS = (0.25, 0.5, 0.75, 1.0)


def evaluate_reliability_scorers(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_eval: np.ndarray,
    meta_eval: pd.DataFrame,
) -> dict[str, Any]:
    id_mask = (meta_eval["domain"] == "isic").to_numpy()
    ood_mask = (meta_eval["domain"] == "pad_ufes").to_numpy()
    z_id, z_ood = z_eval[id_mask], z_eval[ood_mask]

    means, precision = compute_mahalanobis_params_from_arrays(
        z_train, y_train, num_classes=NUM_CLASSES, reg_eps=REG_EPS
    )
    s_id = mahalanobis_min_squared_distances(z_id, means, precision)
    s_ood = mahalanobis_min_squared_distances(z_ood, means, precision)
    maha_auroc, maha_fpr95 = auroc_fpr95(s_id, s_ood)

    cos_auroc, cos_fpr95 = auroc_fpr95(
        cosine_centroid_scores(z_id, means),
        cosine_centroid_scores(z_ood, means),
    )
    knn = compute_knn_scores(z_train, z_id, z_ood, K_VALUES)
    knn_auroc, knn_fpr95 = auroc_fpr95(*knn[PRIMARY_K])
    kde_auroc, kde_fpr95 = auroc_fpr95(
        density_kde_scores(z_train, y_train, z_id),
        density_kde_scores(z_train, y_train, z_ood),
    )
    return {
        "maha_auroc": maha_auroc,
        "maha_fpr95": maha_fpr95,
        "cosine_auroc": cos_auroc,
        "cosine_fpr95": cos_fpr95,
        "knn_k10_auroc": knn_auroc,
        "knn_k10_fpr95": knn_fpr95,
        "kde_auroc": kde_auroc,
        "kde_fpr95": kde_fpr95,
    }


def run_reliability_ladder(
    cfg: BackboneConfig,
    artifacts: EmbeddingArtifacts,
    checkpoints: ArmCheckpoints,
    *,
    r: int,
    output_dir: Path,
    alphas: tuple[float, ...] = RELIABILITY_ALPHAS,
) -> Path:
    z_train_raw = artifacts.train_embeddings
    y_train = artifacts.train_metadata["label_idx"].to_numpy().astype(np.int64)
    z_eval_raw = artifacts.eval_embeddings
    meta_eval = artifacts.eval_metadata
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
            dz_train = compute_delta_z(adapter, z_train_raw)
            dz_eval = compute_delta_z(adapter, z_eval_raw)
            for alpha in alphas:
                z_train_a = apply_alpha(z_train_raw, dz_train, alpha)
                z_eval_a = apply_alpha(z_eval_raw, dz_eval, alpha)
                scores = evaluate_reliability_scorers(z_train_a, y_train, z_eval_a, meta_eval)
                arm_rows.append({"seed": seed, "alpha": alpha, **scores})
        all_results[arm] = arm_rows

    out_path = output_dir / "reliability_scorers.json"
    payload = {"alphas": list(alphas), "results": all_results}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
