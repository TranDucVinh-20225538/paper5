"""Smoke tests for OOD scorers."""

from __future__ import annotations

import numpy as np

from src.estimators.scorers import auroc_fpr95, compute_knn_scores, cosine_centroid_scores


def test_scorers_smoke() -> None:
    rng = np.random.default_rng(0)
    z_train = rng.normal(size=(80, 8)).astype(np.float32)
    y_train = np.repeat(np.arange(8), 10).astype(np.int64)
    z_id = rng.normal(size=(20, 8)).astype(np.float32)
    z_ood = rng.normal(size=(15, 8)).astype(np.float32) + 2.0

    means = np.stack([z_train[y_train == c].mean(axis=0) for c in range(8)])
    cos_id = cosine_centroid_scores(z_id, means)
    cos_ood = cosine_centroid_scores(z_ood, means)
    auroc, fpr95 = auroc_fpr95(cos_id, cos_ood)
    assert 0.0 <= auroc <= 1.0
    assert 0.0 <= fpr95 <= 1.0

    knn = compute_knn_scores(z_train, z_id, z_ood, (1, 10))
    auroc_k, _ = auroc_fpr95(*knn[10])
    assert 0.0 <= auroc_k <= 1.0
