"""Tests for vendored condition_number (Paper 4 replication quantity)."""

from __future__ import annotations

import numpy as np
import pytest

from src.estimators.mahalanobis import compute_mahalanobis_params_from_arrays
from src.geometry.condition_number import condition_number


def test_condition_number_on_diagonal_precision() -> None:
    eigvals = np.array([4.0, 2.0, 1.0, 0.5])
    precision = np.diag(eigvals)
    assert condition_number(precision) == pytest.approx(8.0)


def test_condition_number_matches_eigval_ratio() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8))
    sym = a @ a.T + np.eye(8) * 0.1
    precision = np.linalg.inv(sym)
    pos = np.linalg.eigvalsh(precision)
    pos = pos[pos > 0]
    expected = pos.max() / pos.min()
    assert condition_number(precision) == pytest.approx(expected)


def test_condition_number_import_path() -> None:
    from src.geometry import condition_number as cn  # noqa: F401

    assert callable(cn)


def test_condition_number_via_mahalanobis_pipeline() -> None:
    """κ is derived from Σ_W precision, not marginal covariance (REUSE.md §3)."""
    rng = np.random.default_rng(42)
    n_classes, d, per_class = 8, 16, 40
    features = np.concatenate(
        [
            rng.normal(loc=c, scale=1.0, size=(per_class, d))
            for c in range(n_classes)
        ]
    )
    labels = np.repeat(np.arange(n_classes), per_class)
    _means, precision = compute_mahalanobis_params_from_arrays(
        features, labels, num_classes=n_classes, reg_eps=1e-5
    )
    kappa = condition_number(precision)
    assert kappa >= 1.0
    assert np.isfinite(kappa)


def test_condition_number_too_few_eigenvalues() -> None:
    with pytest.raises(ValueError, match="fewer than 2 positive eigenvalues"):
        condition_number(np.diag([1.0, 0.0, 0.0]))
