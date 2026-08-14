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


def test_kappa_primary_is_lambda1_over_lambdak() -> None:
    from src.geometry.condition_number import kappa_primary

    eig = np.arange(10, 0, -1, dtype=np.float64)
    sigma = np.diag(eig)
    assert kappa_primary(sigma, k=4) == pytest.approx(10.0 / 7.0)


def test_kappa_primary_scale_invariant() -> None:
    from src.geometry.condition_number import (
        kappa_primary,
        unregularized_pooled_within_class_covariance,
    )

    rng = np.random.default_rng(0)
    n, d, k_cls = 400, 32, 8
    z = rng.normal(size=(n, d))
    labels = np.repeat(np.arange(k_cls), n // k_cls)
    s = unregularized_pooled_within_class_covariance(z, labels, num_classes=k_cls)
    assert kappa_primary(s, k=8) == pytest.approx(kappa_primary(4.0 * s, k=8))


def test_kappa_primary_uses_sigma_w_not_marginal() -> None:
    from src.geometry.condition_number import (
        kappa_primary,
        unregularized_pooled_within_class_covariance,
    )

    rng = np.random.default_rng(1)
    n_per, d, k_cls = 80, 32, 8
    parts = []
    labels = []
    for c in range(k_cls):
        parts.append(rng.normal(loc=20.0 * c, scale=1.0, size=(n_per, d)))
        labels.append(np.full(n_per, c))
    z = np.concatenate(parts)
    y = np.concatenate(labels)
    sigma_w = unregularized_pooled_within_class_covariance(z, y, num_classes=k_cls)
    sigma_m = np.cov(z, rowvar=False)
    k_w = kappa_primary(sigma_w, k=8)
    k_m = kappa_primary(sigma_m, k=8)
    assert k_m > 5.0 * k_w


def test_kappa_primary_has_no_epsilon() -> None:
    from src.geometry.condition_number import kappa_primary

    d = 32
    sigma = np.diag(np.linspace(1.0, 1e-8, d))
    unreg = kappa_primary(sigma, k=d)
    regularized = kappa_primary(sigma + 1e-5 * np.eye(d), k=d)
    assert unreg > 10.0 * regularized


def test_kappa_primary_rejects_k_beyond_rank() -> None:
    from src.geometry.condition_number import kappa_primary

    with pytest.raises(ValueError, match="cannot take"):
        kappa_primary(np.eye(4), k=8)


def test_condition_number_function_body_unchanged() -> None:
    import inspect

    from src.geometry.condition_number import condition_number

    src = inspect.getsource(condition_number)
    assert "reg_eps" not in src
    assert "kappa_primary" not in src
    assert "eigvals = eigvals[eigvals > 0]" in src
    assert "return float(eigvals.max() / eigvals.min())" in src
