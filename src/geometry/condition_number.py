"""
Vendored from paper-3/scripts/geometry_diagnostics.py
Source commit: 5fedcb3870b1eb17d15c36f79cc11421e7067522
Do not modify without a decision-log entry and a PanDerm regression check (D-027).

REUSE.md §3: κ is computed on the precision matrix derived from pooled within-class
covariance Σ_W (via compute_mahalanobis_params_from_arrays), not the marginal covariance.

D-029 is open: do not implement κ_primary (top-k, scale-normalized) in this module.
"""

from __future__ import annotations

import numpy as np


def condition_number(precision: np.ndarray) -> float:
    """
    kappa(Sigma) = lambda_max(Sigma) / lambda_min(Sigma).

    Computed directly from the precision matrix (Sigma^-1), not by inverting
    back to Sigma: condition number is invariant under matrix inversion
    (kappa(Sigma) == kappa(Sigma^-1)), since inversion just reciprocates every
    eigenvalue, which flips which one is "max" and which is "min" without
    changing their ratio. This avoids a second, separately-conditioned
    inversion of an already-inverted, already-regularized matrix.
    """
    precision = np.asarray(precision, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(precision)
    eigvals = eigvals[eigvals > 0]
    if eigvals.size < 2:
        raise ValueError("Precision matrix has fewer than 2 positive eigenvalues.")
    return float(eigvals.max() / eigvals.min())
