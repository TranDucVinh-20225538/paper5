"""
Monte Carlo power analysis for the Paper 5 confirmatory design.

Derived from the locked protocol; derives nothing new and changes nothing:
  N = 10 backbones                      (D-017)
  n = 30 observations per backbone      (D-034)
  Kendall tau, kappa_primary vs AUROC   (D-035)
  Holm-Bonferroni across the family     (protocol)
  Tier-1 outcome scored on the count S  (D-032)

Two design notes that make the simulation faithful rather than decorative:

1. Kendall tau is a RANK statistic, so any strictly monotone transformation of
   either marginal leaves it unchanged. Simulating from a Gaussian copula is
   therefore exact for tau even though the real marginals are nothing like
   normal — kappa is heavy-tailed and AUROC is bounded. The copula relation is
   rho = sin(pi*tau/2).

2. The interesting quantity is not per-test power. It is the probability that
   the study lands in the correct Tier-1 outcome, because that is what the
   taxonomy will be read off. A design can have decent per-test power and still
   misclassify the outcome most of the time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

ALPHA = 0.05
N_OBS = 30
N_BACKBONES = 10


def tau_to_rho(tau: float) -> float:
    """Gaussian-copula correlation giving a target Kendall tau."""
    return float(np.sin(np.pi * tau / 2.0))


def _sample_p_values(
    tau_true: float,
    n_tests: int,
    rng: np.random.Generator,
    *,
    n_obs: int = N_OBS,
    seed_cluster_sd: float = 0.0,
    n_seeds: int = 5,
) -> np.ndarray:
    """One p-value per test, from n_obs paired observations at the given true tau.

    seed_cluster_sd > 0 adds a shared per-seed offset to both variables. The 30
    observations are 5 seeds x 6 conditions and are NOT independent; Paper 4
    treated them as if they were. A shared offset inflates the apparent
    association, so ignoring the clustering makes the nominal test anticonservative.
    """
    rho = tau_to_rho(tau_true)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    out = np.empty(n_tests)

    for i in range(n_tests):
        xy = rng.multivariate_normal([0.0, 0.0], cov, size=n_obs)
        if seed_cluster_sd > 0:
            per_obs = np.repeat(
                rng.normal(0.0, seed_cluster_sd, size=n_seeds),
                n_obs // n_seeds,
            )[:n_obs]
            xy = xy + per_obs[:, None]
        out[i] = stats.kendalltau(xy[:, 0], xy[:, 1], method="asymptotic").pvalue
    return out


def holm_reject(p: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Holm-Bonferroni step-down. Returns a boolean mask in the input order."""
    m = p.size
    order = np.argsort(p)
    thresholds = alpha / (m - np.arange(m))
    passed = p[order] <= thresholds
    # step-down: stop at the first failure
    cut = np.argmin(passed) if not passed.all() else m
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:cut]] = True
    return rejected


def tier1(s: int, t: int = N_BACKBONES) -> str:
    """Tier-1 outcome from the count of significant backbones (D-032)."""
    if s >= t:
        return "A"
    if s >= int(np.ceil(0.6 * t)):
        return "B"
    if s >= 2:
        return "C"
    return "D"


@dataclass(frozen=True)
class PowerResult:
    tau_true: float
    n_true: int
    tests_per_backbone: int
    per_test_power: float
    outcome_correct: float
    outcome_dist: dict[str, float]


def simulate(
    tau_true: float,
    n_true: int,
    *,
    tests_per_backbone: int = 1,
    reps: int = 10_000,
    seed: int = 42,
    seed_cluster_sd: float = 0.0,
) -> PowerResult:
    """Power and Tier-1 classification accuracy at a given truth.

    n_true backbones carry a real association of size tau_true; the remaining
    (10 - n_true) are null. The whole family is corrected together, which is
    what Holm actually does — per-test power computed in isolation would
    overstate it.
    """
    rng = np.random.default_rng(seed)
    n_null = N_BACKBONES - n_true
    hits = 0
    counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
    truth = tier1(n_true)

    for _ in range(reps):
        p_true = _sample_p_values(
            tau_true, n_true * tests_per_backbone, rng, seed_cluster_sd=seed_cluster_sd
        )
        p_null = _sample_p_values(
            0.0, n_null * tests_per_backbone, rng, seed_cluster_sd=seed_cluster_sd
        )
        rej = holm_reject(np.concatenate([p_true, p_null]))

        # A backbone counts as significant if any of its tests survives Holm.
        per_bb = rej.reshape(N_BACKBONES, tests_per_backbone).any(axis=1)
        hits += int(per_bb[:n_true].sum())
        counts[tier1(int(per_bb.sum()))] += 1

    denom = reps * max(n_true, 1)
    return PowerResult(
        tau_true=tau_true,
        n_true=n_true,
        tests_per_backbone=tests_per_backbone,
        per_test_power=hits / denom,
        outcome_correct=counts[truth] / reps,
        outcome_dist={k: v / reps for k, v in counts.items()},
    )


def minimum_detectable_tau(
    n_true: int,
    *,
    target: float = 0.80,
    tests_per_backbone: int = 1,
    grid: np.ndarray | None = None,
    reps: int = 2_000,
    seed: int = 42,
) -> float | None:
    """Smallest tau on the grid reaching `target` per-test power. None if never."""
    grid = np.arange(0.20, 0.81, 0.05) if grid is None else grid
    for tau in grid:
        r = simulate(
            float(tau), n_true, tests_per_backbone=tests_per_backbone, reps=reps, seed=seed
        )
        if r.per_test_power >= target:
            return float(tau)
    return None
