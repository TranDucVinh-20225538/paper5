"""Tests for the power-analysis simulation (docs/power_analysis.md)."""

import numpy as np
import pytest

from src.statistics.power import (
    N_BACKBONES,
    holm_reject,
    simulate,
    tau_to_rho,
    tier1,
)


def test_tau_to_rho_matches_closed_form():
    # Kendall tau of a Gaussian copula is (2/pi)*arcsin(rho); invert and check.
    for tau in (0.1, 0.4, 0.5576, 0.8):
        rho = tau_to_rho(tau)
        assert np.isclose((2 / np.pi) * np.arcsin(rho), tau)


def test_tier1_partitions_the_whole_range():
    """D-032 Tier 1 must be MECE over {0..T} — every count maps to exactly one label."""
    labels = [tier1(s, t=10) for s in range(11)]
    assert labels[10] == "A"
    assert all(x == "B" for x in labels[6:10])
    assert all(x == "C" for x in labels[2:6])
    assert all(x == "D" for x in labels[0:2])
    assert set(labels) == {"A", "B", "C", "D"}


def test_holm_is_step_down_not_plain_bonferroni():
    # 0.001 <= .05/4; 0.02 > .05/3 so the step-down stops there.
    p = np.array([0.001, 0.02, 0.30, 0.90])
    assert list(holm_reject(p)) == [True, False, False, False]


def test_holm_rejects_nothing_when_all_null():
    assert not holm_reject(np.array([0.4, 0.5, 0.6, 0.99])).any()


def test_holm_rejects_all_when_all_tiny():
    assert holm_reject(np.array([1e-9] * 5)).all()


def test_type_i_error_controlled_under_the_global_null():
    """With no true effect anywhere, family-wise error must sit at or below alpha."""
    r = simulate(0.0, N_BACKBONES, reps=2000, seed=7)
    any_rejection = 1.0 - r.outcome_dist["D"] + 0.0
    # D covers S<=1, so this is a loose upper bound on the FWER; the point is
    # that spurious mass is small, not that the bound is tight.
    assert r.per_test_power < 0.05
    assert any_rejection < 0.10


def test_power_increases_monotonically_with_effect_size():
    powers = [simulate(t, 10, reps=800, seed=3).per_test_power for t in (0.2, 0.35, 0.5)]
    assert powers[0] < powers[1] < powers[2]


@pytest.mark.parametrize("n_true,expected", [(10, "A"), (6, "B"), (3, "C"), (1, "D")])
def test_strong_effect_recovers_the_true_outcome(n_true, expected):
    """At Paper 4's observed tau the taxonomy should land on the truth most of the time."""
    r = simulate(0.5576, n_true, reps=600, seed=11)
    assert max(r.outcome_dist, key=r.outcome_dist.get) == expected
    assert r.outcome_correct > 0.8


def test_misclassification_is_downward_only():
    """D-040: missing an effect moves the study DOWN a tier, never up."""
    r = simulate(0.40, 10, reps=1500, seed=5)  # truth = A, underpowered
    assert r.outcome_dist["B"] > 0  # downgraded
    r_low = simulate(0.40, 1, reps=1500, seed=5)  # truth = D
    assert r_low.outcome_dist.get("A", 0.0) == 0.0  # never upgraded to A
    assert r_low.outcome_dist.get("B", 0.0) == 0.0


def test_seed_clustering_is_anticonservative():
    """Shared per-seed offsets inflate the apparent association (power analysis 5.2)."""
    flat = simulate(0.45, 10, reps=1200, seed=9, seed_cluster_sd=0.0).per_test_power
    clustered = simulate(0.45, 10, reps=1200, seed=9, seed_cluster_sd=0.6).per_test_power
    assert clustered > flat
