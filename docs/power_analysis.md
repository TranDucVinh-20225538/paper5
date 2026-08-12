# Power Analysis

    Status:  COMPLETE — four decisions proposed (D-037…D-040), awaiting PI sign-off
    Derives: from the locked protocol. Changes nothing in it.
    Code:    src/statistics/power.py   ·   tests/test_power.py
    Date:    2026-08-12

Nothing inherited from Paper 4. Paper 4's Holm family was scorer × metric × arm; Paper 5's grows by
`× backbone`, which changes the correction denominator and therefore every number below.

---

## 1. Statistical design

| | |
|---|---|
| **Primary endpoint** | Kendall's τ between `κ_primary` and estimator AUROC, per backbone (D-035) |
| **Observations** | n = 30 per backbone — Intervention α∈{0.25,0.5,0.75,1.0}×5 seeds, plus Adaptation linear-probe ×5 and partial-FT ×5 (D-034) |
| **Backbones** | N = 10 family members; MedSAM excluded as probe (D-016, D-017) |
| **Correction** | Holm–Bonferroni, step-down |
| **α** | 0.05 family-wise |
| **Target power** | 80% |
| **Simulation** | 10,000 Monte Carlo repetitions (4,000 where noted) |

**Why a Gaussian copula is exact here.** Kendall's τ is a *rank* statistic, so any strictly monotone
transformation of either marginal leaves it unchanged. Simulating from a bivariate normal with
`ρ = sin(πτ/2)` therefore reproduces the τ distribution exactly, even though the real marginals are
nothing like normal — κ is heavy-tailed and AUROC is bounded above. No distributional assumption
about the actual data is being smuggled in.

---

## 2. The family of hypotheses — the single biggest power lever

The confirmatory family size is a **design choice**, not a fact, and it dominates everything else.

Two candidates, and a trap between them:

| Family | Tests | Composition |
|---|---|---|
| **10** | `κ_primary × Mahalanobis × 10 backbones` | one confirmatory scorer |
| **40** | `κ_primary × 4 scorers × 10 backbones` | all four scorers confirmatory |

**The trap.** A first simulation treated the four scorers as independent and made family=40 look
*more* powerful than family=10 — each backbone got four independent shots, and the union gain beat
the Holm penalty. That is wrong: the four scorers run on the **same embeddings, same data**, and
their AUROCs are strongly correlated. Independent replicates are four real chances; correlated ones
are close to one.

Bracketing it honestly:

| true τ | family=10 | family=40, independent *(optimistic, unrealistic)* | family=40, correlated *(honest bound)* |
|---|---|---|---|
| 0.30 | 0.368 | 0.534 | 0.169 |
| 0.35 | 0.596 | 0.798 | 0.331 |
| **0.40** | **0.817** | 0.958 | **0.554** |
| 0.45 | 0.942 | 0.998 | 0.787 |
| 0.50 | 0.985 | 1.000 | 0.931 |
| 0.55 | 0.998 | 1.000 | 0.983 |

At τ = 0.40, adding three correlated scorers to the confirmatory family costs **0.82 → 0.55 power**.
They quadruple the correction denominator while contributing almost no independent information.

**→ D-037: confirmatory family = 10.** Mahalanobis only. The other three scorers are preregistered
**secondary**, reported in full but outside the confirmatory correction.

Mahalanobis is also the right choice on lineage grounds — Papers 1–4 are built on it — not merely
the convenient one.

---

## 3. Power curve and minimum detectable effect

All 10 backbones truly affected, family = 10, 10,000 reps:

| true τ | 0.20 | 0.25 | 0.30 | 0.35 | **0.40** | 0.45 | 0.50 | 0.55 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|
| power | 0.096 | 0.197 | 0.367 | 0.598 | **0.818** | 0.943 | 0.986 | 0.998 | 1.000 |

> ### Minimum detectable τ at 80% power: **τ ≈ 0.40**
> (family = 40 with correlated scorers would need τ ≈ 0.45–0.50)

**Assumed effect size.** Paper 4 observed **τ = 0.5576** for Mahalanobis vs. condition number on
n=30 (`stage5_c2_correlation_v2.json`). The design is comfortably powered there — but that estimate
comes from the sample in which the effect was *discovered*, so winner's curse applies and the true
value is likely lower. The design retains 82% power down to τ = 0.40, which leaves reasonable margin;
it degrades sharply below τ = 0.35.

---

## 4. Operating characteristics of the outcome taxonomy

Power per test is not what gets read off. **The taxonomy is.** So: given a true state of the world,
how often does the study report the right Tier-1 outcome?

Family = 10, 6,000 reps:

| true τ | truth A (10/10) | truth B (6/10) | truth C (3/10) | truth D (1/10) |
|---|---|---|---|---|
| 0.40 | **0.31** | **0.18** | 0.74 | 0.97 |
| 0.50 | 0.89 | 0.74 | 0.98 | 0.96 |
| 0.5576 | 0.98 | 0.93 | 1.00 | 0.96 |

### Two findings that belong in the preregistration

**(a) Misclassification is directionally biased — the taxonomy only ever downgrades.**

At τ = 0.40, a true Outcome B is reported as **C 80% of the time**; a true A is reported as B 61% of
the time. Missing a true effect removes a backbone from `S`, which moves the study *down* a tier.
Nothing moves it up: gaining a spurious backbone under Holm at α = 0.05 is rare.

So the taxonomy is **conservative** — errors run toward under-claiming. That is the right direction
for integrity, and it has a precise consequence that must be stated in advance:

> A reported **A or B is trustworthy**. A reported **C is not evidence against** a stronger truth —
> at τ = 0.40 it is the most likely reading of a true B.

Without this preregistered, a reported C would be discussed as "partial replication" when it may
simply be an underpowered A.

**(b) Outcome D is reliably identified** — 96–97% at every τ examined. The "Paper 4 is a boundary
condition" conclusion, the one the design most needs to be trustworthy, is the best-identified
outcome in the taxonomy.

---

## 5. Two threats the simulation surfaced

### 5.1 The ceiling does not cost power — it costs meaning

Mahalanobis AUROC range across the study population:

| Arm | range |
|---|---|
| Intervention (α-ladder) | 0.0006 |
| Adaptation | 0.0002 |
| **pooled n=30** | **≈ 0.0005** |

Paper 4 nonetheless obtained τ = 0.5576, Holm-significant. That is **not a contradiction**: Kendall's
τ is scale-free. It measures whether κ predicts the *ordering* of AUROC, and is completely indifferent
to whether the spread between best and worst is 0.5 AUROC points or 0.0005.

An earlier reading of this in the project — that saturation would show up as a smaller true τ and
therefore as reduced power — was **wrong**. Saturation does not reduce τ and does not reduce power.
It reduces *interpretability*, which is a different and arguably worse problem: the study can report a
strong, significant, well-powered rank association whose practical magnitude is five ten-thousandths
of AUROC.

A reviewer will ask "so what?" and the paper needs an answer written before the result exists.

**→ D-038: τ is never reported without the AUROC range beside it.** Every τ in the manuscript carries
the observed min–max AUROC of the population it was computed on, so no reader can mistake a rank
association for a practically meaningful effect.

### 5.2 n = 30 is not 30 independent observations

The 30 observations are **5 seeds × 6 conditions**. Observations sharing a seed share an adapter, a
training trajectory and an initialization; they are not exchangeable with observations from other
seeds. Paper 4's analysis treated all 30 as independent.

Simulating a shared per-seed offset:

| seed-cluster SD | power at τ = 0.45 |
|---|---|
| 0.0 | 0.942 |
| 0.3 | 0.968 |
| 0.6 | 0.988 |

Power **rises** with clustering — which is the bad direction. A shared offset moves both variables
together, inflating the apparent association. Clustering therefore makes the nominal test
**anticonservative**: real Type I error exceeds the nominal α, and the inflation grows with the
strength of the seed effect.

This is a validity threat, not a power problem, and it is inherited from Paper 4 rather than
introduced here.

**→ D-039: preregistered sensitivity — a seed-level permutation test**, permuting whole seeds rather
than individual observations, reported alongside the parametric τ. If the two disagree, the
permutation result governs.

---

## 6. Conclusions

1. **Minimum detectable τ ≈ 0.40** at 80% power, with the confirmatory family at 10 tests.
2. **Confirmatory family = κ_primary × Mahalanobis × 10 backbones.** Adding three correlated scorers
   costs 0.82 → 0.55 power at τ = 0.40 and buys almost no independent information (D-037).
3. **Powered for the anticipated effect.** Paper 4's τ = 0.5576 gives ≥98% power and ≥93% correct
   Tier-1 classification; margin holds down to τ = 0.40, then degrades sharply.
4. **The taxonomy only downgrades.** A reported C is not evidence against a true B. Preregister this
   reading (D-040).
5. **Saturation is an interpretability problem, not a power problem** — report the AUROC range with
   every τ (D-038).
6. **Seed clustering makes the nominal test anticonservative** — add a seed-level permutation
   sensitivity (D-039).

## 7. Proposed decisions

| ID | Decision |
|---|---|
| D-037 | Confirmatory family = `κ_primary × Mahalanobis × 10 backbones` (10 tests). Cosine, k-NN and KDE are preregistered **secondary**, reported in full, outside the confirmatory correction |
| D-038 | Every reported τ carries the observed AUROC **range** of its population. τ is scale-free; significance on a rank statistic implies nothing about practical magnitude |
| D-039 | Preregistered sensitivity: **seed-level permutation test**. Where it disagrees with the parametric result, the permutation result governs |
| D-040 | Preregister that **Tier-1 misclassification is directionally downward**: a reported C is not evidence against a true B, and A/B are trustworthy when reported |

## 8. Reproducing

```bash
python3 -m pytest tests/test_power.py -q
python3 -c "from src.statistics.power import simulate; print(simulate(0.40, 10, reps=10000))"
```
