# Preregistration — Paper 5

    Status:  FROZEN
    Frozen:  2026-08-14, before any cross-backbone association was computed
    Anchor:  ONE_PAGE_SUMMARY.md v4
    Log:     ../../lab-notebook/decision_log.md (D-001 … D-050)

This document contains **only what cannot change**. Everything here was decided before the
confirmatory analysis was run. Anything not fixed here is either already fixed in the decision log,
or is exploratory and must be labelled as such in the manuscript.

After the commit that adds this file, the permitted operations are: **run, compute statistics, draw
figures, export tables.** Not permitted: change a test, a correction, a threshold, a metric, the
taxonomy, or the split.

---

## 0. What has already been observed — read this first

A preregistration that overstates its own blindness is worth less than none. The following **have**
been seen before this freeze, and none of them is a Paper 5 outcome:

| Observed | Why it does not compromise the confirmatory test |
|---|---|
| **PanDerm τ = +0.5576, n=30** (D-027) | A regression check against Paper 4's **already-published** value. Its purpose was to prove the pipeline reproduces a known number before any new number was trusted. It is the replication anchor, not a result |
| **κ distributions at α=0, all 10 backbones** (D-049) | Used to test D-019's *stated rationale*, which it falsified (Spearman(d, κ) = −0.308, p = 0.386). The decision was unchanged; only its justification was corrected — before the outcome, deliberately |
| **Gate 0 / Gate 1 outcomes, all backbones** | Gates are eligibility criteria, not outcomes. Their whole design (D-020, D-033) rests on being scored before and independently of the association |
| **MedSAM: Gate 0 12/12, Gate 1 0/12** (D-047) | MedSAM is a probe outside the family contrast and counts toward no denominator (D-016, D-032) |

**Not seen, by anyone, at the time of this freeze:** the κ→AUROC association for any backbone other
than PanDerm; any family-level contrast; any Tier-1 or Tier-2 score; any permutation result.

---

## 1. Analysis population

**Split.** ISIC 2019 (n=25,331) + PAD-UFES-20 (n=2,298). Split seed 42, identical to Papers 1–4 and
verified as such, not assumed (D-050).

    Split fingerprint (D-050)
    7841f7324129881a9b648051fe110f14768f7708c37c602b7f378fe34a3289ce

    hashes image_id, label_idx, domain, partition — sorted by image_id, UTF-8 TAB/LF
    excludes path (machine-local) and isic_val (unused by the protocol)

Verified identical when computed from the live metadata and from Paper 4's frozen PanDerm
`reference_embeddings`. The pin rejects a flipped domain, a moved id, and a different synthetic
split.

    train (fitting population)  16,211
    ID eval                      5,067
    OOD eval (PAD-UFES-20)       2,298
    eval pool                    7,365

**Per-backbone analysis population — n = 30 (D-034), matching Paper 4 exactly:**

    INCLUDED   Intervention arm, α ∈ {0.25, 0.5, 0.75, 1.0} × 5 seeds   = 20
               Adaptation arm, linear-probe rung          × 5 seeds     =  5
               Adaptation arm, partial-FT rung            × 5 seeds     =  5

    EXCLUDED   Adaptation arm, full-adapter-FT rung
               (= Conventional data; excluded to avoid conflating the C4 control
                with the C2 population — Paper 4's own stated reason)

Not the α-ladder alone. On the canonical arm only (n=20) the same association gives τ = 0.242,
p = 0.146 — not significant. The effect depends on pooling across arms of differing capacity, and
that is a property of Paper 4's design, not a choice made here.

Both sides of the join — geometry and scorer — carry all 30 rows for all ten backbones.

---

## 2. Confirmatory hypothesis

> The condition-number → reliability-estimator association established in Paper 4 on one backbone
> holds across deliberately sampled representation families.

**Covariate.**

    κ_primary = λ₁ / λ_k
      over the descending eigenvalues of the pooled within-class covariance Σ_W,
      UNREGULARIZED, k = 256, fixed across every backbone.

`κ_paper4` (full-`d`, `Σ_W + 1e-5·I`, from the precision matrix) is reported alongside for direct
replication comparison only. It is **not** the confirmatory covariate.

*Justification, as corrected by D-049:* scale-and-regularisation commensurability. κ_primary
compresses the cross-backbone spread from **3918× to 5.2×**. The dimension rationale originally given
in D-019 was tested and withdrawn.

**Outcome.** Mahalanobis AUROC (class-conditional means, shared precision, ε = 1e-5, score = min
squared distance to any class centroid). ε is retained here unchanged, because the **scorer** must
match Paper 4 exactly even though the **covariate** is made commensurable (D-030).

**Test.** Kendall's τ, per backbone, on the n = 30 population.

**Units.** The ten family members (D-017):

| Cell | | |
|---|---|---|
| CNN, supervised | ResNet-50 | EfficientNet-B3 |
| Medical, image-only SSL | PanDerm | UNI |
| Medical, vision-language | BiomedCLIP | MONET |
| General, image-only SSL | DINOv3 | MoCo v3 |
| General, vision-language | OpenCLIP ViT-B/16 | SigLIP ViT-L/16 |

**MedSAM is excluded** from every denominator and every contrast. It is an architecture-portability
probe (D-016) and was recorded `not testable` under Step 9 (D-047).

---

## 3. Statistical analysis

**Primary inference is the seed-level permutation test (D-039).** The 30 observations are 5 seeds ×
6 conditions and are not exchangeable at the observation level: rows sharing a seed share an adapter,
a trajectory and an initialisation. Permutation is performed over **whole seeds**.

**The parametric τ is secondary.** Where the two disagree, **the permutation result governs.** This
is fixed here, before either has been computed, precisely so it cannot become a choice between a
friendlier and a less friendly number.

Simulation shows the direction of the risk: adding a shared per-seed offset *raises* apparent power
(0.942 → 0.988 at τ = 0.45), meaning observation-level inference is **anticonservative** — its real
Type I error exceeds nominal α.

**Family-level model.** Backbone as a **fixed effect** with planned contrasts across the five cells
(D-011). The mixed model with backbone as a random effect is fit and reported as **explicitly
exploratory**, its variance component flagged low-confidence.

**Collider hazard.** Representation geometry sits between backbone and intervention strength
(Paper 4, `15_Causal_Graph.md`). The analysis must not condition on geometry in a way that opens that
path. The specification is checked against the causal graph **before** fitting.

**Power.** Minimum detectable τ ≈ **0.40** at 80% power with the confirmatory family below
(`docs/power_analysis.md`). Paper 4's observed τ = 0.5576 gives ≥ 98% power.

---

## 4. Multiple testing

**Confirmatory family — 10 tests (D-037):**

    κ_primary  ×  Mahalanobis  ×  10 backbones

Holm–Bonferroni, step-down, **α = 0.05 family-wise**.

**Cosine-to-centroid, k-NN and KDE are secondary.** Reported in full, **outside** the confirmatory
correction, and never described as confirmatory.

*Reason, fixed in advance:* the four scorers run on the same embeddings and the same data, so they
are not four pieces of evidence. Simulated with realistic correlation, adding the other three costs
**0.82 → 0.55 power at τ = 0.40** while quadrupling the correction denominator.

**LID and within-class spectral-decay slope are secondary**, as in Paper 4, where neither was
Holm-significant.

**Sensitivity, preregistered:** `k ∈ {128, 256, 512}` for κ_primary. The reported conclusion must not
depend on which. Raw κ_paper4 with `d` as a covariate is also reported. If a sensitivity disagrees
with the primary, **that disagreement is itself a result** and is reported as such — it is not
resolved by preferring the friendlier one.

---

## 5. Classification of the outcome

Let `T` = backbones passing both Gate 0-pre and Gate 1. Let `S` = testable backbones with a
Holm-significant κ_primary → Mahalanobis AUROC association **in the same direction**.

**Testability gate first (D-033).** The hypothesis is tested only if `T ≥ 5` **and** at least two
complete family cells survive. Otherwise the study reports as a **portability/feasibility study**,
neither tier is scored, and no claim is made for or against the hypothesis. *At freeze time T = 10
and all five cells are intact.*

**Tier 1 — Outcome.** Defined on `S` alone, partitioning `{0…T}` with no gaps (D-032):

| | Criterion | At T = 10 |
|---|---|---|
| **A** full replication | `S = T` | 10 |
| **B** majority replication | `⌈0.6·T⌉ ≤ S < T` | 6–9 |
| **C** partial replication | `2 ≤ S < ⌈0.6·T⌉` | 2–5 |
| **D** no broader replication | `S ≤ 1` | 0–1 |

No family-pattern condition appears in Tier 1. That was the flaw in the earlier draft: mixing a count
rule with a pattern rule left outcomes satisfying none of A–D.

**Tier 2 — Interpretation**, applied to whichever outcome occurred:

| | Criterion |
|---|---|
| **consistent** | family contrast not significant **and** no family reversed |
| **family-specific** | ≥1 family at 2/2, ≥1 at 0/2, **and** contrast significant |
| **heterogeneous** | everything else — the complement, by construction |

**Reporting is always the pair** — *"Outcome C, heterogeneous"* — never a Tier-1 letter alone.

**Misclassification is one-directional (D-040).** Missing a true effect moves the study *down* a
tier; nothing moves it up. At τ = 0.40 a true B is reported as C 80% of the time. Therefore:

> **A reported C is not evidence against a true B.** A reported A or B is trustworthy. Outcome D is
> the best-identified cell (96–97% at every τ examined).

**All four outcomes are publishable.** D means Paper 4 is a boundary condition rather than a general
account — a finding, not a failure. A single backbone breaking the trend is discussed, not explained
away.

**Effect magnitude is always reported beside τ (D-038).** Kendall's τ is scale-free: Paper 4's
τ = 0.5576 sits on a Mahalanobis AUROC range of ≈ 0.0005. Every τ is accompanied by the observed
min–max AUROC of its population, so a rank association is never mistaken for a practically
meaningful effect.

---

## 6. Stopping rule — what is now immutable

From the commit that adds this file, none of the following may change:

- **No backbone is added or removed.** The set is the ten in §2 plus MedSAM as an excluded probe.
- **No metric is changed.** κ_primary as defined in §2; Mahalanobis AUROC as the confirmatory outcome.
- **α is not changed.** 0.05 family-wise.
- **κ is not redefined.** Not the k, not the matrix, not the regularisation.
- **The split is not changed.** Fingerprint in §1.
- **The confirmatory family is not changed.** Ten tests. Scorers are not promoted into it.
- **The taxonomy is not changed.** Neither tier, neither set of thresholds.
- **EA-02 is not changed.** Its one-sided criterion is inherited from Paper 4 and is known to be
  stricter than Step 9's two-sided prose. It is already known that MedSAM fails one-sided and would
  pass two-sided, so amending it now would be outcome-contingent. The discrepancy is **reported as a
  finding**, not repaired into a result.

**Deviations.** Any departure from the above is recorded as a decision-log entry with
`Outcome data seen at decision time: YES` and reported in the manuscript as a deviation. There is no
silent path.

**Permitted from here:** run, compute statistics, draw figures, export tables.

---

## Appendix — decisions this document freezes

| | |
|---|---|
| D-011 | Backbone as fixed effect; mixed model exploratory |
| D-016, D-047 | MedSAM as probe; recorded not testable |
| D-017 | N = 10, balanced 5 × 2 |
| D-030 | Scorer keeps ε; covariate made commensurable |
| D-032 | Two-tier taxonomy |
| D-033 | Testability gate |
| D-034 | Analysis population, n = 30 |
| D-035 | κ_primary: k = 256, unregularized Σ_W |
| D-037 | Confirmatory family = 10 tests |
| D-038 | AUROC range reported beside every τ |
| D-039 | Seed-level permutation governs |
| D-040 | Misclassification is downward only |
| D-049 | Dimension rationale withdrawn; decision stands |
| D-050 | Portable split pin |
