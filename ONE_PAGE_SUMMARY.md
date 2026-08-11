# One Page Summary

<!-- The pre-commit hook rejects any edit to this file that does not also change `Amended-by:`.
     Every change to this file must be authorized by an entry in the program decision log:
     ../lab-notebook/decision_log.md -->

    Version:     1
    Amended-by:  D-009 (initial)
    Date:        2026-08-11
    Status:      PLAN — not yet a preregistration (see Stop conditions)

This file is the anchor. Every other document in this repository is downstream of it. If something
here changes, the change is a decision, and the decision goes in the log with an ID before the edit
is committed.

---

## Research question

After accounting for backbone identity as a grouping factor, does the causal relationship between
representation geometry (condition number) and distance-based reliability-estimator performance —
established for a single frozen dermatology foundation model in Paper 4 — hold across a deliberately
sampled set of pretrained representation families, each probed with Paper 4's full intervention
ladder rather than a single frozen snapshot?

**Not** the question: *"Is PanDerm's geometry–reliability link better than DINOv3's?"* The target is
whether the effect survives once backbone is modeled as a source of variance — not a ranking.

## Gap

Paper 4 established the condition-number → estimator-performance association causally, but on
**one** backbone. Its own Prediction 3 — *"once geometry is known, architecture adds no further
predictive power for distance-estimator failure"* — was dropped before execution for compute
reasons, not logical ones. Treating "architecture" as a unitary causal category from one CNN plus
one foundation model is unfalsifiable by design (Paper 4, Assumption 6): within-category variance
was never measured. Nothing in the literature closes this with a *causal* multi-backbone design;
existing large-scale multi-backbone work is correlational.

> **Open check, must close before the RQ is locked** — reread arXiv:2510.15202v3 specifically for
> whether its multi-backbone study contains anything causal rather than correlational. Paper 4 ran
> the equivalent check against its own primary source and caught a false "transfer" framing. Tracked
> as **D-008**.

## Hypothesis

The condition-number → AUROC association is a property of representation geometry, not of any
particular backbone. It should therefore reproduce, in the same direction, across representation
families once each backbone is put through the same intervention ladder.

**Collider hazard, designed around, not ignored** — representation geometry sits between backbone
and intervention strength (Paper 4, `15_Causal_Graph.md`). Conditioning on geometry can manufacture
spurious association. The analysis must not condition on geometry as a covariate in a way that opens
this path.

## Intervention

Paper 4's ladder, reapplied independently per backbone. Backbone stays **frozen** throughout.

- Adapter on the frozen output embedding: `z' = z + W2 · act(W1 · z)`, `W2` zero-initialized
  (identity at init). Activation **matched per backbone**, not hardcoded.
- Loss `L = L_task + λ_proj · L_orth(z + Δz, w)`, with `L_orth = cos(z', w)²`.
- Nuisance direction `w = unit(μ_ISIC − μ_PAD)` — closed-form, fixed, **never learned or
  adversarial**. Recomputed per backbone.
- Dose ladder by **post-hoc interpolation from one trained adapter**: `z'(α) = z + α·Δz`,
  `α ∈ {0, 0.25, 0.5, 0.75, 1.0}`.
- Control arms: **conventional** (`L_orth` structurally absent, not λ=0) and **adaptation**
  (3 capacity rungs, task-loss only).
- `r` and `λ_proj` set by a pre-committed grid, smallest value passing Gate 0 + Gate 1,
  **never tuned against outcome results**. Grid re-run fresh per backbone.

## Primary outcome

Kendall's τ (exact) between **condition number** and estimator AUROC, per backbone, with backbone as
a **fixed effect** and planned family-level contrasts (CNN / medical-SSL / medical-VLM / general-SSL
/ general-VLM). Estimators: Mahalanobis, cosine-to-centroid, k-NN (k=10), KDE. Holm–Bonferroni
across backbone × estimator × metric × arm.

## Secondary outcomes

- FPR@95TPR alongside AUROC.
- LID and within-class spectral-decay slope as secondary geometry covariates (neither was
  Holm-significant in Paper 4).
- k-NN at k=1/10/50 (Paper 4 Group B baseline).
- Mixed model with backbone as a random effect — fit anyway, reported as **explicitly exploratory**,
  variance component flagged low-confidence given group count.

## Stop conditions

Nothing below the line may be called "preregistered" until all four clear. Each is a live blocker,
not a caveat.

| # | Blocker | Decision |
|---|---|---|
| 1 | Compute / server access and multi-day job duration — confirmed, not inferred | **D-003** — open |
| 2 | medical-VLM family cell still n=1 (BiomedCLIP alone) | **D-004** — open |
| 3 | general-SSL family cell still n=1 (DINOv3 alone) | **D-005** — open |
| 4 | Outcome-taxonomy thresholds are drafts, not numbers | **D-007** — open |

Plus one that must resolve before the taxonomy can even be written:

| 5 | Backbone count is inconsistent across documents (7 vs 8) and will change again when blockers 2–3 close | **D-006** — open |

**Gate 1 failure on a backbone means "not testable" for that backbone. It is never counted as
falsification of the hypothesis.** This distinction is load-bearing for the outcome taxonomy.

## Success criteria

Preregistered outcome taxonomy — **draft criteria, pending D-006 and D-007.** The denominators below
are written against a backbone count that is not yet fixed; they are placeholders, and are marked as
such deliberately so they cannot be read as preregistered numbers later.

| Outcome | Draft criterion | Reading |
|---|---|---|
| A — full replication | Holm-significant, same direction, in **all N** backbones | Strongest result |
| B — majority replication | Significant, same direction, in **≥⌈N·4/7⌉** backbones, no family reversed | Still strong |
| C — family-conditional | Significant within one family (≥2/2 agreeing) but not another, split **directionally consistent by family**, not scattered | Most interesting; drives the Discussion |
| D — no broader replication | Significant in **≤1** backbone, i.e. not exceeding Paper 4 alone | Paper 4 becomes a boundary condition, not a general account |

All four outcomes are publishable. This follows Paper 4's own precedent that each of its
falsification outcomes was "scientifically informative and publishable." A single backbone breaking
the trend is a finding to be discussed, not a failure to be explained away.

---

## Fixed factors (carried over, not re-litigated)

- **Data**: ISIC 2019 (n=25,331, ID) + PAD-UFES-20 (n=2,298, acquisition-shifted). Split seed=42;
  train n=16,211, ID eval n=5,067, val n=4,053 (unused), OOD n=2,298, eval pool n=7,365.
  **Identical to Papers 1–4** — verify by checksum, do not regenerate.
- **Seeds**: `{42, 52, 62, 72, 82}` per backbone.
- **Primary geometry covariate**: condition number only (sole Holm-significant metric in Paper 4).
- **Excluded estimators**: Energy, ViM, relative-Mahalanobis — classifier-head dependent, out of
  scope for a geometry-only hypothesis.

## Vocabulary lock (from Paper 4)

*evaluate* / *test*, never *propose*. *account*, never *mechanism*, until mediation is formally
defined in Methods.
