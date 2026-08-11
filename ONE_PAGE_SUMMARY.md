# One Page Summary

<!-- The pre-commit hook rejects any edit to this file that does not also change `Amended-by:`.
     Every change to this file must be authorized by an entry in the program decision log:
     ../lab-notebook/decision_log.md -->

    Version:     2
    Amended-by:  D-017 (backbone set fixed at N=10; D-014…D-024 from the backbone audit)
    Date:        2026-08-11
    Status:      PLAN — not yet a preregistration. One stop condition remains (D-007).

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

## Sampled backbones — N = 10, balanced 5 × 2 (D-017)

| Cell | Instance 1 | dim | Instance 2 | dim |
|---|---|---|---|---|
| CNN, supervised | ResNet-50 | 2048 | EfficientNet-B3 | 1536 |
| Medical, image-only SSL | PanDerm | 1024 | UNI | 1024 |
| Medical, vision-language | BiomedCLIP | 768 | MONET | 1024 |
| General, image-only SSL | DINOv3 | 1024 | MoCo v3 | 768 |
| General, vision-language | OpenCLIP ViT-B/16 | 768 | SigLIP ViT-L/16 | 1024 |

Plus **MedSAM** as a declared **architecture-portability probe**: full ladder, included in
per-backbone analysis, **excluded from the family-level model** (D-016). Dimensions are
pre-projection throughout (D-018).

Selection reasoning and every rejected candidate: [`docs/backbone_audit.md`](docs/backbone_audit.md).

## Primary outcome

Kendall's τ (exact) between **condition number** and estimator AUROC, per backbone, with backbone as
a **fixed effect** and planned family-level contrasts (CNN / medical-SSL / medical-VLM / general-SSL
/ general-VLM). Estimators: Mahalanobis, cosine-to-centroid, k-NN (k=10), KDE. Holm–Bonferroni
across backbone × estimator × metric × arm.

**Condition number is computed on a fixed number of principal components** (k below the smallest
embedding dimension in the set), not on the raw full-dimensional covariance (D-019). At fixed sample
size κ grows with dimension as arithmetic, and dimension tracks family across this set — so raw κ
would let the family contrast pick up dimension rather than geometry. Raw κ with `d` as a covariate
is retained as a preregistered sensitivity analysis. Paper 4 could not have detected this: with one
backbone, `d` was constant.

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
| 1 | Compute / server access and multi-day job duration | **D-003** — ✅ closed, confirmed by PI |
| 2 | medical-VLM family cell was n=1 | **D-014** — ✅ closed, MONET |
| 3 | general-SSL family cell was n=1 | **D-015** — ✅ closed, MoCo v3 |
| 4 | Backbone count inconsistent across documents (7 vs 8) | **D-017** — ✅ closed, **N = 10** |
| 5 | **Outcome-taxonomy thresholds are drafts, not numbers** | **D-007** — ⛔ **OPEN — the last one** |

D-007 is now unblocked (it was waiting on the denominator) but not resolved. **Until it closes,
nothing here is preregistered.** The power analysis is likewise unblocked and unwritten.

## Gates — three outcomes each, never two

**Gate 1 failure on a backbone means "not testable" for that backbone. It is never counted as
falsification of the hypothesis.** This distinction is load-bearing for the outcome taxonomy.

**Gate 0 carries the same three-way structure** (D-020). A frozen-feature adequacy check, **Gate
0-pre**, runs before the domain probe:

| Gate 0-pre | Gate 0 | Reading |
|---|---|---|
| pass | pass | proceed |
| pass | fail | **implementation broken** — fix it, not a finding |
| fail | — | **features inadequate for this task** — declared exclusion, neither a bug nor a falsification |

Without this, a backbone whose frozen features are genuinely too weak on dermoscopy either gets
reported as broken code, or invites someone to "fix" the implementation until the gate passes —
outcome-contingent tuning entering through the back door, in the one place the protocol exists to
prevent it.

## Success criteria

Outcome taxonomy — **N is now fixed at 10, but the thresholds themselves are still drafts pending
D-007.** The denominators below are no longer placeholders; the *criteria* still are, and are marked
so deliberately so they cannot be read as preregistered numbers later.

| Outcome | Draft criterion (N=10 family members) | Reading |
|---|---|---|
| A — full replication | Holm-significant, same direction, in **all 10** | Strongest result |
| B — majority replication | Significant, same direction, in **≥6** *(draft — the 4/7 rule rescaled; needs sign-off, not arithmetic)*, no family reversed | Still strong |
| C — family-conditional | Significant within one family (**2/2** agreeing) but not another, split **directionally consistent by family**, not scattered | Most interesting; drives the Discussion |
| D — no broader replication | Significant in **≤1**, i.e. not exceeding Paper 4 alone | Paper 4 becomes a boundary condition, not a general account |

MedSAM is scored and reported but **does not count toward any denominator above** — it is a
portability probe, not a family member (D-016). Fixing this in advance prevents a post-hoc argument
about whether to include it once its result is known.

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
