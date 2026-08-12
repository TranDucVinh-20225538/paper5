# One Page Summary

<!-- The pre-commit hook rejects any edit to this file that does not also change `Amended-by:`.
     Every change to this file must be authorized by an entry in the program decision log:
     ../lab-notebook/decision_log.md -->

    Version:     3
    Amended-by:  D-032 (two-tier taxonomy; D-032…D-036 lock the protocol)
    Date:        2026-08-12
    Status:      PROTOCOL LOCKED — all stop conditions closed. Preregistration may now be written.

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

**Analysis population — n=30 per backbone (D-034), matching Paper 4 exactly:**

    INCLUDED  Intervention arm, α ∈ {0.25,0.5,0.75,1.0} × 5 seeds = 20
              Adaptation arm, linear-probe rung        × 5 seeds =  5
              Adaptation arm, partial-FT rung          × 5 seeds =  5
    EXCLUDED  Adaptation arm, full-adapter-FT rung (= Conventional data; Paper 4's own
              stated reason — avoid conflating the C4 control with the C2 population)

Not the α-ladder alone: on the canonical arm only (n=20) the same association gives τ=0.242, p=0.146
— not significant. The effect depends on pooling across arms of differing capacity.

**κ definitions (D-035):**

    κ_primary = λ₁/λ_k over Σ_W's descending eigenvalues, UNREGULARIZED, k=256 fixed
    κ_paper4  = Paper 4's: full-d, Σ_W + 1e-5·I, from the precision matrix

κ is computed on the **pooled within-class covariance `Σ_W`** (D-029), not the marginal covariance.
Sensitivity at `k ∈ {128,256,512}`; the conclusion must not depend on which.

*Why no normalization:* κ is scale-invariant — `κ(cΣ)=κ(Σ)` — so scale was never the confound. The
**absolute** ε was, since it does not rescale with the matrix. ε exists only to make Σ_W invertible
for the Mahalanobis scorer; κ_primary takes no inverse, so dropping ε removes the confound outright
rather than adjusting for it. Paper 4 could not have seen this: one backbone, one scale, constant `d`.

**Moderators are not equally answerable (D-036):** Family is **confirmatory**; dimension is a
**preregistered sensitivity** (3 of 5 cells carry a within-family 768/1024 contrast); pretraining
objective is **exploratory only and aliased** — `supervised` ≡ the CNN cell and `VL-contrastive` ≡ the
two VLM cells, leaving exactly one non-redundant contrast (DINOv3 vs MoCo v3) at n=2.

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
| 5 | Outcome-taxonomy thresholds were drafts, not numbers | **D-032** — ✅ closed, two-tier taxonomy |
| 6 | κ_primary definition (k, normalization) | **D-035** — ✅ closed, k=256, unregularized |
| 7 | Analysis population unspecified | **D-034** — ✅ closed, n=30 matching Paper 4 |

**All stop conditions are closed.** The power analysis is unblocked and remains to be written; it is
the last artifact before the preregistration itself.

## Testability gate — evaluated before any scoring (D-033)

Let `T` = backbones passing **both** Gate 0-pre and Gate 1. The hypothesis is tested **only if**:

1. `T ≥ 5`, **and**
2. at least **2 complete family cells** survive (both members testable).

Otherwise the hypothesis is **not tested**. The study reports as a **portability/feasibility study** —
which families admit the ladder at all, and why the others do not. Neither tier below is scored, and
no claim is made for or against the hypothesis.

This exists because Gate 1 failure means *not testable*, never falsification — so failures carry no
information about the hypothesis. A study where most units fail Gate 1 is inconclusive, not negative,
and the two must not be reported alike.

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

**Two tiers (D-032). Tier 1 counts; Tier 2 interprets.** Reporting is always the pair — *"Outcome C,
heterogeneous"* — never a Tier-1 letter alone.

Let `S` = testable backbones with a Holm-significant κ→AUROC association **in the same direction**.

**Tier 1 — Outcome.** Defined purely on `S`, so the four partition `{0,…,T}` with no gaps:

| Outcome | Criterion | At T=10 |
|---|---|---|
| **A** full replication | `S = T` | 10 |
| **B** majority replication | `⌈0.6·T⌉ ≤ S < T` | 6–9 |
| **C** partial replication | `2 ≤ S < ⌈0.6·T⌉` | 2–5 |
| **D** no broader replication | `S ≤ 1` | 0–1 |

No family-coherence condition appears here. That was the flaw in the draft taxonomy: mixing a count
rule with a pattern rule left outcomes satisfying none of A–D — 4 significant scattered across
families was not A, not B, not C, not D.

**Tier 2 — Interpretation**, applied to whichever outcome occurred:

| Interpretation | Criterion |
|---|---|
| **consistent** | family contrast **not** significant **and** no family reversed |
| **family-specific** | ≥1 family at 2/2, ≥1 family at 0/2, **and** contrast **is** significant |
| **heterogeneous** | everything else — the complement, by construction |

`heterogeneous` is deliberately residual. Defining it as a complement is what makes Tier 2 exhaustive
without inventing further labels.

**All outcomes are publishable.** D means Paper 4 is a boundary condition rather than a general
account — a finding, not a failure. A single backbone breaking the trend is discussed, not explained
away.

MedSAM is reported but **counts toward no denominator** — probe, not family member (D-016).

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
