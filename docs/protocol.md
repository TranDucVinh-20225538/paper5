# Protocol — per-backbone runbook

    Status: DRAFT — executable order of operations, not yet locked.
    Locks when docs/preregistration.md is written.

This file is the **ordered procedure**. The *reasoning* behind each choice lives in
[`01_Implementation_Brief.md`](01_Implementation_Brief.md) and [`00_Kickoff.md`](00_Kickoff.md), and
is not repeated here — two documents explaining the same decision will drift, and then neither can be
trusted.

Run this loop independently per backbone. Nothing about one backbone's run informs another's, by
design: reusing a resolved value across backbones is the single most likely way to break the study.

---

## Step 0 — Hard stops before any compute

Refuse to start if any is true:

- [ ] `configs/<backbone>.yaml` has `representation.status: UNRESOLVED`
- [ ] `preprocessing.asset` missing, or `sha256` does not match the file on disk
- [ ] The dataset split checksum does not match Papers 1–4 (see `datasets/README.md`)
- [ ] D-003 (compute) is still open **and** this is not the MedSAM pilot

The pipeline should exit non-zero on each of these, not warn. A warning gets scrolled past.

## Step 1 — Freeze preprocessing

Write `assets/preprocessing/<backbone>.json` mirroring that backbone's **own published eval
transform**. Not PanDerm's. Hash it, record the hash in the config, cite the source.

Frozen means frozen: after this point a change to the file is a decision-log entry, not an edit.

## Step 2 — Resolve the representation

Decide and **write down** which tensor is the embedding, before extraction. Fill
`representation.{status, forward, pooling, rationale}` and log the decision.

The hard cases, and why they are hard:

| Backbone | Problem |
|---|---|
| MedSAM | Segmentation encoder — emits a spatial feature map, no pooled vector by default |
| BiomedCLIP / SigLIP / OpenCLIP | Dual encoders — image tower's pooled output; decide pre- vs post-projection. The projected space is trained for text alignment and may not be the geometry of interest |
| DINOv3 | CLS token by convention; confirm against the published linear-probe recipe |

## Step 3 — Extract embeddings

Train split (n=16,211, fitting population) and eval pool (n=7,365 = ID 5,067 + OOD 2,298).

Append to `results/manifest.jsonl`. Store arrays outside git — checksums only in the repo.

> PanDerm skips this step. Frozen embeddings already exist at
> `Paper4/PhaseB/assets/reference_embeddings/` and are reusable as-is.

## Step 4 — Nuisance direction

`w = unit(μ_ISIC − μ_PAD)` from domain-conditional means on the training population. Closed-form,
computed once, **never learned or adversarial**. Recompute per backbone — the direction only has
meaning inside its own embedding space.

## Step 5 — Grid search for `r` and `λ_proj`

Pre-committed grid `{16,32,64,128} × {0.1,1,5}`. Take the **smallest** value passing Gate 0 **and**
Gate 1.

**This selection never sees outcome results.** Only gate outcomes. Log the winner with
`Outcome data seen at decision time: NO` — that entry is the evidence for the claim the paper will
make.

Do not reuse PanDerm's `r=16, λ=0.1`. They won *this grid on that backbone*; that is not
transferable, and assuming otherwise silently converts a per-backbone result into an inherited one.

## Step 6 — Gate 0, implementation integrity

5-seed ℓ2-regularized logistic regression on standardized features:

- domain-probe accuracy > majority baseline + 0.05 (baseline 0.688 for this split)
- balanced accuracy > chance + 0.10

Confirm the majority baseline still holds for this eval pool. It should — the split is reused — but
confirm rather than assume.

**Fail ⇒ the implementation is broken.** Fix it. This is not a finding.

## Step 7 — Train the adapter

`z' = z + W2 · act(W1 · z)`, `W2` zero-init. `act` = **this backbone's** activation.
`L = L_task + λ_proj · cos(z', w)²`. AdamW, lr 1e-3, batch 512, 100 epochs, seeds
`{42,52,62,72,82}`.

Three arms: **canonical**; **conventional** (`L_orth` structurally absent — remove the term, do not
set λ=0); **adaptation** (3 capacity rungs, task-loss only).

## Step 8 — α-ladder

`z'(α) = z + α·Δz` for `α ∈ {0, 0.25, 0.5, 0.75, 1.0}`.

**Post-hoc interpolation from the one trained adapter.** Not five training runs. Retraining per α
would change what the ladder measures.

## Step 9 — Gate 1, manipulation check

At least one preregistered geometry metric (LID and/or spectral-decay slope) shows reproducible
dose-dependence across the ladder. **Monotonicity is not required.**

**Fail ⇒ NOT TESTABLE for this backbone. Record it as such and stop.** It is never counted as
falsification of the hypothesis, and it must not be reported in a way that lets a reader treat it as
a negative result. This distinction is load-bearing for the whole outcome taxonomy.

If a geometry-manipulation check is needed, use **spectral tempering**
`z' = μ + U·diag(λᵢ^(−γ/2))·Uᵀ(z−μ)`. Not the β-scaled ℓ2 knob — see `F-004`.

## Step 10 — Geometry metrics

Condition number (**primary** — the only Holm-significant metric in Paper 4). LID and within-class
spectral-decay slope as secondary.

## Step 11 — Reliability estimators

Fit on the **adapted** training population, evaluate on held-out ID + shifted pool.

Mahalanobis (class-conditional means, shared precision, ε=1e-5, score = min squared distance to any
class centroid) · cosine-to-centroid (max cosine to class centroid) · k-NN (k=10; also report
k=1/10/50) · KDE (class-conditional).

Report AUROC and FPR@95TPR.

Excluded, as in Paper 4: Energy, ViM, relative-Mahalanobis — classifier-head dependent, out of scope
for a geometry-only hypothesis.

## Step 12 — Record

Append to `results/manifest.jsonl`: commit SHA, config hash, backbone, arm, seed, α, gate outcomes,
output checksums.

---

## After all backbones

Cross-backbone analysis. **Do not run any of it before the per-backbone loop is complete** — partial
cross-backbone results are the most tempting way to let the outcome taxonomy drift.

1. Kendall's τ (exact), condition number vs. AUROC, per backbone.
2. Paired t-test + Wilcoxon signed-rank for arm comparisons.
3. Holm–Bonferroni across scorer × metric × arm × **backbone**.
4. **Primary**: backbone as fixed effect, planned family-level contrasts (CNN / medical-SSL /
   medical-VLM / general-SSL / general-VLM).
5. **Secondary, explicitly exploratory**: mixed model with backbone as random effect, variance
   component flagged low-confidence.
6. Score against the outcome taxonomy — using the signed-off numbers from D-007, not the drafts.

**Collider hazard.** Representation geometry sits between backbone and intervention strength
(Paper 4 `15_Causal_Graph.md`). Conditioning on geometry can manufacture spurious association. The
analysis must not condition on it in a way that opens that path — check the model specification
against the causal graph before fitting, not after.

## Pilot first

Run the **complete** ladder on MedSAM before committing to the rest. It is the structurally most
different encoder in the set, so it is the cheapest place to find out that the adapter, the nuisance
direction, or the gates do not port. Finding that out on backbone 7 instead of backbone 1 costs the
whole budget.
