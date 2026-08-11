# Paper 5 — Kickoff

**Slot history.** The P5 slot previously held a chest X-ray / hospital-shift invariance account,
killed at Phase 0 (occupied by a published null result, unreliable manipulation check, confounded
shift construct). That attempt and the roadmap it belonged to (P5 invariance, P6 uncertainty, P7
explainability, P8 fairness, P9 synthesis) have been removed. This document reclaims P5 for a
different, unrelated question, chosen directly as a continuation of Paper 4 rather than as the next
item in that account taxonomy.

## Lineage

Paper 4 (`Paper4/20_One_Page_Summary.md`) originally carried three predictions. Prediction 3 —
*"once geometry is known, architecture adds no further predictive power for distance-estimator
failure"* — was deliberately dropped before execution
(`Paper4/21_Prediction3_Removed_Reevaluation.md`), for three stated reasons:

1. **Assumption 6** — treating "architecture" as a unitary causal category from one CNN + one
   foundation-model instance is unfalsifiable-by-design; within-category variance was never
   measured.
2. **Collider hazard** (`Paper4/15_Causal_Graph.md`) — representation geometry sits between
   architecture and intervention strength; conditioning on geometry can manufacture spurious
   association if not designed around.
3. **Resource ceiling** — P3 needed wide geometric-range sampling across multiple architectures,
   which the single-FM, 1-GPU/6-week budget for Paper 4 could not afford.

Paper 4's own re-evaluation explicitly flagged this as "out of scope for this paper, a natural
extension" — not abandoned, deferred. Paper 5 is that extension, not a new idea invented in chat.

**2026-08-11 revision.** The reason P3 was cut was reason 3 (compute), not reasons 1–2. If reason 3
no longer binds, cutting the design down to match old reason-3 constraints is the wrong move —
reasons 1–2 are the actual design requirements, and they call for *more* structure, not a lighter
protocol. Revised accordingly below. This also means Paper 5's target is not "Paper 4 × N
backbones" but the design reasons 1–2 always implied: a **multi-backbone causal replication study**,
where the object of inference is whether the Paper-4 causal effect holds *after accounting for
backbone identity*, not whether each backbone individually replicates it in isolation.

## Research question (revised)

> After accounting for backbone identity as a grouping factor, does the causal relationship between
> representation geometry (condition number) and distance-based reliability-estimator performance,
> established for a single frozen dermatology foundation model in Paper 4, hold across a
> deliberately sampled set of pretrained representation families — each probed with Paper 4's full
> intervention ladder rather than a single frozen snapshot?

Not: *"Is PanDerm's geometry–reliability link better than DINOv3's?"* The comparison target is
whether the effect survives once backbone is modeled as a source of variance, not a ranking across
backbones.

## Design (resolved 2026-08-11, pending one confirmation)

**Full intervention ladder on every sampled backbone.** Reapply Paper 4's post-hoc reshaping +
matched-fine-tuning-control ladder, manipulation check, and implementation-integrity gate
independently within each backbone. No backbone gets the cheaper frozen-snapshot-only treatment
(design B from the prior draft of this document is dropped as the primary route — see *Outcome
taxonomy* below for where correlational evidence still enters, as a secondary check, not a
substitute).

**One confirmation still needed before this is locked, not assumed:** this design multiplies Paper
4's per-backbone cost by the number of backbones (7 sampled families below), run as multi-day jobs.
The revision above is only valid if server access and multi-day runtime are actually available, not
inferred from "GPU is not the constraint" as a general statement. Please confirm before protocol
work starts — if it turns out partially constrained, the confirmatory/exploratory split from the
prior draft (full ladder on 2, lighter treatment on the rest) is the fallback, not a redesign from
scratch.

## Representation-family sampling strategy

Backbones are **not** chosen because they were on hand. The manuscript should say so explicitly —
draft sentence: *"We intentionally sampled pretrained representation families rather than
individual models, selecting at least two representatives per family whenever feasible, to
distinguish family-level consistency from model-specific behavior."*

| Family | Backbone(s) | Status |
|---|---|---|
| CNN | ResNet-50 (Papers 1–3 baseline) + **EfficientNet-B3** (already in the CSG-Skin/DST-Skin infra as the matched-backbone control) | 2 instances — resolved, reuses existing checkpoints |
| Medical, image-only SSL | PanDerm (Paper 4 anchor, rerun under the ladder here — not reused as-is, since Paper 4 only ran one point of the ladder's causal design on it in the FM setting) + MedSAM | 2 instances — resolved |
| Medical, vision-language | BiomedCLIP | **singleton — unresolved.** Candidates to evaluate for a second instance: PLIP, MedCLIP, or a Quilt-1M-pretrained CLIP variant. Not selected yet; do not proceed to protocol with this cell still at n=1 |
| General-purpose SSL | DINOv3 | **singleton — unresolved.** DINOv2 is the natural second instance (same lineage, well-documented, easy to source) but this is a suggestion, not a decision |
| General-purpose vision-language | SigLIP, OpenCLIP | 2 instances — resolved |

Two of five family cells are still n=1. This is the same Assumption-6 flaw Paper 4 named for its own
one-CNN-one-FM design, just at a smaller scale (2 cells instead of all of them). It should be closed
before the protocol is written, not treated as acceptable because most cells are fine.

## Statistical model

**Primary (confirmatory) analysis: backbone as a fixed effect with planned family-level contrasts**,
not a random effect. Reasoning: a mixed/hierarchical model with backbone as a random grouping factor
needs enough groups for the variance-component estimate to be stable — commonly cited guidance wants
something like 8–10+ groups at minimum for that estimate to behave well, and this design has 7
backbones total across 5 families. Fitting backbone as a random effect on n=7 groups will produce a
variance estimate too unstable to support the claim it's meant to test ("does backbone-level variance
exist"), which is exactly the question this design cares about most. Treating backbone as fixed,
with pre-specified contrasts at the family level (CNN vs. medical-SSL vs. medical-VLM vs.
general-SSL vs. general-VLM) gives a directly interpretable, adequately powered primary result.

**Secondary (exploratory): fit the mixed model anyway**, report it as descriptive, and flag its
variance-component estimate as low-confidence given group count — consistent with the
confidence-labeling convention already used in Paper 4's own predictions (`20_One_Page_Summary.md`).

This is a direct response to the mediation-adjacent framing in the chat proposal
(`Intervention → Backbone family → Reliability` alongside `Intervention → Condition number →
Reliability`): the family-level fixed-effect contrast *is* that comparison, run at adequate power;
the random-effect framing of the same question is not, at this sample size.

## Preregistered outcome taxonomy (draft thresholds — need Vinh's sign-off before this is a real
preregistration, not just a plan)

| Outcome | Draft criterion | Reading |
|---|---|---|
| A — full replication | Condition number–AUROC association Holm-significant, same direction, in all 7 backbones | Strongest possible result |
| B — majority replication | Significant, same direction, in ≥4/7 backbones, no family showing the reverse direction | Still strong |
| C — family-conditional | Significant within one family (≥2/2 backbones agreeing) but not another, with a *directionally consistent* split by family — not scattered | Most interesting outcome; motivates a Discussion about what distinguishes the families, not just "some backbones differ" |
| D — no broader replication | Significant in ≤1 backbone (i.e., not exceeding what Paper 4 already showed alone) | Paper 4 becomes a boundary condition, not a general account — still publishable per Paper 4's own falsification-criteria precedent, but this needs the actual number substituted for "≤1" before it's a real preregistered rule, not a placeholder |

A single backbone breaking the trend is a finding to be discussed, not a failure to be explained
away — carried over directly from the chat discussion, and consistent with Paper 4's own stance that
all three of its original falsification outcomes were "scientifically informative and publishable."

## Fixed factors (carried over, not re-litigated)

- Datasets: ISIC 2019 (in-distribution) vs. PAD-UFES-20 (acquisition-shifted) — unchanged from
  Papers 1–4.
- Estimators: Mahalanobis, cosine-to-centroid, k-NN, KDE — the four carried through Paper 4.
- Primary geometry covariate: condition number only (the sole Holm-significant metric in Paper 4).
  LID and within-class spectral decay retained as secondary/exploratory.
- Statistics: Holm–Bonferroni across backbones × estimators; seeds matching or exceeding Paper 4's
  five per backbone.

## Design risks to close before protocol lock

1. **Compute confirmed, not assumed** (see *Design* above) — the single largest open item.
2. **Two family cells still at n=1** (medical-VLM, general-SSL) — see sampling table.
3. **Intervention-recipe portability across architectures is not guaranteed.** Paper 4's post-hoc
   reshaping + fine-tuning-control ladder was designed against one ViT-based frozen backbone
   (PanDerm). Whether the same recipe is valid, without modification, on a segmentation-oriented
   encoder (MedSAM), contrastive vision-language encoders (BiomedCLIP, SigLIP, OpenCLIP), and a CNN
   (ResNet-50/EfficientNet-B3) is an implementation question that needs its own manipulation check
   per architecture family, not an assumption that one recipe transfers unmodified. This is a new
   risk the chat proposal's design surfaces that the prior draft of this document did not carry —
   worth a dedicated feasibility pass before committing to the full 7-backbone ladder.
4. **Outcome taxonomy thresholds are drafts, not preregistered numbers yet** — "≥4/7," "≤1," and the
   family-agreement rule for Outcome C all need explicit sign-off, or they will read as post-hoc
   flexible when the paper is written, which is exactly what preregistration is meant to prevent.

## One check before locking the RQ

Paper 4 verified its own premise against the primary source (arXiv:2510.15202v3) before writing the
one-sentence paper, and caught a false "transfer" framing in the process. The same check is owed
here: reread that source specifically for whether its "large-scale study across diverse
foundation-model backbones" already includes anything causal, not just correlational, before
Paper 5's RQ is locked as a genuine gap.

## Title candidates (draft — locked vocabulary from Paper 4 applies: *evaluate/test*, not
*propose*; *account*, not *mechanism*, until mediation is formally defined in Methods)

1. *Does Architecture Matter Once Geometry Is Known? A Multi-Backbone Causal Replication Across
   Representation Families*
2. *Beyond a Single Backbone: A Causal Replication of a Geometry-Mediated Account of
   Reliability-Estimator Behavior Across Pretrained Representation Families*
3. *Geometry Across Representation Families: A Multi-Backbone Causal Replication Study for
   Distance-Based Reliability Estimation in Dermatology*

## Immediate next steps, in order

1. Confirm compute/server access and acceptable job duration — everything below assumes this.
2. Close the two n=1 family cells (source second instances for medical-VLM and general-SSL).
3. Run one small architecture-portability check of the intervention recipe on a non-ViT, non-CNN
   backbone (e.g., MedSAM) before committing all 7 — cheap insurance against risk 3 above.
4. Sign off on the outcome-taxonomy thresholds as actual numbers, not drafts.
5. Only then write the full experimental design / preregistration document.
