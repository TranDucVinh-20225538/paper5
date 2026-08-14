# [Title — draft]

Beyond a Single Backbone: A Preregistered Multi-Backbone Test of a Geometry-Mediated
Account of Distance-Based Reliability Estimation in Dermatology

    Draft:      round 0
    Protocol:   frozen at a83c74c (tag v0.2-protocol-frozen)
    Decisions:  lab-notebook D-001 … D-052

> **Structural rule for this file.** Confirmatory and exploratory content are separated by file
> structure, not only by wording. §4 contains preregistered results and no interpretation. §5
> contains interpretation and is explicit at every point about which layer it is speaking from.
> Nothing exploratory is stated in the Abstract or in §4.

---

## Abstract

*[To write. Constraints, not prose:]*

- **Lead with geometry, not with statistics.** The preregistration is a geometry study; an abstract
  that opens on inference invites the reviewer question "where is the geometry?"
- The primary claim is the preregistered one, and is stated with its scope:
  *"Under a preregistered evaluation protocol, we found no confirmatory evidence that the geometric
  relationship reported in [Paper 4] generalizes across backbone families."*
  Not *"geometry does not generalize."*
- The exact replication on the original backbone is stated as a result, because it is what rules out
  a pipeline explanation for the null.
- The resolution limitation is stated in the abstract, not deferred to the Discussion:
  *"the combination of five clusters and Holm correction severely limits attainable family-wise
  significance."*

---

## 1. Introduction

*[To write.]*

Structure the gap so the reader reaches the same question the preregistration asks:

1. Distance-based reliability estimators under acquisition shift — the Paper 1–3 lineage, ending in
   the finding that apparent robustness may be shortcut-driven and that the collapse is not loss of
   information.
2. Paper 4: a preregistered causal test on one frozen foundation model. Condition number the only
   Holm-significant geometry metric.
3. The gap: one backbone. Treating "architecture" as a unitary causal category from a single
   instance is unfalsifiable by design; within-category variance was never measured. Existing
   large-scale multi-backbone work is correlational.
4. This study: the same intervention ladder, applied independently to ten backbones sampled to span
   five representation families, with the analysis preregistered before any cross-backbone
   association was computed.

---

## 2. Methods

Derived from `docs/protocol.md` and `docs/preregistration.md`, both frozen. This section should not
introduce anything absent from those two files.

### 2.1 Data

ISIC 2019 (n = 25,331, in-distribution) and PAD-UFES-20 (n = 2,298, acquisition-shifted). Split seed
42, identical to the preceding papers in the program and verified as such rather than assumed.

    train (fitting population)   16,211
    ID eval                       5,067
    OOD eval                      2,298
    eval pool                     7,365

Split assignments are pinned by a digest over `image_id`, `label_idx`, `domain` and `partition`,
sorted by `image_id` — deliberately excluding file paths, which are machine-local and not split
variables. Digest `7841f732…3289ce`, verified identical when computed from the working metadata and
from the frozen reference embeddings of the previous study.

### 2.2 Backbones

Ten backbones, sampled as five representation families with two instances each, chosen to measure
within-family variance rather than to represent architectures individually.

| Family | Instance 1 | dim | Instance 2 | dim |
|---|---|---|---|---|
| CNN, supervised | ResNet-50 | 2048 | EfficientNet-B3 | 1536 |
| Medical, image-only SSL | PanDerm | 1024 | UNI | 1024 |
| Medical, vision-language | BiomedCLIP | 768 | MONET | 1024 |
| General, image-only SSL | DINOv3 | 1024 | MoCo v3 | 768 |
| General, vision-language | OpenCLIP ViT-B/16 | 768 | SigLIP ViT-L/16 | 1024 |

MedSAM was included as a declared architecture-portability probe, outside the family contrast and
excluded from every denominator.

All embeddings are taken pre-projection. Every checkpoint is pinned to an immutable revision;
library versions are pinned and recorded per run.

### 2.3 Intervention

Backbones remain frozen. A bottleneck adapter acts on the output embedding,
`z' = z + W₂·act(W₁·z)`, `W₂` zero-initialised, activation matched to each backbone. Loss
`L = L_task + λ_proj·cos(z′, w)²`, with nuisance direction `w = unit(μ_ISIC − μ_PAD)` computed in
closed form per backbone and never learned. The dose ladder `z′(α) = z + α·Δz`,
`α ∈ {0, 0.25, 0.5, 0.75, 1.0}`, is post-hoc interpolation from a single trained adapter.

`r` and `λ_proj` are selected per backbone from a pre-committed grid — smallest value passing both
gates, using gate outcomes only and never outcome results.

### 2.4 Gates

**Gate 0** (implementation integrity) and **Gate 1** (manipulation check) are eligibility criteria,
scored before and independently of the association. A Gate 1 failure records a backbone as *not
testable*; it is never counted as falsification.

### 2.5 Analysis population

Per backbone, n = 30: the intervention arm at `α ∈ {0.25, 0.5, 0.75, 1.0}` × 5 seeds, plus the
adaptation arm's linear-probe and partial-FT rungs × 5 seeds each. The full-adapter-FT rung is
excluded, as it reuses the conventional arm's checkpoints.

*The α-ladder alone is not sufficient: on the canonical arm only (n = 20) the same association gives
τ = 0.242, p = 0.146.*

### 2.6 Confirmatory analysis

**Covariate.** `κ_primary = λ₁/λ_k` over the descending eigenvalues of the pooled within-class
covariance, unregularised, k = 256 fixed across backbones. `κ_paper4` — full-dimension, regularised,
from the precision matrix — is reported alongside for direct replication comparison only.

**Outcome.** Mahalanobis AUROC.

**Test.** Kendall's τ per backbone. Primary inference is a **seed-level permutation test**, permuting
whole seeds; the parametric τ is secondary, and where the two disagree the permutation result
governs. This was fixed before either was computed.

**Family.** `κ_primary × Mahalanobis × 10 backbones` = 10 tests, Holm–Bonferroni, α = 0.05
family-wise. Cosine-to-centroid, k-NN and KDE are secondary and lie outside the correction.

**Classification.** Two tiers. Tier 1 is defined on the count of Holm-significant same-direction
backbones alone; Tier 2 describes the family pattern. Results are reported as a pair.

---

## 3. [Reserved]

---

## 4. Results

*Preregistered content only. No interpretation appears in this section.*

### 4.1 Testability

All ten family members passed both gates. **T = 10**, all five family cells intact. The preregistered
testability condition (T ≥ 5 and ≥ 2 complete cells) was met. All ten joined the analysis population
at 30/30.

MedSAM, the declared probe, passed Gate 0 on all twelve grid configurations and failed Gate 1 on all
twelve. It is recorded *not testable* and excluded from every denominator.

### 4.2 Per-backbone association

`κ_primary` (k = 256) against Mahalanobis AUROC, n = 30 per backbone. Holm–Bonferroni applied to the
permutation p-values. AUROC range is reported beside every τ.

| Backbone | τ (perm) | p (perm) | τ (param) | p (param) | p (Holm) | Sig. | AUROC min–max |
|---|---:|---:|---:|---:|---:|:--:|---|
| ResNet-50 | 0.0306 | 0.8500 | 0.0306 | 0.8156 | 1.0000 | no | 0.7786–0.7800 |
| EfficientNet-B3 | −0.5153 | 0.3333 | −0.5153 | 0.0001 | 1.0000 | no | 0.8181–0.8213 |
| PanDerm | 0.6047 | 0.0583 | 0.6047 | <0.0001 | 0.5833 | no | 0.9983–0.9988 |
| UNI | −0.2988 | 0.8250 | −0.2988 | 0.0227 | 1.0000 | no | 0.9722–0.9734 |
| BiomedCLIP | −0.3271 | 0.4250 | −0.3271 | 0.0127 | 1.0000 | no | 0.9680–0.9766 |
| MONET | −0.1859 | 0.1417 | −0.1859 | 0.1565 | 1.0000 | no | 0.9926–0.9934 |
| DINOv3 | 0.3082 | 0.3083 | 0.3082 | 0.0188 | 1.0000 | no | 0.9819–0.9851 |
| MoCo v3 | −0.3788 | 0.2167 | −0.3788 | 0.0039 | 1.0000 | no | 0.9749–0.9752 |
| OpenCLIP | 0.0259 | 0.9083 | 0.0259 | 0.8436 | 1.0000 | no | 0.9781–0.9791 |
| SigLIP | −0.0447 | 0.8083 | −0.0447 | 0.7333 | 1.0000 | no | 0.9655–0.9668 |

PanDerm parametric p = 4.04 × 10⁻⁶.

### 4.3 Outcome

**S = 0. Tier 1 = D. Tier 2 = heterogeneous. Reported pair: Outcome D, heterogeneous.**

All five family cells were 0/2. Mean permutation τ was negative in four families; medical-SSL was
+0.1529.

### 4.4 Permutation and parametric inference disagreed on six of ten backbones

Same τ, different p, at unadjusted α = 0.05:

| Backbone | τ | p (perm) | p (param) |
|---|---:|---:|---:|
| EfficientNet-B3 | −0.5153 | 0.3333 | 0.0001 |
| PanDerm | 0.6047 | 0.0583 | <0.0001 |
| UNI | −0.2988 | 0.8250 | 0.0227 |
| BiomedCLIP | −0.3271 | 0.4250 | 0.0127 |
| DINOv3 | 0.3082 | 0.3083 | 0.0188 |
| MoCo v3 | −0.3788 | 0.2167 | 0.0039 |

Per the preregistration, the permutation result governs.

### 4.5 Replication of the original result

On the original backbone, using the original covariate definition and the same n = 30 population:

    Previous study : τ = +0.5576,  n = 30
    This study     : τ = +0.5576,  n = 30      difference = 0.0000

Under the seed-level permutation test the same τ gives p = 0.1333.

### 4.6 Sensitivity

The preregistered sensitivity analyses agree with the primary result. `κ_primary` at
k ∈ {128, 256, 512} gives S = 0 and Tier 1 = D at every k. `κ_paper4` gives S = 0 and Tier 1 = D.

### 4.7 Secondary scorers

Cosine-to-centroid, k-NN and KDE were computed in full and lie outside the confirmatory correction.
[Table — supplementary.]

---

## 5. Discussion

*Three layers, kept explicitly separate. Each states which layer it speaks from.*

### 5.1 Layer 1 — the preregistered result

Under a preregistered evaluation protocol, we found no confirmatory evidence that the geometric
relationship reported in the previous study generalizes across backbone families.

This is stated with two scope conditions that belong in the same breath as the claim:

**It is not evidence of absence.** Six of ten backbones showed |τ| ≥ 0.29, in both directions.

**The procedure's resolution was limited.** With five seeds the permutation distribution admits 120
arrangements, so p-values are quantised at 1/120 ≈ 0.0083, while Holm's first threshold at ten tests
is 0.005. The combination of five clusters and Holm correction severely limits attainable
family-wise significance. *[Reviewers will find this whether or not it is stated. Stating it is
caution; omitting it is a defect.]*

### 5.2 Layer 2 — replication

The previous result reproduced exactly on its original backbone: τ = +0.5576 against a published
τ = +0.5576, difference 0.0000, on the same population and the same covariate definition.

This is what forecloses the most common explanation for a null: the pipeline. The same code that
returned no confirmatory evidence across ten backbones reproduced the original finding to four
decimal places on the one backbone where a published value exists to check against.

Under the seed-level permutation test, that identical τ gives p = 0.1333. The effect size is
unchanged; the inference is not. The two studies differ in the unit of resampling, not in the data.
*[Wording: the previous inference is not robust to changing the resampling unit. Not that it was
wrong.]*

### 5.3 Layer 3 — boundary of validity

The two studies answer different questions. The first asked whether the phenomenon exists on one
backbone; this one asks whether it holds across many. A negative answer to the second does not
retract the first. What this study establishes is a boundary, not a refutation.

*[One descriptive sentence only, no mechanism:]* Exploratory analyses suggested that variability
attributable to seed and variability attributable to the intervention were distributed differently
across quantities, motivating future work on experimental design.

*[Do not write here: that variance decomposition proves anything; that seed dominates; that ICC
explains the result; that family slopes differ. Exploratory analyses were neither adjusted for
multiplicity nor preregistered, and in some families rested on two models — and were fitted under
the same independence assumption this paper argues is not robust. See D-052.]*

### 5.4 Limitations

- Five seeds; the confirmatory procedure's resolution follows directly from that.
- Two instances per family.
- The manipulation-check criterion is one-sided (inherited), while the protocol prose describes
  dose-dependence without direction. The discrepancy is reported rather than repaired: it was
  already known that one backbone fails one-sided and would pass two-sided, so amending it after the
  fact would have been outcome-contingent.
- Artifact redistribution is not uniform across backbones; several checkpoints are non-commercial and
  one is non-redistributable.

### 5.5 Future work

*[Deferred, not promised. Do not describe a specific next paper.]*

---

## 6. Data and code availability

Repository, frozen preregistration commit, decision log. Release policy is stratified by checkpoint
licence; artifacts derived from non-redistributable backbones are represented by checksums only.

---

## Appendix A — Deviations from the preregistration

*[None at time of writing. Any entry here must carry its decision-log ID.]*
