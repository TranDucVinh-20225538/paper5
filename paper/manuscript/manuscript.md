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

Distance-based reliability estimators flag inputs that a deployed classifier should not be trusted
on, and their behaviour under acquisition shift has been linked to the geometry of the frozen
representation they operate on. A preregistered causal study established that link on a single
dermatology foundation model, where condition number was the only geometry metric significantly
associated with estimator performance. Whether that account describes representations in general or
that representation in particular cannot be determined from one backbone: with a single instance,
the two produce identical evidence.

We applied the same intervention ladder — dose ladder, matched control arms, and implementation and
manipulation gates — independently to ten frozen backbones, sampled as five representation families
with two instances each so that within-family variance could be measured rather than assumed. The
analysis was preregistered and frozen in a single commit before any cross-backbone association was
computed. Primary inference resampled whole random seeds rather than individual observations, since
observations sharing a seed share an adapter and a training trajectory.

We found no confirmatory evidence that the association generalizes: no backbone cleared the
Holm-corrected threshold, and the pattern across families was heterogeneous. Six of ten backbones
nonetheless showed |τ| ≥ 0.29, in both directions, so this is not evidence of absence. On the
original backbone the original result reproduced exactly (τ = +0.5576, difference 0.0000), which
removes the analysis pipeline as an explanation for the null. Evaluated under seed-level permutation,
that same τ gives p = 0.13; the two inference procedures, differing only in their unit of resampling,
disagreed on six of ten backbones.

With five seeds, the permutation distribution and the Holm correction together severely limit
attainable family-wise significance, and the null should be read with that constraint. We report a
boundary of validity rather than a refutation, and note that the earlier inference is not robust to
changing the unit of resampling.

**Keywords:** reliability estimation · distribution shift · representation geometry · preregistration
· replication · dermoscopy

---

## 1. Introduction

A skin-lesion classifier deployed outside the setting it was trained in will meet images it should
not be trusted on: a different camera, a different clinic, a different population. The practical
safeguard is not a better classifier but a *reliability estimator* — a mechanism that flags such
inputs before a prediction is acted upon. Distance-based estimators are the usual choice, because
they operate on a frozen representation, require no retraining, and depend on no classifier head.
Mahalanobis distance to class-conditional centroids is the canonical instance.

Whether these estimators actually survive acquisition shift has proved harder to settle than early
results suggested, and the difficulty is instructive. On dermoscopy-to-smartphone shift, Mahalanobis
distance separated in-distribution from shifted inputs at AUROC 0.97 while softmax confidence
collapsed [1]. That robustness did not survive scrutiny: after domain-adversarial shortcut removal,
the same estimator on the same shift fell to approximately 0.40 [2] — worse than chance, and
therefore not merely a loss of discriminative power. The natural reading is that the original
robustness had been reading a shortcut rather than the shift itself.

The reading that followed was less natural and more interesting. If the estimator fails because the
domain information has been removed, a domain probe applied to the same embeddings should also fail.
It does not. A linear probe still decoded domain at 0.72–0.81 AUROC from embeddings on which eight
distance and density estimators performed at or below chance [3]. The information is present. The
estimators cannot reach it.

That is a statement about the *geometry* of the representation rather than its information content,
and it converts a negative result into a mechanistic question: which property of an embedding space
determines whether a distance-based estimator can exploit the structure it contains?

A preregistered causal study addressed that question on a single frozen dermatology foundation model
[4]. Rather than comparing representations observationally, it intervened: a small adapter on the
frozen output embedding, trained to suppress a closed-form nuisance direction, with a dose ladder
obtained by post-hoc interpolation and matched control arms that isolate capacity from the
orthogonality objective. Among the preregistered geometry metrics, condition number was the only one
Holm-significantly associated with estimator performance; local intrinsic dimensionality and
within-class spectral decay were not.

That study was explicit about what it could not establish. Its own analysis flagged that treating
"architecture" as a causal category on the basis of a single instance is unfalsifiable by design:
with one backbone, a property of *representations* and a property of *that representation* produce
identical evidence, and within-category variance is never measured. It deferred the multi-backbone
test to future work for resource reasons rather than logical ones.

Filling that gap is not simply a matter of running more models. Large-scale multi-backbone studies of
representation quality exist, but they are correlational: they compare backbones as found, so any
association between geometry and estimator behaviour is confounded by everything else that differs
between them. What the question requires is the *same* intervention applied *independently* within
each backbone, so that the causal claim is tested once per representation and then compared across
representations — with backbone identity modelled as a source of variance rather than assumed away.

This study does that. We apply the full intervention ladder — dose ladder, control arms, implementation
and manipulation gates — independently to ten backbones, sampled as five representation families with
two instances each so that within-family variance is measurable rather than assumed. The families span
supervised CNNs, medical and general self-supervised encoders, and medical and general
vision–language models. A segmentation encoder was included separately as a declared
architecture-portability probe.

The analysis was preregistered and frozen before any cross-backbone association was computed. The
population, the covariate definition, the confirmatory family, the inference procedure, the outcome
taxonomy and the stopping rule were each fixed in advance, and the freeze is a single commit whose
timestamp precedes every result reported here. Two features of that protocol matter for reading what
follows: the primary inference resamples whole random seeds rather than individual observations,
because observations sharing a seed share an adapter and a training trajectory; and a backbone whose
manipulation check fails is recorded as *not testable* rather than as evidence against the
hypothesis.

We report three things. Under the preregistered protocol we found no confirmatory evidence that the
geometric relationship generalizes across backbone families. On the original backbone, the original
result reproduced exactly — to four decimal places — which removes the pipeline as an explanation for
the first finding. And the two inference procedures, differing only in their unit of resampling,
disagreed on six of ten backbones, including on the exact effect size the original study reported.

*[Citations: 1 = Paper 1, 2 = CSG-Skin, 3 = DST-Skin, 4 = Paper 4. Replace with final refs.]*

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
relationship reported in the previous study generalizes across backbone families. No backbone
cleared the Holm-corrected threshold, the outcome was classified as D, and the family pattern was
heterogeneous.

Two scope conditions belong in the same breath as that claim, because without them it reads as
something stronger than the data supports.

**This is not evidence of absence.** Six of ten backbones showed |τ| ≥ 0.29. What distinguishes them
is not that the associations were small but that they pointed in both directions: four positive, six
negative, with the largest of each sign at +0.60 and −0.52. A body of near-zero associations would
have been evidence against the hypothesis. A body of substantial associations without a consistent
sign is not the same thing, and should not be reported as though it were.

**The procedure's resolution was limited.** With five seeds the seed-level permutation distribution
admits 120 arrangements, so its p-values are quantised at multiples of 1/120 ≈ 0.0083, while Holm's
first threshold across ten tests is 0.005. The combination of five clusters and Holm correction
severely limits attainable family-wise significance. This is a property of the design, fixed before
any result was seen, and it constrains what a null can be taken to mean here. A reader who takes
Outcome D as a measurement of the world rather than partly a measurement of the procedure will
overread it.

### 5.2 Layer 2 — replication

On its original backbone, using the original covariate definition and the same analysis population,
the previous result reproduced exactly: τ = +0.5576 against a published τ = +0.5576, a difference of
0.0000.

This forecloses the explanation a null result usually invites. The same code that returned no
confirmatory evidence across ten backbones reproduced the original finding to four decimal places on
the one backbone where a published value exists to check against. Whatever accounts for the first
result, it is not that the pipeline was wrong.

The same τ, evaluated under the seed-level permutation test, gives p = 0.1333. Nothing about the data
or the effect size changed; the unit of resampling did. Observations sharing a random seed share an
adapter, a training trajectory and an initialisation, and treating the thirty rows as thirty
independent draws is what separates the two p-values.

We therefore report that the earlier inference is **not robust to changing the unit of resampling**.
That is a statement about a procedure, and it is checkable. It is not a claim that the earlier
finding was wrong: a permutation test over five clusters has little power, and p = 0.1333 records a
failure to resolve rather than a demonstration of absence. Both readings — that the effect is real
and that the earlier significance was overstated — remain open on this evidence, and we do not choose
between them.

### 5.3 Layer 3 — boundary of validity

The two studies answer different questions. The first asked whether the phenomenon exists on one
backbone. This one asks whether it holds across many. A negative answer to the second does not
retract the first, and the exact replication in §5.2 is what makes that separation credible rather
than merely rhetorical. What this study establishes is a boundary, not a refutation.

The boundary extends to the protocol as well as to the finding. The manipulation check inherited from
the earlier study accepts a geometry change only in the decreasing direction. Across the present
backbone set, within-class spectral decay moved *upward* under intervention in eight of ten cases.
With a single backbone, "the intervention decreases this quantity" and "the intervention changes this
quantity" were indistinguishable; with ten they are not. The declared portability probe — a
segmentation encoder, deliberately the most structurally distant backbone in the set — passed the
implementation gate on all twelve grid configurations and failed the manipulation gate on all twelve,
and is recorded *not testable* rather than as a negative result. That is the outcome the probe was
included to make possible.

Exploratory analyses suggested that variability attributable to seed and variability attributable to
the intervention were distributed differently across quantities, motivating future work on
experimental design.

> **Boundary for revision.** The sentence above is the whole of what the exploratory work supports in
> this paper. It must not be strengthened into a claim that variance decomposition proves anything,
> that seed variance dominates, that an intraclass correlation explains the result, or that family
> slopes differ. Those analyses were run after the confirmatory result, were not adjusted for
> multiplicity, rested on two models in some families, and were fitted under the same independence
> assumption this paper argues is not robust. Recorded as D-052.

### 5.4 Limitations

**Five seeds.** The resolution limit in §5.1 follows directly from the number of seeds, not from the
choice of a permutation test. A cluster-level procedure with more clusters would not have this
constraint.

**Two instances per family.** Family-level statements rest on two models each, which is the minimum
that permits within-family variance to be estimated at all and is far from the minimum that would
make it precise.

**A one-sided manipulation criterion.** The inherited gate accepts only decreasing geometry change,
while the protocol text describes dose-dependence without specifying direction. The two do not agree.
We report the discrepancy rather than repairing it: by the time it was identified it was already
known which backbone the repair would change, so amending the criterion would have been
outcome-contingent.

**Non-uniform artifact release.** Several checkpoints carry non-commercial licences and one prohibits
redistribution of derivatives, so released artifacts are represented by checksums where the licence
requires it. Reproduction from a clean clone is therefore not uniform across backbones.

**Saturated outcome.** Mahalanobis AUROC varied within a narrow band on every backbone; the widest
range across the analysis population was under 0.01 and the narrowest under 0.001. Kendall's τ is
scale-free and is unaffected by this, but a rank association over differences of this size should not
be read as a practically meaningful difference in estimator performance. Every τ in §4 is reported
with its AUROC range for that reason.

### 5.5 Future work

The design constraint identified in §5.1 is specific and actionable: at a fixed budget, the binding
resource for this class of study is the number of independent seeds rather than the number of rungs
on the dose ladder. We do not develop that further here.

---

## 6. Data and code availability

Repository, frozen preregistration commit, decision log. Release policy is stratified by checkpoint
licence; artifacts derived from non-redistributable backbones are represented by checksums only.

---

## Appendix A — Deviations from the preregistration

*[None at time of writing. Any entry here must carry its decision-log ID.]*
