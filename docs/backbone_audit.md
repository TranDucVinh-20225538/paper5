# Backbone Audit — Task 3

    Status:  RECOMMENDATION — awaiting PI sign-off. Nothing here is decided.
    Feeds:   D-004, D-005, D-006  (and surfaces three new decisions)
    Date:    2026-08-11
    Author:  protocol lead (Claude), reviewed by PI before anything is locked

## Method and confidence policy

Every backbone currently listed for Paper 5 was assessed on the nine axes in the task brief. Facts
below are split into two kinds, and the split is deliberate:

- **Structural claims** (paradigm, pooling mechanism, whether a pooled vector exists at all) — these
  drive the recommendations and I am confident in them.
- **Numeric claims** (exact embedding dimension, exact hub id, exact checkpoint tag) — marked
  `VERIFY`. These must be confirmed empirically at extraction time. `configs/_template.yaml` already
  says *"confirm empirically, do not trust the model card"*; that applies to every number here.

No recommendation below depends on an unverified number. If a `VERIFY` turns out different, the
config changes and the reasoning stands.

---

## 1. Headline

**The audit reopens a cell the kickoff recorded as resolved, and closes both cells it recorded as
open.**

| Cell | Kickoff status | After audit |
|---|---|---|
| CNN | resolved — ResNet-50 + EfficientNet-B3 | **confirmed**, no change |
| Medical, image-only SSL | resolved — PanDerm + MedSAM | **REOPENED** — MedSAM is not SSL (§2.1) |
| Medical, vision-language | open, n=1 | **closed** — BiomedCLIP + MONET (§3.3) |
| General, image-only SSL | open, n=1 | **closed** — DINOv3 + MoCo v3, *not* DINOv2 (§3.4) |
| General, vision-language | resolved — SigLIP + OpenCLIP | **confirmed**, variants pinned (§3.5) |

Recommended final set: **10 family members (5 cells × 2) + 1 architecture-portability probe = 11.**

Three findings below are more consequential than the backbone list itself. §2.2 in particular is the
kind of thing a statistics reviewer ends a paper with.

---

## 2. Cross-cutting findings

These emerge only from looking at the set as a whole. None of them are visible in a single-backbone
study, which is why Paper 4 never had to deal with them.

### 2.1 MedSAM is not self-supervised, and the family label is doing real work

The kickoff places MedSAM in *"Medical, image-only SSL"* alongside PanDerm. PanDerm is
self-supervised on dermatology images. **MedSAM is supervised segmentation** — SAM fine-tuned on
medical segmentation masks. The two share "medical" and "image-only" and nothing else.

This matters because **family is the primary analysis**, not a label. The fixed-effect model
estimates planned contrasts *between* cells, which assumes each cell is homogeneous in the thing the
cell is named for. A cell containing one SSL model and one supervised-segmentation model is not
homogeneous in pretraining paradigm, so the "medical-SSL vs. general-SSL" contrast would not be
estimating what it claims to estimate.

**Recommendation.** Split the two roles MedSAM is currently being asked to play:

- Restore the cell to genuine medical image-only SSL: **PanDerm + UNI** (§3.2).
- Keep MedSAM, but as a **declared architecture-portability probe outside the family contrast** —
  included in the per-backbone analysis, excluded from the family-level model, and declared as such
  in advance.

This is not a demotion. MedSAM's scientific value was always as a stress test of whether the ladder
survives a structurally different encoder; that value is fully preserved, and stating it in advance
converts an awkward cell member into a preregistered robustness check.

*Rejected alternative:* rename the cell to "medical, image-only" and accept heterogeneity. Cheaper,
but it silently weakens the primary contrast, and a reviewer comparing the cell contents to the cell
name will find it.

### 2.2 Embedding dimension confounds the primary covariate

**This is the most serious finding in the audit.**

Condition number is `λ_max / λ_min` of the embedding covariance. For fixed sample size, higher-
dimensional embeddings carry more small eigenvalues, so **κ grows with dimension as a matter of
arithmetic, independent of any property the study cares about**.

Across the recommended set, dimension is not random — it is correlated with family:

| Family | Typical dim |
|---|---|
| CNN | 1536 – 2048 |
| ViT-B backbones (most VLMs, MoCo v3) | 768 |
| ViT-L backbones (PanDerm, UNI, DINOv3) | 1024 |

So the CNN cell could show systematically higher condition number *because those embeddings are
2048-d*, and if AUROC also differs by family, the family contrast picks up dimension rather than
geometry. Paper 4 could not have detected this: with one backbone, `d` was constant.

**Recommendation — both, not either:**

1. **Primary:** compute condition number over the **top-k principal components with k fixed across
   all backbones**, k chosen below the smallest embedding dimension in the set (k = 256 is a safe
   default at min-dim 768; pin the exact value in the preregistration). This makes κ dimension-
   commensurable by construction rather than by adjustment.
2. **Sensitivity:** also report raw full-dimension κ with `d` entered as a covariate in the
   fixed-effect model, and confirm the family contrasts do not change sign or significance.

If (1) and (2) disagree, that disagreement is itself a reportable result and must not be resolved by
picking the friendlier one after the fact — which is precisely why this goes in the preregistration
now, before any κ has been computed.

### 2.3 Gate 0 has a third outcome it does not currently allow

Gate 0 reads: domain-probe accuracy > majority + 0.05 and balanced accuracy > chance + 0.10, and
**"Fail ⇒ the implementation is broken. This is not a finding."**

That interpretation is calibrated to PanDerm, whose frozen features are strong on dermoscopy. Across
a set that deliberately includes domain-mismatched backbones (UNI on pathology, MoCo v3 and DINOv3 on
natural images, MedSAM on segmentation), Gate 0 can fail for an entirely legitimate reason: **the
frozen features are too weak on this task to support any probe.** That is not a bug.

Leaving the gate binary creates two bad outcomes, and the second is worse:

1. Weak features get reported as broken code.
2. Someone "fixes" the implementation until the gate passes — which is outcome-contingent tuning
   entering through the back door, in the one place the protocol was built to prevent it.

**Recommendation.** Add **Gate 0-pre**, run *before* the domain probe: a frozen-feature adequacy
check (ID-task linear probe accuracy and k-NN accuracy against a preregistered floor). Then:

| Gate 0-pre | Gate 0 | Reading |
|---|---|---|
| pass | pass | proceed |
| pass | fail | **implementation broken** — fix it, not a finding |
| fail | — | **features inadequate for this task** — declared exclusion, pre-specified, not a bug and not a falsification |

This gives Gate 0 the same three-way structure Gate 1 already has, and for the same reason: *not
testable* must be distinguishable from *broken* and from *falsified*. The protocol already
understands this distinction — it just has not applied it at Gate 0.

### 2.4 Adapter capacity `r` is absolute, but embeddings are not

The pre-committed grid is `r ∈ {16,32,64,128}`, absolute. On a 768-d embedding r=16 is 2.1% of the
representation; on 2048-d it is 0.78%. The **relative** capacity of the intervention therefore varies
by a factor of ~2.7 across the set, systematically by family.

The grid is re-run per backbone and takes the smallest passing value, so this partly self-corrects.
But the grid's *range* may be miscalibrated at the high-dimensional end.

**Recommendation.** Keep the grid absolute (changing it would break comparability with Paper 4), but
(a) record `r/d` alongside `r` in the manifest, and (b) preregister an extension of the grid to
`{16,32,64,128,256}` for backbones with `d ≥ 1536`, so the CNN cell is not capacity-starved relative
to the rest. Decide now, not when a grid fails to find a passing value.

### 2.5 Licenses are heterogeneous, and this breaks a uniform reproducibility story

Confirmed 2026-08-11: **UNI is CC-BY-NC-ND-4.0**, and its card states that derivatives include
*"models trained on outputs from the UNI model or datasets created from the UNI model."*

Under that definition, this study's own artifacts are derivatives: the **adapter is a model trained
on UNI outputs**, and the **extracted embeddings are a dataset created from UNI**. Non-commercial
academic research is explicitly permitted, so the *research* is fine. The problem is downstream:

- **UNI's adapter checkpoints and adapted embeddings cannot be released publicly**, while every other
  backbone's can. `scripts/reproduce.sh` therefore reproduces **non-uniformly** across the set —
  checksums-only for UNI, checksums plus artifacts elsewhere.
- Attribution must extend to **ViT and DINOv2**, not just UNI.
- DermLIP (if ever adopted) and Derm1M are CC BY-NC-4.0 — same non-commercial constraint, no ND.
- Any commercially-affiliated co-author needs this checked before submission, not after.

**Recommendation.** Add a license column to `assets/README.md` covering every checkpoint, and
**declare the asymmetric release policy in the preregistration**. A reviewer or a reproducibility
editor discovering at submission time that one backbone's artifacts are undistributable is a bad
surprise; stating it up front is simply a constraint. This is the kind of thing that is free to
handle now and expensive to handle in a rebuttal.

### 2.6 UNI and DINOv3 share an objective family — and that is a feature

UNI is trained with the **DINOv2 recipe** (self-distillation + iBOT + KoLeo). DINOv3 is the same
self-distillation lineage. So the two SSL cells are:

| Cell | Instance 1 | Instance 2 |
|---|---|---|
| Medical SSL | PanDerm | UNI *(DINOv2 recipe)* |
| General SSL | DINOv3 *(DINO lineage)* | MoCo v3 *(contrastive)* |

This is worth stating explicitly because it looks like a problem and is the opposite. The
**medical-SSL vs. general-SSL contrast is the comparison of interest**, and having UNI and DINOv3
share a pretraining objective means that contrast is **partially controlled for objective** —
what differs between them is principally the pretraining *domain*, which is exactly what the family
label names.

Note this does **not** weaken the §3.4 argument against DINOv2. That argument was about *within-cell*
spread: two members of one cell must differ, or within-family variance is under-estimated. Sharing an
objective *across* cells along a controlled comparison is a different thing entirely, and desirable.

**Residual imbalance, accepted and declared.** Dimensions are medical-SSL `{1024, 1024}` vs
general-SSL `{1024, 768}`. Forcing a match would mean picking a ViT-L contrastive SSL model, and the
credible options carry Gate 0-pre risk (MAE) or DINO-lineage overlap (iBOT). The imbalance is left in
place and carried analytically by §2.2's fixed-k construction, which is what that construction is
for. State it in the limitations rather than contorting the sample to hide it.

---

## 3. Per-backbone assessment

Nine axes per the task brief. `VERIFY` = confirm empirically before extraction.

### 3.1 CNN cell — confirmed, no change

**ResNet-50**

| | |
|---|---|
| Family | CNN, supervised ImageNet |
| Dim | 2048 post-GAP `VERIFY` |
| Pooling | global average pool — unambiguous, no decision needed |
| Preprocessing | 224×224, resize-256 + center-crop, ImageNet mean/std |
| Activation | **ReLU** — adapter must match, do not inherit GELU |
| Maturity | maximal |
| Availability | `timm`, `torchvision` |
| Ladder compatibility | **high** — adapter on a clean 2048-d vector |
| Difficulty | trivial |
| Scientific value | **high**, for a specific reason: it is the Papers 1–3 baseline, so it is the continuity anchor for the whole program, and it is the only *supervised* pretraining objective in the set |

**EfficientNet-B3**

| | |
|---|---|
| Family | CNN, supervised ImageNet |
| Dim | 1536 post-GAP `VERIFY` |
| Pooling | global average pool |
| Preprocessing | 300×300 at native B3 eval resolution `VERIFY` — **not** 224; do not copy ResNet-50's spec |
| Activation | **SiLU/Swish** |
| Maturity | high |
| Availability | `timm` |
| Ladder compatibility | high |
| Difficulty | trivial |
| Scientific value | moderate — within-CNN variance instance, already present in the CSG-Skin / DST-Skin infrastructure |

Within-cell spread is adequate: residual vs. compound-scaled inverted-bottleneck, ReLU vs. SiLU,
2048 vs. 1536-d. Two genuinely different CNNs, not one model at two scales.

### 3.2 Medical image-only SSL — reopened, recommend PanDerm + UNI

**PanDerm** — unchanged, and the single most important backbone in the study.

| | |
|---|---|
| Dim | 1024, `forward_features(x, is_train=False)` |
| Pooling | none needed — pooled vector direct, no CLS ambiguity |
| Maturity | research checkpoint, but validated end-to-end in Paper 4 |
| Difficulty | **none** — frozen embeddings already exist at `Paper4/PhaseB/assets/reference_embeddings/` |
| Scientific value | **maximal** — without it Paper 5 is not a replication. It is also the only backbone whose `r`/`λ_proj` are already resolved (16 / 0.1) |

**UNI** — recommended second instance. **All facts below CONFIRMED from the model card 2026-08-11;
access granted.**

| | |
|---|---|
| Family | medical, image-only SSL — **DINOv2 recipe confirmed**: DINO self-distillation + iBOT MIM + KoLeo, on Mass-100K histopathology |
| Dim | **1024** — ViT-L/16, `feature_emb` shape `[1, 1024]` ✓ |
| Pooling | CLS token ✓ |
| Preprocessing | 224×224, `Resize(224)` + ImageNet mean/std ✓ |
| Loader | `timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)` — **`init_values` is required** or LayerScale params fail to load |
| Maturity | high — the card explicitly documents **frozen** use: logistic regression, k-NN, and nearest-centroid on the class token |
| Availability | access **granted** ✓ — no longer a lead-time item |
| Ladder compatibility | high |
| Difficulty | low |
| Scientific value | high — genuine medical SSL, paradigm-matched to PanDerm |
| **License** | **CC-BY-NC-ND-4.0 — see §2.5, this has consequences** |

The frozen-use documentation is the specific confirmation that mattered: the Gate 0-pre risk that
disqualified RETFound is evidenced as low here by the authors' own benchmarking regime.

*Why UNI over RETFound.* RETFound is the other credible medical image-only SSL model and is openly
available, which is a real advantage. It is rejected because it is **MAE-pretrained**, and MAE frozen
features are well known to be weak under linear probing and k-NN without fine-tuning. Given §2.3,
that is a direct Gate 0-pre risk: RETFound would carry a meaningful chance of burning a backbone slot
on *features inadequate*, in the cell that most needs two working instances. UNI is designed for the
frozen-feature regime this study lives in. If UNI access is refused, RETFound is the fallback and the
Gate 0-pre risk must be stated in the paper.

Both are domain-mismatched to dermoscopy (pathology, retina). That is acceptable and arguably
desirable — mismatched backbones widen the geometric range, which is what the τ association needs.
What is *not* acceptable is mismatching the paradigm, which is why MedSAM moves out (§2.1).

**MedSAM** — retained as architecture-portability probe, outside the family contrast.

| | |
|---|---|
| Family | supervised segmentation (SAM fine-tune) — **not SSL** |
| Dim | **no pooled vector exists.** ViT-B image encoder emits a 256-channel × 64×64 spatial feature map at 1024×1024 input `VERIFY` |
| Pooling | **UNRESOLVED — this is a genuine research decision, see Task 4** |
| Preprocessing | 1024×1024, SAM-specific normalization — by far the heaviest in the set |
| Maturity | high, widely used |
| Ladder compatibility | **the open question the whole pilot exists to answer** |
| Difficulty | **high** — highest in the set |
| Scientific value | high *as a portability stress test*; low as a family representative |

Run it first, as the kickoff already specifies. Full treatment in Task 4.

### 3.3 Medical vision-language — closed, recommend BiomedCLIP + MONET

**BiomedCLIP**

| | |
|---|---|
| Variant | PubMedBERT + ViT-B/16, 224px `VERIFY` |
| Dim | **768 pre-projection**, ~512 projected `VERIFY` — see §2.2 and the projection decision below |
| Pooling | image tower pooled output |
| Availability | `open_clip` / HuggingFace |
| Maturity | high |
| Difficulty | low |
| Scientific value | high — broad biomedical image-text pretraining |

**MONET** — recommended second instance. **Verified 2026-08-11.**

> ⚠ **Name collision — read this before searching.** There is an unrelated model called
> **`NOVAglow646/Monet-7B`** on HuggingFace: a Qwen2.5-VL-7B fine-tune for latent-visual-space
> reasoning (arXiv 2511.21395). It has nothing to do with dermatology and is not a CLIP model.
> The correct model is **`suinleelab/monet`**.

| | |
|---|---|
| Family | medical VLM, **dermatology-specific** — CLIP contrastive, 105,550 derm image–text pairs from medical literature |
| Architecture | **ViT-L/14** image encoder + masked self-attention text encoder, 0.4B params ✓ |
| Dim | 1024 pre-projection (ViT-L), 768 projected `VERIFY` — take pre-projection per D-018 |
| Availability | `suinleelab/monet` — **public, not gated** ✓ |
| Loader | `transformers`: `AutoProcessor` + `AutoModelForZeroShotImageClassification` ✓ |
| Maturity | high — Nature Medicine 2024, checkpoint released with the paper |
| Scientific value | **high** — the only candidate both paradigm-matched *and* domain-matched |

Being **ViT-L/14** rather than ViT-B is a bonus that was not part of the original reasoning: paired
with BiomedCLIP (ViT-B/16), the medical-VLM cell spans `{B, L}`, which is better within-cell spread
*and* better dimension balance against the medical-SSL cell than a second ViT-B would give. It also
forces a revision to §3.5 — see there.

*Why MONET over PLIP, MedCLIP and Quilt-1M-CLIP.* All three kickoff candidates are medical VLMs in
name but are trained on the wrong medical domain: PLIP and Quilt are **histopathology**, MedCLIP is
**chest radiography**. Pairing any of them with BiomedCLIP would make the medical-VLM cell
heterogeneous in imaging domain, adding a second uncontrolled axis to a cell that already differs in
corpus and scale — and MedCLIP is worse still, since grayscale radiographs have image statistics
unlike anything else in the study. MONET keeps the cell homogeneous in domain so the family contrast
estimates family, not domain.

### 3.4 General image-only SSL — closed, recommend DINOv3 + MoCo v3, explicitly **not** DINOv2

**DINOv3**

| | |
|---|---|
| Dim | depends on variant — **pin one and record it** `VERIFY` |
| Pooling | CLS token conventional; confirm against the published linear-probe recipe, and decide explicitly whether mean-pooled patch tokens are concatenated (some recipes do) |
| Difficulty | low–moderate |
| Scientific value | high — current-generation general SSL |

**MoCo v3 (ViT-B/16)** — recommended second instance.

| | |
|---|---|
| Objective | **instance-contrastive** — genuinely different from DINOv3's self-distillation |
| Dim | 768 `VERIFY` |
| Pooling | CLS token |
| Maturity | high, official checkpoints |
| Difficulty | low |

**Why not DINOv2, despite the kickoff naming it.** DINOv2 and DINOv3 are the same lineage — same
team, same self-distillation family, DINOv3 being the scaled successor. Using them as the two
instances of "general SSL" would measure *scaling* variance and call it *family* variance, which
systematically **under-estimates within-family spread**.

That is Assumption 6 again, in miniature — the exact flaw Paper 4 named for its own one-CNN-one-FM
design, and the reason the ≥2-per-cell rule exists at all. A cell whose two members differ only in
scale satisfies the rule's letter and defeats its purpose. MoCo v3 differs in pretraining
*objective*, which is what the cell is named for.

*Why not MAE, which would be the maximal objective contrast.* Same reason RETFound was rejected in
§3.2: MAE frozen features are weak under linear probing and k-NN, so the Gate 0-pre risk is
material. MoCo v3 buys most of the objective contrast at a fraction of the risk.

### 3.5 General vision-language — confirmed, variants pinned

**SigLIP**

| | |
|---|---|
| Variant | `google/siglip-base-patch16-224` `VERIFY` |
| Dim | 768 pre-projection `VERIFY` |
| Pooling | **attention-pooling (MAP) head — no CLS token.** Do not assume CLS |
| Objective | sigmoid pairwise contrastive |

**OpenCLIP**

| | |
|---|---|
| Variant | `ViT-B-16` / `laion2b_s34b_b88k` `VERIFY` |
| Dim | 768 pre-projection, 512 projected `VERIFY` |
| Pooling | image tower pooled output |
| Objective | softmax contrastive |

*"OpenCLIP" alone does not identify a model* — architecture **and** pretraining corpus both need
pinning, and the pinned choice recorded with a decision ID. This is literally the reviewer question
the decision log was built for.

**Scale pinning — revised 2026-08-11** after MONET was confirmed as ViT-L/14.

The original recommendation was to pin both general-VLMs at ViT-B/16 to match BiomedCLIP. That is now
wrong. The medical-VLM cell is `{BiomedCLIP ViT-B/16, MONET ViT-L/14}` = `{B, L}`, so pinning the
general-VLM cell to `{B, B}` would leave the two VLM cells differing in **scale composition** — and
since dimension tracks scale, §2.2's confound would then run straight along the medical-vs-general
VLM contrast, which is one of the planned comparisons.

**Pin the general-VLM cell to `{B, L}` as well:**

- **OpenCLIP ViT-B/16** (768 pre-projection)
- **SigLIP ViT-L/16** — e.g. `google/siglip-large-patch16-256` `VERIFY` (1024 pre-projection)

Both VLM cells then carry one ViT-B and one ViT-L, so the medical-vs-general VLM contrast is
**controlled for scale by construction** rather than adjusted for it afterwards. This costs nothing —
both variants exist and are equally available.

---

## 4. Rejected candidates

| Candidate | Considered for | Rejected because |
|---|---|---|
| **DINOv2** | general SSL, 2nd instance | Same lineage as DINOv3 — measures scaling variance, not family variance (§3.4) |
| **MAE** | general SSL, 2nd instance | Maximal objective contrast, but weak frozen features ⇒ material Gate 0-pre risk (§2.3) |
| **PLIP** | medical VLM, 2nd instance | Histopathology — wrong imaging domain, makes the cell heterogeneous in domain |
| **Quilt-1M-CLIP** | medical VLM, 2nd instance | Histopathology — same objection |
| **MedCLIP** | medical VLM, 2nd instance | Chest radiography — same objection, plus grayscale image statistics unlike anything else in the set |
| **RETFound** | medical SSL, 2nd instance | MAE-pretrained ⇒ Gate 0-pre risk. Moot — UNI access granted |
| **DermLIP-PanDerm** | medical VLM, 2nd instance | **Disqualified — its vision encoder *is* PanDerm** (§4.1) |
| **DermLIP ViT-B/16** | medical VLM, 2nd instance | Credible and confirmed available, but ViT-B duplicates BiomedCLIP's scale and gives less within-cell spread than MONET. **Verified fallback** |
| **Monet-7B** (`NOVAglow646`) | — | Name collision only. Qwen2.5-VL reasoning model, unrelated to dermatology |

Every rejection above is a reviewer question with a written answer, which is the point.

### 4.1 DermLIP-PanDerm — the trap worth recording

Derm1M released two DermLIP checkpoints (confirmed public, CC BY-NC-4.0):

- `redlessone/DermLIP_ViT-B-16` — independent ViT-B/16 vision encoder
- `redlessone/DermLIP_PanDerm-base-w-PubMed-256` — **vision encoder is PanDerm-B**

The second is trained on a million-scale derm corpus and is superficially the strongest
dermatology-VLM candidate available. **It must not be used in this study.**

Its vision encoder is PanDerm, which already occupies the medical-SSL cell. Including it would put
the same backbone on both sides of the **medical-SSL vs. medical-VLM contrast** — one of the planned
family comparisons would then be partly comparing a model to itself, and the two cells would no
longer be independent. The contrast would be biased toward "no difference" for a purely structural
reason, and the direction of that bias is exactly toward the study's own hypothesis.

Recorded here because this is a trap a later collaborator would plausibly walk into: DermLIP-PanDerm
looks like the obvious best choice right up until you check what its encoder is.

---

## 5. Recommended final set

| Cell | Instance 1 | dim | Instance 2 | dim |
|---|---|---|---|---|
| CNN, supervised | ResNet-50 | 2048 | EfficientNet-B3 | 1536 |
| Medical, image-only SSL | PanDerm (ViT-L) | 1024 | **UNI** (ViT-L/16) | 1024 ✓ |
| Medical, vision-language | BiomedCLIP (ViT-B/16) | 768 | **MONET** (ViT-L/14) | 1024 |
| General, image-only SSL | DINOv3 (ViT-L) | 1024 | **MoCo v3** (ViT-B/16) | 768 |
| General, vision-language | OpenCLIP (ViT-B/16) | 768 | **SigLIP (ViT-L/16)** | 1024 |
| **Portability probe** *(outside family contrast)* | MedSAM | — | — | — |

Dimensions are pre-projection throughout (D-018). The two VLM cells are scale-matched at `{B, L}`;
the two SSL cells are not, and §2.6 explains why that is accepted rather than forced.

**N = 10 family members, balanced 5 × 2, plus 1 declared probe.**

Three consequences worth stating:

1. **D-006 resolves to N=10**, not 7 or 8. The `7` in both documents traces to *7 backbones needing
   extraction* (8 minus PanDerm) being carried over as *7 total*.
2. **The balanced 5×2 design is materially better** for the planned family contrasts than any
   unbalanced alternative, and it is only affordable because D-003 came back confirmed.
3. **The fixed-effect argument needs restating, and gets easier.** The kickoff argued against random
   effects citing "guidance wants 8–10+ groups" against "this design has 7". At N=10 that particular
   sentence no longer works. **The decision does not change** — fixed effects with planned contrasts
   remain right on interpretability and power grounds, and the contrasts are the actual question. But
   the justification must be rewritten to say so, and the exploratory mixed model can now be reported
   with somewhat more confidence than "low". Do not leave the original sentence in the manuscript.

---

## 6. Proposed decision-log entries

All **Pending**, awaiting PI sign-off. None written to the log as accepted.

| ID | Decision | Depends on |
|---|---|---|
| D-014 | Medical-VLM 2nd instance = **MONET**; PLIP/MedCLIP/Quilt rejected on imaging-domain mismatch | closes D-004 |
| D-015 | General-SSL 2nd instance = **MoCo v3**; DINOv2 rejected as same-lineage, MAE on Gate 0-pre risk | closes D-005 |
| D-016 | **MedSAM reclassified** — architecture-portability probe outside the family contrast; medical-SSL cell = PanDerm + **UNI** | reopens/closes the cell in §2.1 |
| D-017 | **N = 10** family members + 1 probe; balanced 5×2 | closes D-006, unblocks D-007 and power |
| D-018 | **All VLM embeddings taken pre-projection**, uniformly across BiomedCLIP / MONET / SigLIP / OpenCLIP | resolves 4 `UNRESOLVED` configs |
| D-019 | **Condition number computed on top-k PCs, k fixed across backbones**; raw κ with `d` as covariate as sensitivity | §2.2 — must be preregistered |
| D-020 | **Gate 0-pre added** — frozen-feature adequacy check; *features inadequate* becomes a declared exclusion distinct from *broken* | §2.3 |
| D-021 | Grid extended to `r ∈ {16,32,64,128,256}` for `d ≥ 1536`; `r/d` recorded in manifest | §2.4 |
| D-022 | **OpenCLIP ViT-B/16 + SigLIP ViT-L/16**, so both VLM cells are scale-matched at `{B, L}` | §3.5, revised after MONET confirmed as ViT-L/14 |
| D-023 | **Asymmetric artifact-release policy declared in advance** — UNI adapter checkpoints and adapted embeddings are not publicly redistributable (CC-BY-NC-ND-4.0); every checkpoint's license recorded in `assets/README.md` | §2.5 |
| D-024 | **DermLIP-PanDerm permanently excluded** — its vision encoder is PanDerm, which would put the same backbone on both sides of the medical-SSL vs. medical-VLM contrast | §4.1 |

**D-018 explained**, since it resolves four configs at once and is easy to get wrong. Every VLM in
the set has a projection head trained for image–text alignment. The projected space is lower-
dimensional and optimized for cross-modal retrieval; the pre-projection space is the image tower's
native representation. A learned linear projection can change condition number substantially, so
choosing wrong would corrupt the primary covariate. Take **pre-projection**, because: it is the
analogue of `forward_features` for PanDerm and of GAP for the CNNs — the native representation in
each case; the projection head optimizes a different objective and would inject text-alignment
structure into the geometry being measured; and the choice must be uniform across all four VLMs or
the two VLM cells are confounded by an inconsistent extraction rule.

---

## 7. Immediate action items

1. ~~Request UNI access~~ — **granted** ✓. No lead-time items remain; every checkpoint in the
   recommended set is obtainable today.
2. ~~Verify MONET~~ — **verified** ✓ (`suinleelab/monet`, ViT-L/14, public, ungated).
3. **Check MONET's license explicitly.** Not shown on the card excerpt reviewed. UNI already forces
   the study non-commercial, so this is unlikely to change the overall constraint — but it feeds
   D-023 and must be recorded, not assumed.
4. Pin exact variants and confirm every remaining `VERIFY` dimension empirically — one forward pass
   each, recorded in the manifest.
5. Then Task 4 (MedSAM), then Task 2 (portability), then Task 1 (lock).

## 8. Verification record

| Claim | Status | Source |
|---|---|---|
| UNI = ViT-L/16, 1024-d, CLS, DINOv2 recipe | **confirmed** | model card, 2026-08-11 |
| UNI documented for frozen linear-probe / k-NN use | **confirmed** | model card |
| UNI access granted | **confirmed** | PI account |
| UNI = CC-BY-NC-ND-4.0, derivatives include models trained on its outputs | **confirmed** | model card |
| MONET = `suinleelab/monet`, CLIP ViT-L/14, public, ungated | **confirmed** | HF + Nature Medicine 2024 |
| `Monet-7B` is an unrelated Qwen2.5-VL model | **confirmed** | HF |
| DermLIP checkpoints public, two variants | **confirmed** | Derm1M repo |
| DermLIP-PanDerm's vision encoder is PanDerm | **confirmed** | Derm1M repo |
| All remaining dimensions and hub ids | `VERIFY` | one forward pass each |

Sources: [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) ·
[suinleelab/monet](https://huggingface.co/suinleelab/monet) ·
[MONET, Nature Medicine 2024](https://www.nature.com/articles/s41591-024-02887-x) ·
[suinleelab/MONET](https://github.com/suinleelab/MONET) ·
[Derm1M / DermLIP](https://github.com/SiyuanYan1/Derm1M) ·
[Derm1M paper](https://arxiv.org/abs/2503.14911)
