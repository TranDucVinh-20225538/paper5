# MedSAM Integration Design — Task 4

    Status:  RECOMMENDATION — awaiting PI sign-off
    Role:    Architecture-portability probe, outside the family contrast (D-016)
    Feeds:   configs/medsam.yaml — the last UNRESOLVED representation in the set
    Date:    2026-08-11

This document is a specification, not code. It should be sufficient for an engineer to implement
without re-deriving any decision. Everything marked `VERIFY` must be confirmed by one forward pass
before extraction; nothing else should be revisited.

---

## 1. Why MedSAM is hard, in one paragraph

Every other backbone in the set hands you a vector. MedSAM does not. SAM's image encoder is a
ViTDet-style plain ViT with **no CLS token**, and its output is a spatial feature map intended for a
mask decoder, not a classifier. So two questions have to be answered that no other backbone in the
set poses: **which tensor is the representation**, and **how is it reduced to one vector**. Both are
genuine research decisions, and both are answered below by extending rules the protocol has already
committed to elsewhere — not by inventing new ones.

## 2. Architecture, as it bears on the choice

```
input 1024×1024×3
   │
   ├─ patch embed 16×16                    → 64×64 grid, 768-d tokens
   │
   ├─ 12 transformer blocks                  windowed attention (window 14),
   │                                         global attention at blocks 2, 5, 8, 11
   │
   ├─────────────────────────────── (A) PRE-NECK: 64×64×768        ← RECOMMENDED
   │
   ├─ neck: conv1×1 768→256, LN2d,
   │        conv3×3 256→256, LN2d
   │
   └─────────────────────────────── (B) POST-NECK: 256×64×64
                                          (the "image embedding" the mask decoder consumes)
```

Neither (A) nor (B) is a vector. Both are 64×64 spatial maps.

## 3. Representation choice — **(A) pre-neck, 768-d**

**Decision: take the transformer output before the neck.**

**Reason — this is D-018 applied consistently, not a new rule.** D-018 established that for every
VLM in the set the embedding is taken **pre-projection**, because the projection head is trained for
a different objective (text alignment) and would inject that objective's structure into the geometry
being measured.

**The neck is MedSAM's projection head.** It is a task-specific 768→256 projection trained to feed
the mask decoder. Taking post-neck features would measure the geometry of a
segmentation-decoder interface, not the geometry of the representation — the exact error D-018
exists to prevent, in a different costume.

Two further reasons, both secondary but pointing the same way:

- **Dimension.** Pre-neck is 768-d, matching the other ViT-B backbones in the set (BiomedCLIP,
  MoCo v3, OpenCLIP). Post-neck is **256-d**, which would be a severe outlier against a set spanning
  768–2048 and would make MedSAM the most exposed point in the D-019 dimension analysis.
- **Analogy to the rest of the set.** Pre-neck is the transformer's native output, which is what
  `forward_features` gives for PanDerm and what the CLS token gives for UNI and DINOv3.

*Rejected: post-neck (B).* It is the canonical "image embedding" and the more obvious choice, which
is why the rejection needs recording. It loses on all three arguments above.

## 4. Pooling — **global average pool over the 64×64 grid**

There is no CLS token, so a pooling rule must be chosen explicitly.

**Decision: unweighted global average pool over valid (non-padded) spatial positions.** See §5 for
why "valid" is doing real work in that sentence.

| Option | Verdict |
|---|---|
| **GAP** | **chosen** — parameter-free, deterministic, and already the set's precedent: ResNet-50 and EfficientNet-B3 both use GAP over their final stage |
| Max pool | rejected — dominated by outlier patches; unstable geometry |
| Attention pooling | rejected — introduces **trained parameters**, breaking "the backbone stays frozen" |
| Center-crop pooling | rejected — plausible for centred dermoscopy lesions, but adds an unprincipled hyperparameter |

**Known caveat, accepted.** SAM uses windowed attention with global attention at only 4 of 12 blocks,
so its tokens are less globally contextualised than a standard ViT's. GAP over such tokens may yield
different geometric behaviour than CLS pooling elsewhere in the set. This is not a defect to
engineer around — **it is part of what the portability probe exists to detect**, and it should be
stated in the results rather than corrected.

## 5. Preprocessing — and the one thing that could silently invalidate this backbone

SAM's transform is unusual and is the highest-risk part of this integration.

```
1. resize longest side to 1024, preserving aspect ratio
2. normalize:  mean = [123.675, 116.28, 103.53]
               std  = [ 58.395,  57.12,  57.375]
3. pad bottom/right with zeros to 1024×1024
```

> **Implementation trap.** Those constants are ImageNet statistics **on a 0–255 scale**, not 0–1.
> Applying them to `ToTensor()` output (0–1) produces silently wrong features with no error. This is
> the single most likely bug in the integration.

### 5.1 The padding problem — read this before implementing

Step 3 pads non-square images to square. Dermoscopy and smartphone images are generally not square,
so **a variable fraction of every padded image is zeros**, and that fraction is a deterministic
function of aspect ratio.

If ISIC 2019 and PAD-UFES-20 differ in aspect-ratio distribution — and there is no reason to expect
they match, given one is dermoscope output and the other is smartphone photography — then the padded
fraction differs systematically **by domain**.

The consequence is specific and serious. The nuisance direction is

    w = unit(μ_ISIC − μ_PAD)

computed from domain-conditional means. If padding differs by domain, **`w` partly encodes "how much
zero padding" rather than acquisition shift**. The intervention would then be orthogonalising against
a preprocessing artifact, on the one backbone whose preprocessing differs most from the rest of the
set. Gate 1 could pass and the result would still be meaningless.

### 5.2 Required pre-check, before any extraction

Compute and record the aspect-ratio distributions of ISIC 2019 and PAD-UFES-20, and the implied
padded-fraction distributions at 1024×1024. **Cheap, and it sizes the problem before any compute is
spent.** Record the result in the manifest either way.

### 5.3 Decision — masked pooling

**Pool only over spatial positions corresponding to non-padded image content.**

Each 16×16 patch maps to a known region of the padded canvas, so a valid-token mask is computable
deterministically from the original image dimensions. GAP is then taken over valid tokens only.

This keeps SAM's native preprocessing intact — so the encoder sees inputs it was trained on — while
removing the padding artifact from the pooled vector. It is strictly better than the alternatives:

- *Resize to square, ignoring aspect ratio* — no padding, but distorts every image, and the
  distortion again differs by domain.
- *Centre-crop to square, then resize* — no padding, but discards periphery, and the discarded
  fraction differs by domain too.
- *Keep padding, pool over everything* — the confound described in §5.1.

**Preregistered fallback:** if masked pooling proves infeasible, use centre-crop-to-square, and
report the change as a deviation with a decision-log entry. Do **not** fall back to unmasked pooling.

## 6. Checkpoint and loading

| | |
|---|---|
| Checkpoint | `medsam_vit_b.pth`, official MedSAM release `VERIFY` |
| Source | `bowang-lab/MedSAM` `VERIFY` |
| Loader | MedSAM's own repo loader — not `timm`, not HF |
| Load | **image encoder only**. Prompt encoder and mask decoder are not used and need not be constructed |
| License | `VERIFY` — record in `assets/README.md` per D-023 |

## 7. Intervention point

Unchanged from every other backbone, which is the point of the probe:

    z  = GAP_valid( image_encoder(x) )        768-d
    z' = z + W2 · GELU(W1 · z)                W2 zero-init
    L  = L_task + λ_proj · cos(z', w)²

- **Activation: GELU**, matching SAM's own — consistent with the per-backbone activation rule.
- The adapter acts **only on the pooled output vector**. Nothing inside the encoder is touched.
- `w` recomputed in MedSAM's own 768-d space, closed-form, never learned.
- `r` and `λ_proj` from a fresh grid — `d = 768 < 1536`, so **no 256 rung** (D-021).
- α-ladder by post-hoc interpolation from the one trained adapter, as everywhere.

## 8. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Padding creates artificial domain signal in `w`** | **high** | §5.2 pre-check + §5.3 masked pooling |
| 2 | **Extraction cost.** 1024×1024 input = 4,096 tokens vs. 196 at 224/16 elsewhere. Roughly 20–50× the per-image cost of any other backbone, and quadratic in the global-attention blocks | **high** | Budget it explicitly. It is the most expensive extraction in the set by a wide margin; do not size the job from the other backbones |
| 3 | **Gate 0-pre failure.** MedSAM is a segmentation model — its features encode boundaries and regions, not necessarily class-discriminative semantics. Frozen features may not support lesion classification | medium | Legitimate declared outcome under D-020, **provided §9 is followed** |
| 4 | Normalization scale bug (0–255 vs 0–1) | medium | §5, explicit in the frozen preprocessing asset |
| 5 | Windowed attention ⇒ GAP behaves unlike CLS pooling elsewhere | low | Accepted and reported; this is what the probe measures |
| 6 | No CLS token invites someone to "add one" | low | Documented here as a rejected option |

## 9. Sequencing correction — run PanDerm first, not MedSAM

**The kickoff, the implementation brief and `protocol.md` all say to run MedSAM first. That is right
about architecture risk and wrong about debuggability, and it should be changed.**

MedSAM is simultaneously the structurally most different encoder **and** the one with the highest
prior probability of a legitimate Gate 0-pre failure (risk 3). Run it first and a failure is
**ambiguous**: it cannot be distinguished from a misconfigured pipeline, because there is no
validated baseline yet to compare against. That ambiguity would arrive at the worst moment — the
first result of the study.

**Recommendation:**

1. **PanDerm first, as a pipeline regression test.** Its embeddings already exist from Paper 4, so
   the full ladder can be re-run and checked against **known values**. This is the only backbone in
   the set that can validate the pipeline against ground truth, and it costs almost nothing.
2. **MedSAM second, as the portability probe.** Any failure is then unambiguous: the pipeline is
   already known good, so the failure is a property of MedSAM.
3. Then the remaining eight, in any order.

This costs one cheap run and converts MedSAM's result from ambiguous to diagnostic. The original
instinct — front-load the architecture risk — is preserved: MedSAM still runs before the eight
backbones whose budget depends on the recipe porting.

## 10. Acceptance criteria for the implementation

Implementation is complete when all of the following hold. These are checks, not aspirations.

- [ ] Loads the image encoder alone; prompt encoder and mask decoder are never constructed
- [ ] `assets/preprocessing/medsam.json` frozen and hashed; hash recorded in the config
- [ ] Normalization verified against a known reference output — **not** assumed from the constants
- [ ] Aspect-ratio / padded-fraction pre-check (§5.2) run and recorded in the manifest **for both
      datasets**, before extraction
- [ ] Valid-token mask derived from original image dimensions and unit-tested on a
      deliberately non-square input
- [ ] Extracted embedding is `[N, 768]`, pre-neck `VERIFY`
- [ ] `w` recomputed in MedSAM's own space; not reused from any other backbone
- [ ] Gate 0-pre runs **before** the domain probe and its outcome is recorded distinctly from Gate 0
- [ ] PanDerm regression run (§9) completed and matching Paper 4 **before** this backbone is run

## 11. Proposed decision-log entries

| ID | Decision |
|---|---|
| D-025 | MedSAM representation = **pre-neck 768-d**, by direct extension of D-018 — the neck is a projection head |
| D-026 | MedSAM pooling = **GAP over valid (non-padded) spatial positions**; masked pooling required, centre-crop the preregistered fallback, unmasked pooling forbidden |
| D-027 | **PanDerm runs first as a pipeline regression test**, MedSAM second. Supersedes "MedSAM first" in the kickoff, the brief and `protocol.md` |
