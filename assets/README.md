# assets/

**Frozen, hashed specifications.** Distinct in kind from `configs/`.

| | `configs/` | `assets/` |
|---|---|---|
| Contains | values that may be searched over (`r`, `λ_proj`) | specs that must never be adjusted |
| Changing one is | normal work | a decision-log entry |
| Verified by | nothing | sha256, checked at Step 0 |

Keeping these apart is not tidiness. If preprocessing sits next to a tunable hyperparameter, someone
eventually "just tweaks" it, and the run silently stops being comparable to every run before it.

## preprocessing/

One JSON per backbone, mirroring **that backbone's own published eval transform** — not PanDerm's.
Record the sha256 in the backbone's config and cite the source of the transform.

A mismatch between file and recorded hash is a **hard error**, not a warning.

## checkpoints/

Weights are gitignored. Track provenance here: hub id or URL, sha256, download date, license.


## Checkpoint licences (D-046)

Recorded per checkpoint, with the release tier each implies. Verified 2026-08-14.

| Backbone | Repository | Licence | Tier |
|---|---|---|---|
| ResNet-50 | `timm/resnet50.a1_in1k` | Apache-2.0 | A |
| EfficientNet-B3 | `timm/efficientnet_b3.ra2_in1k` | Apache-2.0 | A |
| SigLIP | `google/siglip-large-patch16-256` | Apache-2.0 | A |
| MedSAM | `wanglab/medsam-vit-base` | Apache-2.0 | A |
| BiomedCLIP | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | MIT | A |
| OpenCLIP | `laion/CLIP-ViT-B-16-laion2B-s34B-b88K` | MIT | A |
| DINOv3 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 Licence | B — notice |
| MoCo v3 | `nyu-visionx/moco-v3-vit-b` | CC-BY-NC-4.0 (FAIR) | B — non-commercial |
| UNI | `MahmoodLab/UNI` | CC-BY-NC-ND-4.0 | **C — no derivatives** |
| MONET | `suinleelab/monet` | CC-BY-NC-SA-4.0 | **C — under review** |
| PanDerm | local checkpoint | see source | — |

**Tier B obligations.** DINOv3 requires a copy of its agreement, a "Built with DINOv3" notice
wherever artifacts are distributed, and acknowledgement in the publication. MoCo v3 requires
attribution to FAIR; note that the HuggingFace mirror declares no licence while FAIR's own
repository is CC-BY-NC-4.0, which is one of the two reasons D-042 required provenance verification
before that checkpoint was frozen.

**Tier C.** UNI's card defines derivatives to include "models trained on outputs from the UNI model
or datasets created from the UNI model", which covers this study's adapters and extracted
embeddings by the licensor's own statement. MONET is withheld pending the CC-BY-NC-SA question in
D-046.
