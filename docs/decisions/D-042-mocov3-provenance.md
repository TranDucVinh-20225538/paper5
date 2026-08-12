# D-042 — MoCo v3 checkpoint provenance

**Status:** CLOSED (2026-08-12)

## Decision

Study backbone **MoCo v3 ViT-B/16** uses weights from Hugging Face mirror
`nyu-visionx/moco-v3-vit-b` (revision `7d091cd70772c5c0ecf7f00b5f12ca609a99d69d`,
`model.safetensors` SHA256 `345a04069364250ef448402363443ae8d2da68678e6a88239a8906d2325b912e`).

Architecture: `timm` `vit_base_patch16_224`, `num_classes=0`, CLS pooling.

## Canonical FAIR release

| Field | Value |
|-------|-------|
| Repository | https://github.com/facebookresearch/moco-v3 |
| Official weights | `vit-b-300ep.pth` |
| URL | https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep.pth |

Direct download from `dl.fbaipublicfiles.com` returned **HTTP 403** from the study server
(2026-08-12). Byte-level hash comparison against the `.pth` file was therefore not performed
on this host.

## Mirror verification performed

1. Loaded `model.safetensors` from `nyu-visionx/moco-v3-vit-b` into `timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)`.
2. `load_state_dict(strict=False)` → **0 missing keys**, 2 unexpected (`head.weight`, `head.bias` — discarded with `num_classes=0`).
3. Output dimension **768**, matching config `embed_dim`.

The mirror is widely used to distribute FAIR MoCo v3 ViT-B weights in Hugging Face / safetensors
format. Remaining provenance risk: format conversion (`.pth` → safetensors) not byte-verified here
due to FAIR CDN block; accepted for production with pinned revision + weights SHA256.

## Rejected alternatives

| Source | Reason |
|--------|--------|
| `facebook/moco-v3-vit-b` (HF) | 404 — no official FAIR HF repo |
| DINOv2 as SSL partner | Rejected by D-015 (same-lineage) |

## Config pins

- `configs/mocov3.yaml` — checkpoint + `loader_kwargs.safetensors_hub`
- `assets/preprocessing/mocov3.json` — eval transform + SHA256
