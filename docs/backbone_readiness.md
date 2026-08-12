# Backbone readiness tracker

Last updated: 2026-08-12

| Backbone | Strategic status | Repo / pipeline gate |
|----------|------------------|----------------------|
| PanDerm | Done | manifest step 12 |
| ResNet-50 | Done | manifest step 12 |
| EfficientNet-B3 | Done | manifest step 12 |
| UNI | Running | sequential rerun in progress |
| MONET | Ready | preprocessing + SHA256 frozen |
| BiomedCLIP | Ready | preprocessing + SHA256 frozen |
| OpenCLIP | Ready (checkpoint) | preprocessing SHA256 not pinned in repo |
| SigLIP | Ready (checkpoint) | preprocessing SHA256 not pinned in repo |
| MoCo v3 | Provenance (D-042) | checkpoint null; verify mirror vs FAIR release |
| MedSAM | Design locked | checkpoint/preprocessing still null in configs |
| **DINOv3** | **Blocked** | HF gating — pending Meta approval; cannot download |

## DINOv3

Model: `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, LVD-1689M pretrain).

Freeze requires: model ID, revision SHA, weights SHA256, preprocessing hash, embed_dim, loader — **plus** authenticated HF download.

When approved:

```bash
huggingface-cli login
# Fill revision + weights SHA256 in configs/dinov3.yaml after verify
# Uncomment configs/dinov3.yaml in scripts/production_queue.txt
export RETRY_FAILED=1 MAX_CPU=2
./scripts/run_production_queue.sh
```

## MoCo v3 (D-042)

Engineering task only: verify checkpoint provenance (official FAIR vs `nyu-visionx/moco-v3-vit-b` mirror), record revision + weights SHA256, freeze preprocessing. Does not block other backbones.

## MedSAM

Representation/pooling decisions in `docs/medsam_integration.md`. Remaining: pin checkpoint (e.g. `wanglab/medsam-vit-base`), finalize `assets/preprocessing/medsam.json` SHA256 in config.
