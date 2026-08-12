# Backbone readiness tracker

Last updated: 2026-08-12 (post pin pass)

| Backbone | Status | Gate-0 |
|----------|--------|--------|
| PanDerm | Done | — |
| ResNet-50 | Done | — |
| EfficientNet-B3 | Done | — |
| UNI | Running | sequential rerun |
| MONET | Ready | preprocessing + SHA256 |
| BiomedCLIP | Ready | preprocessing + SHA256 |
| OpenCLIP | Ready | preprocessing + SHA256 |
| SigLIP | Ready | preprocessing + SHA256 |
| MoCo v3 | Ready (D-042) | safetensors_hub + preprocessing SHA256 |
| MedSAM | Ready | wanglab checkpoint + preprocessing SHA256 |
| **DINOv3** | **Deferred** | HF gating pending Meta approval |

## DINOv3

Uncomment `configs/dinov3.yaml` in `scripts/production_queue.txt` after HF access approved.

## MoCo v3 (D-042)

See `docs/decisions/D-042-mocov3-provenance.md`.

## Production kick

Waiter (`wait_and_run_production.sh`) auto-starts queue with `MAX_CPU=2` when UNI finishes.
