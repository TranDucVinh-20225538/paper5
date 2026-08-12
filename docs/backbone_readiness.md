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
| **DINOv3** | Ready | HF approved; preprocessing + SHA256 pinned |

## DINOv3

Production queue includes all 7 remaining backbones (DINOv3 re-enabled 2026-08-12).

See `docs/decisions/D-042-mocov3-provenance.md`.

## Production kick

Waiter (`wait_and_run_production.sh`) auto-starts queue with `MAX_CPU=2` when UNI finishes.
