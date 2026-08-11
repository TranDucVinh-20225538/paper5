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
