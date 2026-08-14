# geometry-reliability-generalization

**Paper 5** — multi-backbone causal replication of a geometry-mediated account of
distance-based reliability-estimator behaviour, in dermatology image classification under
acquisition shift.

| | |
|---|---|
| **Local path** | `/Users/cubo/Research/geometry-reliability-generalization/` |
| **GitHub** | https://github.com/TranDucVinh-20225538/paper5 |
| **Latest tag** | `p5-round0` |
| **Protocol frozen** | `a83c74c` · tag `v0.2-protocol-frozen` |

**Status: analysis complete, manuscript round 0 written.**

The protocol was frozen before any cross-backbone association was computed. All stop conditions are
closed. Ten backbones passed both gates; one declared probe (MedSAM) is recorded *not testable*.
The confirmatory result is Outcome D, heterogeneous — no backbone cleared the Holm-corrected
threshold — alongside an exact replication of the prior study on its original backbone
(τ = +0.5576, difference 0.0000).

The confirmatory analysis is reproducible from version-controlled inputs:
`scripts/run_confirmatory_analysis.py` implements the frozen specification and regenerates every
number in the manuscript from the per-backbone intermediate outputs. It was written independently of
the original run, which was lost before reaching any branch, and reproduces its reported table
exactly (D-053).

## Start here

[**`ONE_PAGE_SUMMARY.md`**](ONE_PAGE_SUMMARY.md) — the anchor. Research question, gap, hypothesis,
intervention, outcomes, stop conditions, success criteria. Every other document is downstream of it.
A pre-commit hook blocks edits to it that do not go through a decision ID.

## Repository layout

```
ONE_PAGE_SUMMARY.md      the anchor — read first, hook-protected
docs/                    kickoff, brief, protocol, preregistration, power analysis
configs/                 one YAML per backbone; things that get tuned
assets/                  frozen, hashed specs; things that must never be tuned
datasets/                pointers and checksums only — no data in git
src/                     backbone / intervention / geometry / estimators / statistics
experiments/             one directory per backbone, named not numbered
results/                 csv, figures, logs + manifest.jsonl
paper/                   manuscript, supplementary, reviewer_response, history
scripts/                 run_all.sh, analyze.sh, reproduce.sh
```

**Decisions do not live here.** They live one tier up, in
[`../lab-notebook/`](../lab-notebook/README.md), because the log's value is cumulative and scoping it
to one paper would reset it at every new paper (D-009).

## Four conventions worth knowing before you touch anything

**`configs/` and `assets/` are different in kind.** A config holds values that get searched over
(`r`, `λ_proj`). An asset holds a spec that is frozen and hashed and must never be adjusted
(preprocessing transforms, the dataset split). Mixing them invites someone to "just tweak" something
that is supposed to be immutable.

**`experiments/` uses names, not numbers.** `experiments/panderm/`, not `experiments/backbone01/`.
The backbone list churned during design — two family cells were resolved late (D-014, D-015) — so
numbered directories would have broken every reference in the decision log.

**Large arrays never enter git.** Embeddings run to tens of GB across the backbone set. The repo
tracks checksums and JSON/CSV summaries; the arrays live outside. See
[`datasets/README.md`](datasets/README.md) and [`results/README.md`](results/README.md).

**Manuscript rounds are git tags, not copied folders.** Tag the source (`p5-round0`, `p5-round1`);
archive the built PDF in `paper/history/`. Duplicating editable sources means edits eventually land
in the wrong copy. See [`paper/history/README.md`](paper/history/README.md).

## Setup

```bash
git config core.hooksPath .githooks
```

Required. Without it the anchor guard does not run.

```bash
conda env create -f environment.yml && conda activate geomrel
```

## Provenance

Every run appends one line to `results/manifest.jsonl` recording commit SHA, config hash, backbone,
arm, seed, and output checksums. Without it `scripts/reproduce.sh` is a promise with nothing behind
it.

## Lineage

| | |
|---|---|
| Paper 1 | Mahalanobis robust to acquisition shift (AUROC 0.97) where softmax confidence collapses |
| Paper 2 (CSG-Skin) | after shortcut removal, the same Mahalanobis AUROC collapses to ≈0.40 — the robustness may have been shortcut-driven |
| Paper 3 (DST-Skin) | the collapse is not information loss: a probe still decodes domain at 0.72–0.81 from embeddings that defeat 8 distance/density scorers |
| Paper 4 | preregistered causal test on one frozen foundation model (PanDerm ViT-L/16). Condition number is the **only** Holm-significant geometry metric; LID and spectral decay are not |
| **Paper 5** | Paper 4's own Prediction 3 — dropped for compute, not logic — revived as a multi-backbone causal replication (`F-003`) |

## Licensing and artifact release

Release is **stratified by checkpoint licence** (D-046), not uniform. Three of the ten backbones are
non-commercial, so the study's artifacts are non-commercial overall regardless of what the code
licence says.

| Tier | Backbones | Derived artifacts |
|---|---|---|
| **A** unrestricted | ResNet-50, EfficientNet-B3, BiomedCLIP, OpenCLIP, SigLIP, MedSAM | released |
| **B** conditional | DINOv3 (notice), MoCo v3 (non-commercial, attribution) | released with conditions |
| **C** withheld | UNI (no derivatives), MONET (**under legal review**) | checksums only |

See [`LICENSE`](LICENSE). The code licence is not yet chosen — it is blocked on the MONET question
in D-046, not on indecision.
