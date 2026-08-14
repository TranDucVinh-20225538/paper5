# Supplementary

| | Contents | Source |
|---|---|---|
| **S1** | Secondary scorers (cosine-to-centroid, k-NN, KDE) per backbone | `results/csv/confirmatory/secondary_scorers.csv` |
| **S2** | Full-precision confirmatory table — τ, both p-values, Holm-adjusted p, AUROC range | `results/csv/confirmatory/per_backbone_confirmatory.csv` |
| **S3** | Sensitivity: κ_primary at k ∈ {128, 256, 512}; κ_paper4 | `results/csv/confirmatory/` |
| **S4** | Per-backbone gate outcomes, including the probe's twelve grid configurations | `results/csv/*/gate1_ea03.json`, `results/manifest.jsonl` |
| **S5** | Checkpoint revisions, library versions, split digest | `configs/*.yaml`, `results/manifest.jsonl` |

All five are buildable from the repository. Regenerate S1–S3 with:

    python3 scripts/run_confirmatory_analysis.py
    python3 scripts/make_figures.py

Figures: `results/figures/fig1_confirmatory.{pdf,png}` (association per backbone; permutation versus
parametric) and `results/figures/fig2_auroc_range.{pdf,png}` (outcome variation per backbone).
