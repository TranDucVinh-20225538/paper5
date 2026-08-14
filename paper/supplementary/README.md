# Supplementary

| | Contents | Source |
|---|---|---|
| **S1** | Secondary scorers (cosine-to-centroid, k-NN, KDE) per backbone | `results/csv/confirmatory/secondary_scorers.csv` |
| **S2** | Full-precision confirmatory table — τ, both p-values, Holm-adjusted p, AUROC range | `results/csv/confirmatory/per_backbone_confirmatory.csv` |
| **S3** | Sensitivity: κ_primary at k ∈ {128, 256, 512}; κ_paper4 | `results/csv/confirmatory/` |
| **S4** | Per-backbone gate outcomes, including the probe's twelve grid configurations | `results/csv/*/gate1_ea03.json`, `results/manifest.jsonl` |
| **S5** | Checkpoint revisions, library versions, split digest | `configs/*.yaml`, `results/manifest.jsonl` |

**Not yet buildable.** S1–S3 depend on `results/csv/confirmatory/`, which is not present on any
pushed branch — see D-053. The tables are specified here so the gap is visible rather than
discovered at submission.
