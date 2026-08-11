# datasets/

**No data in this directory.** Only pointers and checksums. `.gitignore` enforces it.

## Composition

| Source | n | Role |
|---|---|---|
| ISIC 2019 (dermoscopy) | 25,331 | in-distribution |
| PAD-UFES-20 (smartphone) | 2,298 | acquisition-shifted |
| | **27,629** | total |

## Split — seed 42, identical to Papers 1-4

| Partition | n | Use |
|---|---|---|
| ISIC train | 16,211 | fitting population |
| ISIC test (20%) | 5,067 | ID eval |
| ISIC val (20%) | 4,053 | unused |
| PAD-UFES-20 (full) | 2,298 | OOD |
| **eval pool** | **7,365** | 5,067 ID + 2,298 OOD |

**Regenerating this split would invalidate the comparison to Papers 1-4.** Verify by checksum, never
by re-running the split code — a library version change is enough to reorder it silently.

Store the manifest of assigned ids in `datasets/checksums/split_seed42.sha256` and check it in
Step 0 of the protocol. That file is small, deterministic, and belongs in git; the images do not.

## Where the actual data lives

Fill in per machine via **environment variables** or a gitignored local file — never hardcode in `src/`.

```bash
# Option A — environment (recommended on server)
export CSG_DATA_ROOT=/path/to/CSG-SKin/data
export PANDERM_EMBEDDINGS_ROOT=/path/to/Paper4/PhaseB/assets/reference_embeddings

# Option B — gitignored local file
cp datasets/paths.local.example datasets/paths.local
# edit paths.local
```

| Variable | Points to |
|---|---|
| `CSG_DATA_ROOT` or `CSG_ROOT` | `CSG-SKin/data/` — images + `master_metadata.csv` **only** |
| `PANDERM_EMBEDDINGS_ROOT` | Paper 4 frozen PanDerm embeddings (M3+) |
| `RESEARCH_ROOT` | Optional fallback: `$RESEARCH_ROOT/CSG-SKin/data` |

Split verification: `datasets/checksums/split_seed42.sha256` pins `master_metadata.csv` sha256 from Papers 1–4. When `CSG_DATA_ROOT` is set, Step 0 verifies the live file matches.

Typical layout under `CSG-SKin/data/`:

    master_metadata.csv
    ISIC_2019_Training_Input/
    pad_ufes20/
