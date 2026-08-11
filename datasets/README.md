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

Fill in per machine. Do not hardcode paths in `src/` — read them from an env var or a local,
gitignored path file.

    ISIC 2019:      <TODO>
    PAD-UFES-20:    <TODO>
    Embeddings:     <TODO — outside git, see .gitignore for the size argument>
