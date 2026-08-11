# results/

    csv/       tabular results, tracked
    figures/   generated figures, tracked
    logs/      run logs, gitignored
    manifest.jsonl   one line per run, tracked  <- the provenance record

## manifest.jsonl

Without this, `scripts/reproduce.sh` is a promise with nothing behind it. One JSON object per run,
appended, never rewritten:

```json
{
  "run_id": "medsam-canonical-s42-a1.00",
  "utc": "2026-08-11T09:14:22Z",
  "commit": "a3f9c21",
  "config_sha256": "…",
  "backbone": "medsam",
  "arm": "canonical",
  "seed": 42,
  "alpha": 1.0,
  "r": 16,
  "lambda_proj": 0.1,
  "gate0": "pass",
  "gate1": "pass",
  "embedding_sha256": "…",
  "output_sha256": "…",
  "env": {"python": "3.11.8", "torch": "2.4.0", "cuda": "12.1"}
}
```

`commit` and `config_sha256` are the two that matter. A result whose commit is unknown, or was
produced from a dirty tree, cannot be reproduced and should be regenerated rather than reported —
record `"dirty": true` if the working tree was not clean, so this is detectable later.

## What is not tracked

Raw embedding arrays and adapter checkpoints. Tens of GB across the backbone set — see `.gitignore`
for the arithmetic. Their **checksums** are tracked here, which is what reproducibility actually
requires.
