#!/usr/bin/env bash
# Re-run a recorded result from results/manifest.jsonl and verify it matches.
#
#   ./scripts/reproduce.sh medsam-canonical-s42-a1.00

set -euo pipefail
RUN_ID="${1:?usage: reproduce.sh <run_id>}"

echo "!! NOT IMPLEMENTED — scaffold only."
echo
echo "  1  look up $RUN_ID in results/manifest.jsonl"
echo "  2  git checkout the recorded commit  (refuse if it was dirty)"
echo "  3  verify config_sha256 and embedding_sha256"
echo "  4  re-run with the recorded seed"
echo "  5  diff output_sha256 — report, do not overwrite"
exit 1
