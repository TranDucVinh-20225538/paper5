#!/usr/bin/env bash
# Full ladder for one backbone. Steps mirror docs/protocol.md.
#
#   ./scripts/run_all.sh configs/panderm.yaml
#
# Run order (D-027): PanDerm first (regression), then MedSAM (portability probe).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:?usage: run_all.sh configs/<backbone>.yaml}"
THROUGH="${2:-6}"

cd "$ROOT"
python -m src.pipeline.cli "$CFG" --through-step "$THROUGH"
status=$?

if [[ "$THROUGH" -lt 7 ]]; then
  echo
  echo "Steps 7–12 not implemented yet (Milestone 4)."
  echo "  7  train adapter x 3 arms x 5 seeds"
  echo "  8  alpha ladder"
  echo "  9  Gate 1"
  echo " 10  geometry metrics"
  echo " 11  reliability estimators"
  echo " 12  manifest (partial — Step 3/6 append records)"
fi

exit "$status"
