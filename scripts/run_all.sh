#!/usr/bin/env bash
# Full ladder for one backbone. Steps mirror docs/protocol.md.
#
#   ./scripts/run_all.sh configs/panderm.yaml
#
# Run order (D-027): PanDerm first (regression), then MedSAM (portability probe).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:?usage: run_all.sh configs/<backbone>.yaml}"
THROUGH="${2:-3}"

cd "$ROOT"
python -m src.pipeline.cli "$CFG" --through-step "$THROUGH"
status=$?

if [[ "$THROUGH" -lt 4 ]]; then
  echo
  echo "Steps 4–12 not implemented yet (Milestone 3+)."
  echo "  4  nuisance direction"
  echo "  5  grid search r x lambda_proj"
  echo "  6  Gate 0"
  echo "  7  train adapter x 3 arms x 5 seeds"
  echo "  8  alpha ladder"
  echo "  9  Gate 1"
  echo " 10  geometry metrics"
  echo " 11  reliability estimators"
  echo " 12  manifest (partial — Step 3 appends extract record)"
fi

exit "$status"
