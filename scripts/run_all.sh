#!/usr/bin/env bash
# Full ladder for one backbone. Steps mirror docs/protocol.md.
#
#   ./scripts/run_all.sh configs/panderm.yaml
#
# Run order (D-027): PanDerm first (regression), then MedSAM (portability probe).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:?usage: run_all.sh configs/<backbone>.yaml [through_step] [-- extra cli flags]}"
shift
THROUGH="${1:-12}"
if [[ $# -ge 1 && "$1" =~ ^[0-9]+$ ]]; then
  shift
fi
EXTRA=("$@")

cd "$ROOT"
python -m src.pipeline.cli "$CFG" --through-step "$THROUGH" "${EXTRA[@]}"
status=$?

if [[ "$THROUGH" -lt 13 ]]; then
  echo
  echo "Cross-backbone analysis (post loop) is not implemented in run_all.sh."
fi

exit "$status"
