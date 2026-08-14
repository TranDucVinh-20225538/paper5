#!/usr/bin/env bash
# GPU phase: Steps 0–7 (extract, grid, train arms).
#
#   export CSG_DATA_ROOT=/path/to/CSG-Skin/data
#   ./scripts/run_gpu_phase.sh configs/medsam.yaml
#
# Extra CLI flags after config are forwarded (e.g. --skip-split-check).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:?usage: run_gpu_phase.sh configs/<backbone>.yaml [-- extra flags]}"
shift || true
EXTRA=("$@")

if [[ -z "${CSG_DATA_ROOT:-}" ]]; then
  echo "ERROR: CSG_DATA_ROOT must be set for extraction." >&2
  exit 1
fi

cd "$ROOT"
exec python -m src.pipeline.cli "$CFG" --through-step 7 "${EXTRA[@]}"
