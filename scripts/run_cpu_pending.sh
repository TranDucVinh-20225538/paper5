#!/usr/bin/env bash
# Run CPU phase (Steps 8–12) for every backbone at gpu_done.
#
#   ./scripts/run_cpu_pending.sh scripts/production_queue.txt
#   ./scripts/run_cpu_pending.sh configs/monet.yaml configs/openclip.yaml

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QUEUE=("$@")
if ((${#QUEUE[@]} == 0)); then
  echo "usage: run_cpu_pending.sh configs/a.yaml [configs/b.yaml ...]" >&2
  exit 1
fi

COMMON=(--skip-split-check)
LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

for cfg in "${QUEUE[@]}"; do
  read -r name phase _rest <<< "$(python -m src.utils.scheduler_state "$cfg")"
  if [[ "$phase" != "gpu_done" ]]; then
    echo "skip $name ($phase)"
    continue
  fi
  log="$LOG_DIR/${name}_cpu_${STAMP}.log"
  echo "=== CPU phase $name @ $(date -u +%H:%M:%S) ==="
  if ./scripts/run_cpu_phase.sh "$cfg" "${COMMON[@]}" >"$log" 2>&1; then
    echo "=== $name OK ==="
  else
    echo "=== $name FAILED (see $log) ==="
  fi
done

echo "CPU pending batch done $STAMP"
