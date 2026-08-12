#!/usr/bin/env bash
# Launch the production GPU/CPU scheduler for remaining backbones.
#
# Waits until no external GPU pipeline job is running (e.g. UNI rerun), then
# starts gpu_cpu_scheduler.sh with configs from production_queue.txt.
#
# Usage:
#   export CSG_DATA_ROOT=/path/to/CSG-Skin/data
#   nohup ./scripts/run_production_queue.sh > results/logs/production_queue.log 2>&1 &
#
# Env:
#   MAX_CPU                CPU worker pool size (required — set from benchmark data)
#   WAIT_FOR_GPU=1         Block until external GPU job finishes (default 1)
#   POLL_SEC=60            Wait poll when GPU busy
#   QUEUE_FILE             Override queue file path

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QUEUE_FILE="${QUEUE_FILE:-$ROOT/scripts/production_queue.txt}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
POLL_SEC="${POLL_SEC:-60}"

if [[ -z "${CSG_DATA_ROOT:-}" ]]; then
  echo "ERROR: CSG_DATA_ROOT must be set." >&2
  exit 1
fi

if [[ ! -f "$QUEUE_FILE" ]]; then
  echo "ERROR: queue file not found: $QUEUE_FILE" >&2
  exit 1
fi

mapfile -t CONFIGS < <(grep -E '^configs/.*\.yaml$' "$QUEUE_FILE" || true)
if ((${#CONFIGS[@]} == 0)); then
  echo "ERROR: no configs/*.yaml entries in $QUEUE_FILE" >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/results/logs/production_queue_${STAMP}.log"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

external_gpu_running() {
  local line pid cmd
  while IFS= read -r line; do
    cmd="$(echo "$line" | sed 's/^[0-9]* //')"
    if [[ "$cmd" == *"src.pipeline.cli"* && "$cmd" != *"--from-step 8"* ]]; then
      return 0
    fi
  done < <(pgrep -af 'src\.pipeline\.cli' 2>/dev/null || true)
  return 1
}

log "=== production queue launcher $STAMP ==="
log "configs (${#CONFIGS[@]}): ${CONFIGS[*]}"
: "${MAX_CPU:?Set MAX_CPU from benchmark results before production run}"
log "MAX_CPU=$MAX_CPU WAIT_FOR_GPU=$WAIT_FOR_GPU"

if (("$WAIT_FOR_GPU" == 1)); then
  while external_gpu_running; do
    log "waiting for external GPU job to finish (e.g. UNI)..."
    sleep "$POLL_SEC"
  done
  log "GPU lane free — starting scheduler"
fi

export MAX_CPU
export RESPECT_EXTERNAL_GPU=1
export RESUME=1

exec ./scripts/gpu_cpu_scheduler.sh "${CONFIGS[@]}" 2>&1 | tee -a "$LOG"
