#!/usr/bin/env bash
# Launch the production GPU/CPU scheduler for remaining backbones.
#
# Production-first: start with MAX_CPU=2, read scheduler metrics.csv after 2–3
# backbones, then restart remaining queue with MAX_CPU=3/4 if CPU headroom exists.
# Separate benchmark runs are optional (scheduler debugging only).
#
# Usage:
#   export CSG_DATA_ROOT=/path/to/CSG-Skin/data
#   nohup ./scripts/run_production_queue.sh > results/logs/production_queue.log 2>&1 &
#
# Hands-off after UNI (recommended):
#   nohup ./scripts/wait_and_run_production.sh > results/logs/wait_production.log 2>&1 &
#
# Env:
#   MAX_CPU=2              Default conservative pool (override after telemetry)
#   WAIT_FOR_GPU=1         Block until external GPU job finishes (default 1)
#   POLL_SEC=60            Wait poll when GPU busy
#   RETRY_FAILED=1         Retry backbones listed in results/scheduler/failures.jsonl
#   QUEUE_FILE             Override queue file path

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QUEUE_FILE="${QUEUE_FILE:-$ROOT/scripts/production_queue.txt}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
POLL_SEC="${POLL_SEC:-60}"
MAX_CPU="${MAX_CPU:-2}"

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

./scripts/gpu_cpu_scheduler.sh "${CONFIGS[@]}" 2>&1 | tee -a "$LOG"
exit_code="${PIPESTATUS[0]}"
if ((exit_code != 0)); then
  log "scheduler finished with failures (exit=$exit_code) — see results/scheduler/failures.jsonl"
  log "passed backbones kept; re-run with RETRY_FAILED=1 to retry failed only"
fi
exit "$exit_code"
