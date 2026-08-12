#!/usr/bin/env bash
# Wait for UNI (or any external GPU job) to finish, then start production queue.
#
# Production-first workflow: MAX_CPU=2 (conservative), tune from real scheduler
# metrics after 2–3 backbones — no separate benchmark run required.
#
# Usage (kick once, then hands-off):
#   export CSG_DATA_ROOT=/path/to/CSG-Skin/data
#   nohup ./scripts/wait_and_run_production.sh > results/logs/wait_production.log 2>&1 &
#
# Env:
#   MAX_CPU=2              Conservative default (override after reading metrics)
#   WAIT_FOR_GPU=1         Wait until GPU lane free (default 1)
#   POLL_SEC=60            Poll while waiting for UNI

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MAX_CPU="${MAX_CPU:-2}"
export WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
export POLL_SEC="${POLL_SEC:-60}"
export RESPECT_EXTERNAL_GPU=1
export RESUME=1

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/results/logs/wait_production_${STAMP}.log"

{
  echo "=== wait_and_run_production $STAMP ==="
  echo "MAX_CPU=$MAX_CPU (tune from scheduler metrics.csv after 2–3 backbones)"
  echo "queue: scripts/production_queue.txt"
  echo ""
} | tee "$LOG"

exec ./scripts/run_production_queue.sh 2>&1 | tee -a "$LOG"
