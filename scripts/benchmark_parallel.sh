#!/usr/bin/env bash
# Benchmark the GPU/CPU scheduler with 2 (then optionally 3) backbones in parallel.
#
# Phase 1 (default): MAX_CPU=2, 2 backbones — measure overlap vs sequential ResNet-50 baseline.
# Phase 2: MAX_CPU=3, 3 backbones — only if Phase-1 metrics show GPU util >70% and CPU headroom.
#
# Usage:
#   export CSG_DATA_ROOT=/path/to/CSG-Skin/data
#   export SCHEDULER_BACKBONES="configs/monet.yaml configs/biomedclip.yaml"
#   nohup ./scripts/benchmark_parallel.sh > results/logs/benchmark_parallel.log 2>&1 &
#
# Env:
#   BENCHMARK_PHASE=1|2     Phase 1: 2 backbones / MAX_CPU=2 (default). Phase 2: 3 / MAX_CPU=3.
#   SCHEDULER_BACKBONES     Space-separated config paths (required unless BENCHMARK_PHASE=2 adds a third)
#   SKIP_IF_GPU_BUSY=1      Abort if another pipeline.cli holds the GPU (default 1)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PHASE="${BENCHMARK_PHASE:-1}"
SKIP_IF_GPU_BUSY="${SKIP_IF_GPU_BUSY:-1}"

if [[ -z "${CSG_DATA_ROOT:-}" ]]; then
  echo "ERROR: CSG_DATA_ROOT must be set." >&2
  exit 1
fi

if [[ -z "${SCHEDULER_BACKBONES:-}" ]]; then
  SCHEDULER_BACKBONES="configs/resnet50.yaml configs/efficientnet_b3.yaml"
fi

read -r -a BACKBONES <<< "$SCHEDULER_BACKBONES"

if (("$PHASE" == 1)); then
  MAX_CPU=2
  if ((${#BACKBONES[@]} < 2)); then
    echo "ERROR: Phase 1 requires at least 2 backbones in SCHEDULER_BACKBONES." >&2
    exit 1
  fi
  BACKBONES=("${BACKBONES[0]}" "${BACKBONES[1]}")
elif (("$PHASE" == 2)); then
  MAX_CPU=3
  if ((${#BACKBONES[@]} < 3)); then
    echo "ERROR: Phase 2 requires 3 backbones in SCHEDULER_BACKBONES." >&2
    exit 1
  fi
  BACKBONES=("${BACKBONES[0]}" "${BACKBONES[1]}" "${BACKBONES[2]}")
else
  echo "ERROR: BENCHMARK_PHASE must be 1 or 2." >&2
  exit 1
fi

if (("$SKIP_IF_GPU_BUSY" == 1)) && command -v nvidia-smi >/dev/null 2>&1; then
  gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -n "$gpu_util" && "$gpu_util" -gt 10 ]]; then
    other="$(pgrep -af 'pipeline.cli' 2>/dev/null | grep -v benchmark_parallel || true)"
    if [[ -n "$other" ]]; then
      echo "WARN: GPU util ${gpu_util}% and pipeline.cli already running:" >&2
      echo "$other" >&2
      echo "Stop sequential jobs before benchmarking, or set SKIP_IF_GPU_BUSY=0." >&2
      exit 1
    fi
  fi
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BENCH_LOG="$ROOT/results/logs/benchmark_${STAMP}.log"

{
  echo "=== benchmark phase $PHASE start $STAMP ==="
  echo "backbones: ${BACKBONES[*]}"
  echo "MAX_CPU=$MAX_CPU"
  echo "baseline (sequential ResNet-50): ~624 min wall / ~10.4 h (2026-08-12 rerun)"
  echo ""
} | tee -a "$BENCH_LOG"

START_EPOCH="$(date +%s)"
export MAX_CPU CPU_THREADS="${CPU_THREADS:-16}"
./scripts/gpu_cpu_scheduler.sh "${BACKBONES[@]}" 2>&1 | tee -a "$BENCH_LOG"
END_EPOCH="$(date +%s)"
WALL_MIN="$(( (END_EPOCH - START_EPOCH) / 60 ))"

SCHED_DIR="$(ls -td "$ROOT/results/logs/scheduler_"* 2>/dev/null | head -1)"
SUMMARY="$ROOT/results/logs/benchmark_${STAMP}_summary.txt"

{
  echo "=== benchmark phase $PHASE summary ==="
  echo "wall_clock_min: $WALL_MIN"
  echo "backbones: ${BACKBONES[*]}"
  echo "scheduler_log_dir: ${SCHED_DIR:-none}"
  if [[ -f "${SCHED_DIR}/metrics.csv" ]]; then
    echo ""
    echo "--- avg GPU util (%) ---"
    awk -F, 'NR>1 && $2!="na" {s+=$2; n++} END {if(n) printf "%.1f\n", s/n; else print "na"}' "${SCHED_DIR}/metrics.csv"
    echo "--- peak concurrent CPU jobs ---"
    awk -F, 'NR>1 {if($5>m) m=$5} END {print m+0}' "${SCHED_DIR}/metrics.csv"
    echo "--- metrics file ---"
    echo "${SCHED_DIR}/metrics.csv"
  fi
  echo ""
  echo "Decision rubric:"
  echo "  Phase-1 PASS → GPU util mostly >50% during GPU phases, MAX concurrent CPU jobs = 2,"
  echo "                 wall time for 2 backbones < 2× single-backbone sequential."
  echo "  Then try: BENCHMARK_PHASE=2 MAX_CPU=3 with 3 ready backbones."
} | tee "$SUMMARY" >>"$BENCH_LOG"

echo "Summary written to $SUMMARY"
