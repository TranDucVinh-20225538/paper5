#!/usr/bin/env bash
# Benchmark MAX_CPU settings — measure wall time, GPU idle, CPU utilization.
#
# Run a ladder (default 2 → 3 → 4) or a single value. Optimum MAX_CPU is chosen
# from data, not assumed.
#
# Usage:
#   export CSG_DATA_ROOT=/path/to/CSG-Skin/data
#   export MAX_CPU=3                                    # single run
#   export SCHEDULER_BACKBONES="configs/monet.yaml configs/biomedclip.yaml"
#   ./scripts/benchmark_parallel.sh
#
#   export BENCHMARK_CPU_LADDER="2 3 4 5"               # sequential sweep
#   ./scripts/benchmark_parallel.sh
#
# Env:
#   MAX_CPU                 Required for single run (omit when using BENCHMARK_CPU_LADDER)
#   BENCHMARK_CPU_LADDER    Space-separated values to sweep (default: "2 3 4")
#   SCHEDULER_BACKBONES     Space-separated config paths (required)
#   SKIP_IF_GPU_BUSY=1      Abort if another GPU pipeline.cli is running (default 1)
#   GPU_IDLE_THRESHOLD=10   GPU util % below this counts as idle (default 10)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SKIP_IF_GPU_BUSY="${SKIP_IF_GPU_BUSY:-1}"
GPU_IDLE_THRESHOLD="${GPU_IDLE_THRESHOLD:-10}"

if [[ -z "${CSG_DATA_ROOT:-}" ]]; then
  echo "ERROR: CSG_DATA_ROOT must be set." >&2
  exit 1
fi

if [[ -z "${SCHEDULER_BACKBONES:-}" ]]; then
  echo "ERROR: set SCHEDULER_BACKBONES=\"configs/a.yaml configs/b.yaml ...\"" >&2
  exit 1
fi

read -r -a BACKBONES <<< "$SCHEDULER_BACKBONES"

if ((${#BACKBONES[@]} < 2)); then
  echo "ERROR: benchmark needs at least 2 backbones in SCHEDULER_BACKBONES." >&2
  exit 1
fi

if (("$SKIP_IF_GPU_BUSY" == 1)) && command -v nvidia-smi >/dev/null 2>&1; then
  gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -n "$gpu_util" && "$gpu_util" -gt 10 ]]; then
    other="$(pgrep -af 'src\.pipeline\.cli' 2>/dev/null | grep -v benchmark_parallel || true)"
    if [[ -n "$other" ]]; then
      echo "WARN: GPU util ${gpu_util}% and pipeline.cli already running:" >&2
      echo "$other" >&2
      echo "Stop other GPU jobs before benchmarking, or set SKIP_IF_GPU_BUSY=0." >&2
      exit 1
    fi
  fi
fi

summarize_metrics() {
  local csv=$1
  [[ -f "$csv" ]] || { echo "metrics_missing"; return; }
  awk -F, -v idle_thr="$GPU_IDLE_THRESHOLD" '
    NR == 1 { next }
    {
      n++
      if ($3 != "na" && $3 != "") { gsu += $3; gn++ }
      if ($5 != "na" && $5 != "") { csu += $5; cn++ }
      if ($8 == "idle") idle_slot++
      if ($3 != "na" && $3 != "" && $3 + 0 < idle_thr) low_gpu++
      nb = $9
      if (nb == "") c = 0
      else {
        c = 1
        for (i = 1; i <= length(nb); i++) if (substr(nb, i, 1) == ",") c++
      }
      if (c > peak_cpu) peak_cpu = c
    }
    END {
      printf "samples=%d\n", n+0
      printf "avg_gpu_util_pct=%.1f\n", (gn ? gsu/gn : -1)
      printf "avg_cpu_util_pct=%.1f\n", (cn ? csu/cn : -1)
      printf "gpu_slot_idle_pct=%.1f\n", (n ? 100*idle_slot/n : -1)
      printf "gpu_low_util_pct=%.1f\n", (n ? 100*low_gpu/n : -1)
      printf "peak_concurrent_cpu_jobs=%d\n", peak_cpu+0
    }
  ' "$csv"
}

run_one_benchmark() {
  local max_cpu=$1
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local bench_log="$LADDER_DIR/benchmark_maxcpu${max_cpu}_${stamp}.log"
  local summary="$LADDER_DIR/max_cpu_${max_cpu}_summary.txt"

  {
    echo "=== benchmark MAX_CPU=$max_cpu start $stamp ==="
    echo "backbones: ${BACKBONES[*]}"
    echo "GPU_IDLE_THRESHOLD=$GPU_IDLE_THRESHOLD"
    echo "baseline (sequential single backbone): ~624 min wall (ResNet50 2026-08-12)"
    echo ""
  } | tee "$bench_log"

  local start_epoch end_epoch wall_min sched_dir
  start_epoch="$(date +%s)"
  export MAX_CPU="$max_cpu"
  export CPU_THREADS="${CPU_THREADS:-16}"
  ./scripts/gpu_cpu_scheduler.sh "${BACKBONES[@]}" 2>&1 | tee -a "$bench_log"
  end_epoch="$(date +%s)"
  wall_min="$(( (end_epoch - start_epoch) / 60 ))"

  sched_dir="$(ls -td "$ROOT/results/logs/scheduler_"* 2>/dev/null | head -1)"
  local metrics_csv="${sched_dir}/metrics.csv"

  {
    echo "=== MAX_CPU=$max_cpu summary ==="
    echo "wall_clock_min: $wall_min"
    echo "backbones: ${BACKBONES[*]}"
    echo "scheduler_log_dir: ${sched_dir:-none}"
    echo ""
    summarize_metrics "$metrics_csv"
    echo ""
    echo "metrics_file: ${metrics_csv:-none}"
  } | tee "$summary" >>"$bench_log"

  echo "$max_cpu,$wall_min,$(grep avg_gpu_util_pct "$summary" | cut -d= -f2),$(grep avg_cpu_util_pct "$summary" | cut -d= -f2),$(grep gpu_slot_idle_pct "$summary" | cut -d= -f2),$(grep gpu_low_util_pct "$summary" | cut -d= -f2),$(grep peak_concurrent_cpu_jobs "$summary" | cut -d= -f2),$summary" >>"$COMPARE_CSV"

  echo "Summary: $summary"
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LADDER_DIR="$ROOT/results/logs/benchmark_ladder_${STAMP}"
mkdir -p "$LADDER_DIR"
COMPARE_CSV="$LADDER_DIR/compare.csv"
echo "max_cpu,wall_min,avg_gpu_util_pct,avg_cpu_util_pct,gpu_slot_idle_pct,gpu_low_util_pct,peak_concurrent_cpu_jobs,summary_path" >"$COMPARE_CSV"

if [[ -n "${BENCHMARK_CPU_LADDER:-}" || -z "${MAX_CPU:-}" ]]; then
  LADDER="${BENCHMARK_CPU_LADDER:-2 3 4}"
  read -r -a CPU_VALUES <<< "$LADDER"
  echo "=== MAX_CPU ladder benchmark: ${CPU_VALUES[*]} ===" | tee "$LADDER_DIR/ladder.log"
  for v in "${CPU_VALUES[@]}"; do
    run_one_benchmark "$v"
  done
  echo ""
  echo "=== ladder comparison (pick MAX_CPU from data, do not assume) ==="
  column -t -s, "$COMPARE_CSV" 2>/dev/null || cat "$COMPARE_CSV"
  echo ""
  echo "Compare: $COMPARE_CSV"
else
  run_one_benchmark "$MAX_CPU"
  column -t -s, "$COMPARE_CSV" 2>/dev/null || cat "$COMPARE_CSV"
fi
