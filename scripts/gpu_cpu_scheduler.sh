#!/usr/bin/env bash
# Production GPU/CPU pipeline scheduler.
#
# GPU lane (MAX_GPU=1): Steps 0–7 — extract, Gate-0, train arms.
# CPU lane (MAX_CPU):   Steps 8–12 — alpha ladder, geometry, reliability.
#
# Features:
#   - flock GPU lock (results/locks/gpu.lock) — one GPU consumer at a time
#   - per-backbone locks — prevent duplicate launches
#   - manifest-based resume — skip done / resume CPU-only after GPU phase
#   - RESPECT_EXTERNAL_GPU — wait while non-scheduler pipeline.cli holds GPU (e.g. UNI rerun)
#   - monitoring CSV: wall time, GPU/CPU util, RAM, queue, ETA
#   - DRY_RUN=1 — fixture timing simulation (no GPU/CPU pipeline)
#
# Usage:
#   export CSG_DATA_ROOT=/path/to/CSG-Skin/data
#   ./scripts/gpu_cpu_scheduler.sh configs/monet.yaml configs/biomedclip.yaml
#
# Env:
#   MAX_GPU=1              GPU slots (default 1)
#   MAX_CPU=2              Concurrent CPU analysis jobs
#   CPU_THREADS=16         Threads per CPU job
#   CUDA_VISIBLE_DEVICES=0
#   POLL_SEC=30            Scheduler poll interval
#   COMMON_FLAGS           Extra CLI flags (default: --skip-split-check)
#   RESPECT_EXTERNAL_GPU=1 Wait if another pipeline.cli is running (default 1)
#   DRY_RUN=0              Set 1 for simulation (tests only)
#   BASELINE_GPU_MIN=159   ETA: minutes per GPU phase (ResNet50 benchmark)
#   BASELINE_CPU_MIN=465   ETA: minutes per CPU phase
#   LOCK_DIR               Override lock directory (default results/locks)
#   STATE_FILE             Persistent state for resume (default results/scheduler/state.json)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if (("$#" < 1)); then
  echo "usage: gpu_cpu_scheduler.sh configs/<a>.yaml [configs/<b>.yaml ...]" >&2
  exit 1
fi

if [[ -z "${CSG_DATA_ROOT:-}" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "ERROR: CSG_DATA_ROOT must be set (or DRY_RUN=1)." >&2
  exit 1
fi

MAX_GPU="${MAX_GPU:-1}"
MAX_CPU="${MAX_CPU:-2}"
CPU_THREADS="${CPU_THREADS:-16}"
GPU_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
POLL_SEC="${POLL_SEC:-30}"
RESPECT_EXTERNAL_GPU="${RESPECT_EXTERNAL_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"
BASELINE_GPU_MIN="${BASELINE_GPU_MIN:-159}"
BASELINE_CPU_MIN="${BASELINE_CPU_MIN:-465}"
read -r -a COMMON <<< "${COMMON_FLAGS:---skip-split-check}"

LOCK_DIR="${LOCK_DIR:-$ROOT/results/locks}"
GPU_LOCK="$LOCK_DIR/gpu.lock"
STATE_FILE="${STATE_FILE:-$ROOT/results/scheduler/state.json}"
mkdir -p "$LOCK_DIR" "$(dirname "$STATE_FILE")"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -f "$STATE_FILE" && "${RESUME:-1}" == "1" ]]; then
  prev_log="$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('log_dir',''))" 2>/dev/null || true)"
  if [[ -n "$prev_log" && -d "$prev_log" && -f "$prev_log/.active" ]]; then
    LOG_DIR="$prev_log"
    STAMP="$(basename "$LOG_DIR" | sed 's/^scheduler_//')"
    echo "Resuming scheduler session $STAMP" >&2
  else
    LOG_DIR="$ROOT/results/logs/scheduler_${STAMP}"
  fi
else
  LOG_DIR="$ROOT/results/logs/scheduler_${STAMP}"
fi
mkdir -p "$LOG_DIR"
touch "$LOG_DIR/.active"
SCHED_LOG="$LOG_DIR/scheduler.log"
METRICS_CSV="$LOG_DIR/metrics.csv"
SCHEDULER_PID="$$"

QUEUE=("$@")
declare -A BACKBONE_PHASE=()
declare -A CPU_PIDS=()
declare -A CPU_START_EPOCH=()
declare -A GPU_START_EPOCH=()
declare -A DONE_BACKBONES=()
declare -A FAILED_BACKBONES=()
GPU_PID=""
GPU_CFG=""
GPU_BACKBONE=""
GPU_LOCK_FD=""
queue_idx=0
SCHED_START_EPOCH="$(date +%s)"

# --- helpers ---

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$SCHED_LOG"; }

cfg_name() { basename "$1" .yaml; }

backbone_phase() {
  python3 -m src.utils.scheduler_state "$1" 2>/dev/null | awk -F'\t' '{print $2}'
}

init_queue_phases() {
  local cfg name phase
  local -a pending=() cpu_ready=() skipped=()
  for cfg in "${QUEUE[@]}"; do
    name="$(cfg_name "$cfg")"
    phase="$(backbone_phase "$cfg")"
    BACKBONE_PHASE["$name"]="$phase"
    case "$phase" in
      done)
        DONE_BACKBONES["$name"]=1
        skipped+=("$name")
        log "skip $name (already done)"
        ;;
      gpu_done)
        cpu_ready+=("$cfg")
        log "resume $name at CPU phase (GPU artifacts on disk)"
        ;;
      failed)
        FAILED_BACKBONES["$name"]=1
        log "skip $name (manifest: failed)"
        ;;
      pending)
        pending+=("$cfg")
        ;;
    esac
  done
  # Rebuild queue: pending GPU jobs first, then track cpu_ready separately
  QUEUE=("${pending[@]}")
  CPU_READY_QUEUE=("${cpu_ready[@]}")
  log "queue pending=${#QUEUE[@]} cpu_ready=${#CPU_READY_QUEUE[@]} done=${#skipped[@]}"
}

backbone_lock_path() { echo "$LOCK_DIR/backbone_$(cfg_name "$1").lock"; }

try_backbone_lock() {
  local cfg=$1
  local lock
  lock="$(backbone_lock_path "$cfg")"
  exec {fd}>"$lock"
  if flock -n "$fd"; then
    echo "$fd"
    return 0
  fi
  exec {fd}>&-
  return 1
}

release_backbone_lock() {
  local fd=$1
  [[ -n "$fd" ]] && exec {fd}>&- || true
}

is_external_gpu_job() {
  [[ "$RESPECT_EXTERNAL_GPU" != "1" ]] && return 1
  local line pid cmd
  while IFS= read -r line; do
    pid="$(echo "$line" | awk '{print $1}')"
    [[ "$pid" == "$SCHEDULER_PID" ]] && continue
    [[ -n "${GPU_PID:-}" && "$pid" == "$GPU_PID" ]] && continue
    cmd="$(echo "$line" | sed 's/^[0-9]* //')"
    if [[ "$cmd" == *"src.pipeline.cli"* && "$cmd" != *"DRY_RUN"* ]]; then
      if [[ "$cmd" != *"--from-step 8"* ]]; then
        return 0
      fi
    fi
  done < <(pgrep -af 'src\.pipeline\.cli' 2>/dev/null || true)
  return 1
}

try_gpu_lock() {
  exec {GPU_LOCK_FD}>"$GPU_LOCK"
  if flock -n "$GPU_LOCK_FD"; then
    return 0
  fi
  exec {GPU_LOCK_FD}>&-
  GPU_LOCK_FD=""
  return 1
}

release_gpu_lock() {
  [[ -n "${GPU_LOCK_FD:-}" ]] && exec {GPU_LOCK_FD}>&- || true
  GPU_LOCK_FD=""
}

cpu_util_pct() {
  if command -v mpstat >/dev/null 2>&1; then
    mpstat 1 1 2>/dev/null | awk '/Average/ && NF>2 {printf "%.0f", 100-$NF; exit}'
    return
  fi
  if [[ -f /proc/stat ]]; then
    local u1 n1 s1 i1 iw1 irq1 si1 st1 g1 gn1
    read -r _ u1 n1 s1 i1 iw1 irq1 si1 st1 g1 gn1 _ < /proc/stat
    local t1=$((u1 + n1 + s1 + i1 + iw1 + irq1 + si1 + st1))
    sleep 1
    local u2 n2 s2 i2 iw2 irq2 si2 st2 g2 gn2
    read -r _ u2 n2 s2 i2 iw2 irq2 si2 st2 g2 gn2 _ < /proc/stat
    local t2=$((u2 + n2 + s2 + i2 + iw2 + irq2 + si2 + st2))
    local dt=$((t2 - t1))
    local di=$((i2 - i1))
    if ((dt > 0)); then
      echo $(( (100 * (dt - di)) / dt ))
    else
      echo "na"
    fi
    return
  fi
  echo "na"
}

ram_stats() {
  if [[ -f /proc/meminfo ]]; then
    local total avail
    total="$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)"
    avail="$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)"
    echo "$((total - avail)) $total"
  else
    echo "na na"
  fi
}

count_cpu_running() {
  local n=0 name
  for name in "${!CPU_PIDS[@]}"; do
    kill -0 "${CPU_PIDS[$name]}" 2>/dev/null && n=$((n + 1))
  done
  echo "$n"
}

eta_minutes() {
  local pending_gpu=${#QUEUE[@]}
  pending_gpu=$((pending_gpu - queue_idx))
  local running_cpu
  running_cpu="$(count_cpu_running)"
  local cpu_ready=${#CPU_READY_QUEUE[@]}
  local gpu_min=$((pending_gpu * BASELINE_GPU_MIN))
  if [[ -n "${GPU_PID:-}" ]]; then
    local elapsed=$(( $(date +%s) - GPU_START_EPOCH ))
    local rem=$((BASELINE_GPU_MIN * 60 - elapsed))
    ((rem < 0)) && rem=0
    gpu_min=$((rem / 60 + pending_gpu * BASELINE_GPU_MIN))
  fi
  local cpu_jobs=$((cpu_ready + pending_gpu + running_cpu))
  local cpu_min=$(( (cpu_jobs * BASELINE_CPU_MIN + MAX_CPU - 1) / MAX_CPU ))
  if ((gpu_min > cpu_min)); then echo "$gpu_min"; else echo "$cpu_min"; fi
}

write_state() {
  python3 - "$STATE_FILE" "$LOG_DIR" "$SCHEDULER_PID" <<'PY'
import json, sys
path, log_dir, pid = sys.argv[1:4]
data = {
    "log_dir": log_dir,
    "scheduler_pid": int(pid),
    "updated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
}
with open(path, "w") as f:
    json.dump(data, f, indent=2)
PY
}

export_results() {
  local name=$1
  local src="$ROOT/experiments/$name/alpha_ladder"
  local dst="$ROOT/results/csv/$name"
  if [[ -d "$src" ]]; then
    mkdir -p "$dst"
    cp "$src/"*.json "$dst/" 2>/dev/null || true
  fi
}

start_gpu() {
  local cfg=$1
  local name
  name="$(cfg_name "$cfg")"
  if [[ -n "${GPU_PID:-}" && "$GPU_BACKBONE" == "$name" ]]; then
    log "skip duplicate GPU launch $name"
    return 1
  fi
  for n in "${!CPU_PIDS[@]}"; do
    if [[ "$n" == "$name" ]] && kill -0 "${CPU_PIDS[$n]}" 2>/dev/null; then
      log "skip GPU launch $name (CPU phase already running)"
      return 1
    fi
  done
  if ! flock -n "$(backbone_lock_path "$cfg")" true 2>/dev/null; then
    log "skip GPU launch $name (backbone lock held)"
    return 1
  fi
  local log="$LOG_DIR/${name}_gpu.log"
  log "GPU phase start $name"
  GPU_START_EPOCH="$(date +%s)"
  if [[ "$DRY_RUN" == "1" ]]; then
    (
      sleep "${DRY_RUN_GPU_SEC:-2}"
      exit 0
    ) >"$log" 2>&1 &
  else
    (
      export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
      exec python -m src.pipeline.cli "$cfg" --through-step 7 "${COMMON[@]}"
    ) >"$log" 2>&1 &
  fi
  GPU_PID=$!
  GPU_CFG=$cfg
  GPU_BACKBONE=$name
}

start_cpu() {
  local cfg=$1
  local name
  name="$(cfg_name "$cfg")"
  local log="$LOG_DIR/${name}_cpu.log"
  log "CPU phase start $name"
  CPU_START_EPOCH["$name"]="$(date +%s)"
  if [[ "$DRY_RUN" == "1" ]]; then
    (
      sleep "${DRY_RUN_CPU_SEC:-3}"
      exit 0
    ) >"$log" 2>&1 &
  else
    (
      export OMP_NUM_THREADS="$CPU_THREADS"
      export MKL_NUM_THREADS="$CPU_THREADS"
      export OPENBLAS_NUM_THREADS="$CPU_THREADS"
      export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
      export NUMEXPR_NUM_THREADS="$CPU_THREADS"
      unset CUDA_VISIBLE_DEVICES
      exec python -m src.pipeline.cli "$cfg" --from-step 8 --through-step 12 "${COMMON[@]}"
    ) >"$log" 2>&1 &
  fi
  CPU_PIDS["$name"]=$!
}

try_start_cpu_from_ready() {
  local running
  running="$(count_cpu_running)"
  if (("$running" >= MAX_CPU)); then
    return 1
  fi
  if ((${#CPU_READY_QUEUE[@]} == 0)); then
    return 1
  fi
  local cfg="${CPU_READY_QUEUE[0]}"
  CPU_READY_QUEUE=("${CPU_READY_QUEUE[@]:1}")
  start_cpu "$cfg"
  return 0
}

reap_gpu() {
  [[ -z "${GPU_PID:-}" ]] && return 0
  if kill -0 "$GPU_PID" 2>/dev/null; then
    return 1
  fi
  local code=0
  wait "$GPU_PID" || code=$?
  local name="$GPU_BACKBONE"
  release_gpu_lock
  if ((code == 0)); then
    log "GPU phase OK $name"
    start_cpu "$GPU_CFG"
  else
    log "GPU phase FAILED $name exit=$code"
    FAILED_BACKBONES["$name"]=1
  fi
  GPU_PID=""
  GPU_CFG=""
  GPU_BACKBONE=""
  return 0
}

reap_cpu() {
  local name pid code
  for name in "${!CPU_PIDS[@]}"; do
    pid="${CPU_PIDS[$name]}"
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    code=0
    wait "$pid" || code=$?
    if ((code == 0)); then
      export_results "$name"
      DONE_BACKBONES["$name"]=1
      log "CPU phase OK $name"
    else
      log "CPU phase FAILED $name exit=$code"
      FAILED_BACKBONES["$name"]=1
    fi
    unset 'CPU_PIDS[$name]'
    unset 'CPU_START_EPOCH[$name]'
  done
}

metrics_header_written=0

metrics_loop() {
  if [[ ! -f "$METRICS_CSV" || ! -s "$METRICS_CSV" ]]; then
    echo "timestamp_utc,wall_sec,gpu_util_pct,gpu_mem_mib,cpu_util_pct,ram_used_gb,ram_total_gb,gpu_backbone,cpu_backbones,queue_pending,queue_done,failed_count,eta_min" >"$METRICS_CSV"
  fi
  while [[ -f "$LOG_DIR/.monitor" ]]; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    wall=$(( $(date +%s) - SCHED_START_EPOCH ))
    gpu_util="na"
    gpu_mem="na"
    if command -v nvidia-smi >/dev/null 2>&1; then
      read -r gpu_util gpu_mem _ <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr ',' ' ')"
    fi
    cpu_util="$(cpu_util_pct)"
    read -r ram_used ram_total <<< "$(ram_stats)"
    gpu_name="${GPU_BACKBONE:-idle}"
    cpu_list=""
    for n in "${!CPU_PIDS[@]}"; do
      kill -0 "${CPU_PIDS[$n]}" 2>/dev/null && cpu_list+="${n},"
    done
    cpu_list="${cpu_list%,}"
    q_pending=$((${#QUEUE[@]} - queue_idx))
    q_done=${#DONE_BACKBONES[@]}
    q_failed=${#FAILED_BACKBONES[@]}
    eta="$(eta_minutes)"
    echo "$ts,$wall,$gpu_util,$gpu_mem,$cpu_util,$ram_used,$ram_total,$gpu_name,$cpu_list,$q_pending,$q_done,$q_failed,$eta" >>"$METRICS_CSV"
    write_state
    sleep "$POLL_SEC"
  done
}

cleanup() {
  rm -f "$LOG_DIR/.monitor" "$LOG_DIR/.active"
  release_gpu_lock
  wait "${MONITOR_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

CPU_READY_QUEUE=()
init_queue_phases

touch "$LOG_DIR/.monitor"
metrics_loop &
MONITOR_PID=$!

log "=== scheduler start $STAMP ==="
log "queue: ${QUEUE[*]}"
log "MAX_GPU=$MAX_GPU MAX_CPU=$MAX_CPU CPU_THREADS=$CPU_THREADS GPU=$GPU_DEVICE DRY_RUN=$DRY_RUN"
log "RESPECT_EXTERNAL_GPU=$RESPECT_EXTERNAL_GPU BASELINE_GPU_MIN=$BASELINE_GPU_MIN BASELINE_CPU_MIN=$BASELINE_CPU_MIN"

# Bootstrap CPU-ready jobs from resume
while try_start_cpu_from_ready; do :; done

while true; do
  reap_gpu || true
  reap_cpu
  try_start_cpu_from_ready || true

  if [[ -z "${GPU_PID:-}" && "$queue_idx" -lt ${#QUEUE[@]} && "$MAX_GPU" -ge 1 ]]; then
    if is_external_gpu_job; then
      log "waiting: external GPU pipeline.cli running (RESPECT_EXTERNAL_GPU=1)"
    elif try_gpu_lock; then
      start_gpu "${QUEUE[$queue_idx]}"
      queue_idx=$((queue_idx + 1))
    else
      log "waiting: GPU lock held"
    fi
  fi

  running_cpu="$(count_cpu_running)"
  if [[ -z "${GPU_PID:-}" && "$queue_idx" -ge ${#QUEUE[@]} && "$running_cpu" -eq 0 && ${#CPU_READY_QUEUE[@]} -eq 0 ]]; then
    break
  fi

  sleep "$POLL_SEC"
done

rm -f "$LOG_DIR/.monitor"
wait "$MONITOR_PID" 2>/dev/null || true

log "=== scheduler done $(date -u +%Y%m%dT%H%M%SZ) wall=$(( $(date +%s) - SCHED_START_EPOCH ))s ==="
if ((${#FAILED_BACKBONES[@]})); then
  log "failed: ${!FAILED_BACKBONES[*]}"
  exit 1
fi
log "all backbones passed Gate 1"
log "metrics: $METRICS_CSV"
rm -f "$STATE_FILE"
