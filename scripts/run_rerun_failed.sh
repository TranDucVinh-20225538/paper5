#!/usr/bin/env bash
# Rerun failed overnight backbones (ResNet-50, UNI). EfficientNet-B3 already PASS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_LOG="$LOG_DIR/rerun_failed_${STAMP}.log"
COMMON_FLAGS=(--skip-split-check)

BACKBONES=(
  configs/resnet50.yaml
  configs/uni.yaml
)

echo "=== rerun failed start $STAMP ===" | tee -a "$BATCH_LOG"
echo "CSG_DATA_ROOT=${CSG_DATA_ROOT:-unset}" | tee -a "$BATCH_LOG"

if [[ -z "${CSG_DATA_ROOT:-}" ]]; then
  echo "ERROR: CSG_DATA_ROOT must be set." | tee -a "$BATCH_LOG"
  exit 1
fi

FAILED=()
for cfg in "${BACKBONES[@]}"; do
  name="$(basename "$cfg" .yaml)"
  log="$LOG_DIR/${name}_ladder_${STAMP}.log"
  echo "--- starting $name @ $(date -u +%H:%M:%S) ---" | tee -a "$BATCH_LOG"
  if ./scripts/run_all.sh "$cfg" 12 "${COMMON_FLAGS[@]}" >"$log" 2>&1; then
    echo "--- $name OK ---" | tee -a "$BATCH_LOG"
  else
    code=$?
    echo "--- $name FAILED exit=$code (see $log) ---" | tee -a "$BATCH_LOG"
    FAILED+=("$name")
  fi
done

echo "=== rerun failed done $(date -u +%Y%m%dT%H%M%SZ) ===" | tee -a "$BATCH_LOG"
if ((${#FAILED[@]})); then
  echo "failed: ${FAILED[*]}" | tee -a "$BATCH_LOG"
  exit 1
fi
echo "all reruns passed Gate 1" | tee -a "$BATCH_LOG"
