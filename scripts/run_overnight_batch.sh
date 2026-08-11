#!/usr/bin/env bash
# Sequential full ladder (Steps 0–12) for timm backbones overnight.
#
# PanDerm is skipped (already complete). Runs ResNet-50 → EfficientNet-B3 → UNI.
#
# Usage (on GPU server):
#   export CSG_DATA_ROOT=/mnt/data2/Vinh/CSG-Skin/data
#   export PANDERM_EMBEDDINGS_ROOT=/mnt/data2/Vinh/reference_embeddings
#   nohup ./scripts/run_overnight_batch.sh > results/logs/overnight_batch.log 2>&1 &
#   echo $! > results/logs/overnight_batch.pid
#
# Optional: SKIP_UNI=1 to run only CNN baselines (~6–7 h on A100).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"
BATCH_LOG="$LOG_DIR/overnight_batch.log"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

COMMON_FLAGS=(--skip-split-check)

BACKBONES=(
  configs/resnet50.yaml
  configs/efficientnet_b3.yaml
)

if [[ "${SKIP_UNI:-0}" != "1" ]]; then
  BACKBONES+=(configs/uni.yaml)
fi

echo "=== overnight batch start $STAMP ===" | tee -a "$BATCH_LOG"
echo "backbones: ${BACKBONES[*]}" | tee -a "$BATCH_LOG"
echo "CSG_DATA_ROOT=${CSG_DATA_ROOT:-unset}" | tee -a "$BATCH_LOG"

if [[ -z "${CSG_DATA_ROOT:-}" ]]; then
  echo "ERROR: CSG_DATA_ROOT must be set for timm extraction." | tee -a "$BATCH_LOG"
  exit 1
fi

# Quick sanity: at least one ISIC image under CSG root (flat or nested layout).
SAMPLE="$(find "$CSG_DATA_ROOT" -path '*/ISIC_2019_Training_Input/*' -type f 2>/dev/null | head -1 || true)"
if [[ -z "$SAMPLE" ]]; then
  echo "WARN: no ISIC images under $CSG_DATA_ROOT — extract may fail." | tee -a "$BATCH_LOG"
else
  echo "ISIC sample: $SAMPLE" | tee -a "$BATCH_LOG"
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

echo "=== overnight batch done $(date -u +%Y%m%dT%H%M%SZ) ===" | tee -a "$BATCH_LOG"
if ((${#FAILED[@]})); then
  echo "failed: ${FAILED[*]}" | tee -a "$BATCH_LOG"
  exit 1
fi
echo "all backbones passed Gate 1" | tee -a "$BATCH_LOG"
