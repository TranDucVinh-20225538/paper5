#!/usr/bin/env bash
# Rerun backbones that failed overnight (ResNet-50 grid, UNI extract).
#
#   export CSG_DATA_ROOT=/mnt/data2/Vinh/CSG-Skin/data
#   nohup ./scripts/rerun_failed_backbones.sh > results/logs/rerun_failed.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMMON=(--skip-split-check)
LOG_DIR="$ROOT/results/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

for cfg in configs/resnet50.yaml configs/uni.yaml; do
  name="$(basename "$cfg" .yaml)"
  log="$LOG_DIR/${name}_ladder_${STAMP}.log"
  echo "--- rerun $name @ $(date -u +%H:%M:%S) ---"
  ./scripts/run_all.sh "$cfg" 12 "${COMMON[@]}" | tee "$log"
done
