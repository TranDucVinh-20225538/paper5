#!/usr/bin/env bash
# Push GPU-phase artifacts (experiments/<backbone>/ + manifest slice) to server.
#
# Usage (from Mac, after local GPU phase):
#   ./scripts/push_gpu_artifacts.sh medsam user@nat.ioit.science:/mnt/data2/Vinh/paper5
#
# On server, append manifest slice then run CPU phase:
#   cat results/manifest_import/medsam.jsonl >> results/manifest.jsonl
#   ./scripts/run_cpu_phase.sh configs/medsam.yaml --skip-split-check

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKBONE="${1:?usage: push_gpu_artifacts.sh <backbone> user@host:/path/to/paper5}"
REMOTE="${2:?usage: push_gpu_artifacts.sh <backbone> user@host:/path/to/paper5}"

cd "$ROOT"
EXP="$ROOT/experiments/$BACKBONE"
if [[ ! -d "$EXP/embeddings" && ! -d "$EXP/arms" ]]; then
  echo "ERROR: no GPU artifacts under $EXP (run ./scripts/run_gpu_phase.sh first)" >&2
  exit 1
fi

PYTHON="${PYTHON:-python}"
IMPORT="$ROOT/results/manifest_import/${BACKBONE}.jsonl"
"$PYTHON" "$ROOT/scripts/export_manifest_backbone.py" "$BACKBONE" --out "$IMPORT"

echo "=== rsync experiments/$BACKBONE → $REMOTE ==="
rsync -avP --delete \
  "$EXP/" \
  "${REMOTE}/experiments/${BACKBONE}/"

echo "=== rsync manifest slice ==="
rsync -avP "$IMPORT" "${REMOTE}/results/manifest_import/"

echo ""
echo "Done. On server:"
echo "  cd ${REMOTE#*:}"
echo "  git pull origin feat/gpu-cpu-scheduler   # or main"
echo "  cat results/manifest_import/${BACKBONE}.jsonl >> results/manifest.jsonl"
echo "  python -m src.utils.scheduler_state configs/${BACKBONE}.yaml   # expect gpu_done"
echo "  nohup ./scripts/run_cpu_phase.sh configs/${BACKBONE}.yaml --skip-split-check \\"
echo "    > results/logs/${BACKBONE}_cpu.log 2>&1 &"
