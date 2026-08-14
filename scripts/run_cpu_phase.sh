#!/usr/bin/env bash
# CPU phase: Steps 8–12 (alpha ladder, geometry, reliability, manifest).
# Requires GPU artifacts on disk (experiments/<backbone>/arms/...).
#
#   ./scripts/run_cpu_phase.sh configs/medsam.yaml
#
# GPU is hidden from child processes so analysis stays on CPU.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:?usage: run_cpu_phase.sh configs/<backbone>.yaml [-- extra flags]}"
shift || true
EXTRA=("$@")

cd "$ROOT"
export OMP_NUM_THREADS="${CPU_THREADS:-16}"
export MKL_NUM_THREADS="${CPU_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS:-16}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS:-16}"
unset CUDA_VISIBLE_DEVICES

exec python -m src.pipeline.cli "$CFG" --from-step 8 --through-step 12 "${EXTRA[@]}"
