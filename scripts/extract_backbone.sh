#!/usr/bin/env bash
# Extract embeddings for one timm backbone (Step 3 only).
#
#   ./scripts/extract_backbone.sh configs/resnet50.yaml
#
# Requires CSG_DATA_ROOT and pinned master_metadata.csv on the machine.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:?usage: extract_backbone.sh configs/<backbone>.yaml}"

cd "$ROOT"
python -m src.pipeline.cli "$CFG" --through-step 3 --skip-split-check
