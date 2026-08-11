#!/usr/bin/env bash
# Full ladder for one backbone. Steps mirror docs/protocol.md exactly.
#
#   ./scripts/run_all.sh configs/medsam.yaml
#
# Run MedSAM first. It is the architecture-portability pilot: the structurally
# most different encoder in the set, so it is the cheapest place to discover the
# recipe does not port. Finding that out on backbone 7 costs the whole budget.

set -euo pipefail
CFG="${1:?usage: run_all.sh configs/<backbone>.yaml}"

echo "!! NOT IMPLEMENTED — scaffold only."
echo "Blocked on D-003 (compute) except for the MedSAM pilot; see ONE_PAGE_SUMMARY.md."
echo
echo "Config: $CFG"
echo "Steps, per docs/protocol.md:"
echo "   0  hard stops        — UNRESOLVED representation / hash mismatch / split mismatch => exit 1"
echo "   1  freeze preprocessing + hash"
echo "   2  resolve representation (refuse if UNRESOLVED)"
echo "   3  extract embeddings  -> manifest"
echo "   4  nuisance direction  w = unit(mu_ISIC - mu_PAD)"
echo "   5  grid search r x lambda_proj   (gate outcomes only, never outcome results)"
echo "   6  Gate 0  implementation integrity  — fail => broken, not a finding"
echo "   7  train adapter x 3 arms x 5 seeds"
echo "   8  alpha ladder by interpolation    — NOT five training runs"
echo "   9  Gate 1  manipulation check       — fail => NOT TESTABLE, stop, not falsification"
echo "  10  geometry metrics    (condition number primary)"
echo "  11  reliability estimators"
echo "  12  append results/manifest.jsonl"
exit 1
