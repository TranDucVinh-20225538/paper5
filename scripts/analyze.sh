#!/usr/bin/env bash
# Cross-backbone analysis. Run ONLY after the per-backbone loop is complete for
# every backbone — partial cross-backbone results are the easiest way to let the
# outcome taxonomy drift toward whatever the early runs happened to show.

set -euo pipefail

echo "!! NOT IMPLEMENTED — scaffold only."
echo "Blocked on D-006 (N not fixed) and D-007 (thresholds not signed off)."
echo
echo "  1  Kendall tau (exact): condition number vs AUROC, per backbone"
echo "  2  paired t-test + Wilcoxon signed-rank for arm comparisons"
echo "  3  Holm-Bonferroni across scorer x metric x arm x BACKBONE"
echo "  4  PRIMARY: backbone as fixed effect + family-level contrasts"
echo "  5  SECONDARY (exploratory, labelled): mixed model, random backbone"
echo "  6  score against the outcome taxonomy — signed-off numbers only"
echo
echo "Before fitting: check the specification against Paper4/15_Causal_Graph.md."
echo "Geometry sits between backbone and intervention strength — conditioning on"
echo "it can manufacture spurious association. Check before, not after."
exit 1
