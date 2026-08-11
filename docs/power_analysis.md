# Power Analysis

    Status:      BLOCKED — deliberately empty
    Blocked on:  D-006 (backbone count N is not fixed)
    Inherited:   NOTHING from Paper 4. See below.

## Why nothing carries over from Paper 4

Paper 4 applied Holm–Bonferroni across a family of **scorer × metric × arm**.

Paper 5's family grows by one dimension:

    scorer x metric x arm x BACKBONE

That multiplies the number of hypotheses in the family, which raises the correction denominator,
which lowers the per-test significance threshold, which changes the detectable effect size at any
given power. Paper 4's number is therefore not a conservative approximation of Paper 5's — it is
simply the wrong number, and reusing it would overstate power.

The implementation brief states this directly: the correction "must be redone, not inherited from
Paper 4's number."

## Why it cannot be computed yet

Every term below depends on **N**, the backbone count, and N is not fixed (D-006):

- The documents disagree — the kickoff's sampling table enumerates 8, both documents say 7
  throughout, and the likely origin is that *7 backbones needing extraction* (8 minus PanDerm, whose
  embeddings already exist) was carried over as *7 total*.
- Closing D-004 and D-005 adds two more instances, making 10.

So N is currently one of {7, 8, 10}, and the design **guarantees** it will change, because the two
singleton family cells must be closed before protocol lock. Computing power against any of the three
now would produce a number that is wrong by construction.

## What to compute when unblocked

1. **Family size.** Enumerate the full hypothesis family explicitly: 4 estimators × geometry metrics
   × arms × N backbones. Write the count, do not estimate it.
2. **Effect size.** Paper 4's observed condition-number ↔ AUROC association is the anchor for the
   expected effect. Note that Paper 4 measured it on **one** backbone, so it carries no information
   about between-backbone variance — which is exactly what Paper 5 is powered to detect.
3. **Per-backbone power** for Kendall's τ at the Holm-adjusted α, given 5 seeds × 5 α-rungs.
4. **Family-contrast power** for the primary analysis — the CNN / medical-SSL / medical-VLM /
   general-SSL / general-VLM contrasts, at 2 instances per cell.
5. **Sensitivity to N.** Report power at each candidate N. If the study is adequately powered at
   N=10 but not at N=8, that is itself an argument for closing D-004 and D-005 rather than
   proceeding.
6. **Seed count.** Confirm 5 seeds per backbone still suffices under the enlarged correction family,
   or raise it. The kickoff commits to "seeds matching or exceeding Paper 4's five" — *exceeding* is
   available and may be needed.

## One thing to be careful about

The fixed-effect decision (D-011) is argued in the kickoff partly on group count: "commonly cited
guidance wants something like 8–10+ groups" against "this design has 7."

At N=8 the design sits exactly on that boundary; at N=10 it is inside it. The **decision** still
holds on interpretability and power grounds, and does not depend on which of 7/8/10 is right — but
the stated numerical justification does not survive as written, and must be restated once N is fixed.
Do not leave that sentence in the manuscript unchanged.
