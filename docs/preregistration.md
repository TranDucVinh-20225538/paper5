# Preregistration

    Status:      BLOCKED — deliberately empty
    Blocked on:  D-003, D-004, D-005, D-006, D-007
    Do not fill in until every one of those is closed.

This file is empty on purpose, and the emptiness is the point.

A preregistration written while its own inputs are still open is not a preregistration — it is a
plan that will be quietly revised once results arrive, which is the exact failure mode
preregistration exists to prevent. Paper 5's own kickoff already names this risk: draft thresholds
"will read as post-hoc flexible when the paper is written."

**Until this file has content, no document, commit message, talk, or draft may describe any part of
this study as preregistered.**

## The critical path

These blockers are ordered. Each one genuinely gates the next — this is not a checklist that can be
worked in parallel.

```
D-003  compute confirmed
   |     (if partially constrained -> fallback F-006, not a redesign)
   v
D-004  medical-VLM 2nd instance  ----+
D-005  general-SSL  2nd instance  ----+
                                      |
                                      v
                              D-006  backbone count N fixed
                                      |
                    +-----------------+-----------------+
                    v                                   v
          D-007  outcome thresholds            power_analysis.md
          (denominators need N)                (Holm family is x backbone,
                    |                           so the correction denominator
                    |                           and the power calc need N)
                    +-----------------+-----------------+
                                      v
                            THIS FILE can be written
                                      v
                              protocol lock, runs start
```

Two consequences worth stating plainly:

**The power analysis cannot be inherited from Paper 4.** The Holm–Bonferroni family grows from
scorer × metric × arm to scorer × metric × arm **× backbone**. That changes the correction
denominator, which changes the power calculation. The implementation brief already flags this; it is
repeated here because it is easy to skip.

**D-008 sits outside this chain but blocks the RQ itself.** If arXiv:2510.15202v3 turns out to
contain a causal multi-backbone component rather than a purely correlational one, the gap claim
weakens and the research question needs revising before anything below it is written.

## What goes here when it is unblocked

1. Final research question, with N fixed and stated.
2. Hypotheses, with direction, per family.
3. Sampling frame — the backbone list, closed, with the reason each was selected and each
   considered-but-rejected alternative pointing at its decision-log ID. *(This is what answers the
   reviewer question "why OpenCLIP and not EVA".)*
4. Intervention protocol, per backbone, including each resolved layer/pooling decision.
5. Gate 0 and Gate 1 criteria as executable thresholds.
6. Primary analysis: the fixed-effect model and the exact family-level contrasts.
7. Secondary and exploratory analyses, labelled as such **here**, not retrospectively.
8. Outcome taxonomy with real numbers.
9. Stopping rules and what counts as *not testable* versus *falsified* — the Gate-1 distinction.
10. Deviation policy: how a departure gets recorded (a decision-log entry with
    `Outcome data seen: YES`) rather than silently absorbed.
