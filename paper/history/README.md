# paper/history/

One snapshot per major revision round: `P5_round0`, `P5_round1`, …

## Built PDFs only — not copied sources

The manuscript source is versioned by **git tags**:

```bash
git tag -a p5-round0 -m "Round 0: first full draft"
```

and the built PDF for that tag is archived here as `P5_round0.pdf`.

Copying the whole manuscript directory per round *feels* safer but is not: two editable copies of the
same document means edits eventually land in the wrong one, and the divergence is usually noticed
weeks later. Git already does source snapshots correctly and does them for free. What git does *not*
give you is a browsable, openable artifact of exactly what was submitted — that is what this folder
is for.

Rule of thumb: **tag what you edit, archive what you send.**

## Contents per round

| File | |
|---|---|
| `P5_roundN.pdf` | exactly what was submitted |
| `P5_roundN_notes.md` | what changed since N−1, and which decision IDs drove it |

Reviewer responses go in `../reviewer_response/`, not here.
