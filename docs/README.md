# docs/

| File | Holds | Status |
|---|---|---|
| [`00_Kickoff.md`](00_Kickoff.md) | Design reasoning, lineage, sampling strategy, outcome taxonomy, open decisions | Reference |
| [`01_Implementation_Brief.md`](01_Implementation_Brief.md) | Exact reusable protocol from Paper 4 + what is new | Reference |
| [`backbone_audit.md`](backbone_audit.md) | Task 3 — nine-axis assessment, final backbone set, every rejected candidate with its reason | Approved, D-014…D-024 |
| [`medsam_integration.md`](medsam_integration.md) | Task 4 — MedSAM representation, pooling, preprocessing, risks; implementation-ready | Awaiting sign-off, D-025…D-027 |
| [`protocol.md`](protocol.md) | Ordered per-backbone runbook — the *how*, not the *why* | Draft — **§9 of the MedSAM doc supersedes its "MedSAM first" ordering** |
| [`preregistration.md`](preregistration.md) | The preregistration | **BLOCKED — deliberately empty** |
| [`power_analysis.md`](power_analysis.md) | Power calculation | **BLOCKED — nothing inherited from Paper 4** |

Task 2's portability audit lives at [`../REUSE.md`](../REUSE.md) — repo root, since it governs what `src/` is built from.

Decisions are **not** here. They are program-tier: [`../../lab-notebook/`](../../lab-notebook/README.md).

## Division of labour

`00_Kickoff.md` and `01_Implementation_Brief.md` carry the reasoning and the exact numbers
respectively. `protocol.md` carries the order of operations and deliberately does not restate the
reasoning — two documents explaining the same decision will drift, and then neither can be trusted.

If you find yourself copying a rationale from the brief into the protocol, link instead.
