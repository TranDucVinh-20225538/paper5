# REUSE.md — Paper 4 → Paper 5 portability audit (Task 2)

    Status:  RECOMMENDATION — awaiting PI sign-off
    Scope:   architecture portability only. No implementation.
    Audited: /Users/cubo/Research/Paper4/PhaseB/analysis/ (30 modules) + 2 external dependencies
    Date:    2026-08-11

---

## 1. Verdict

`01_Implementation_Brief.md` §3 states:

> *everything downstream (`stage4_adapter.py` onward) should be backbone-agnostic already since it
> operates on saved embedding arrays, not on the backbone directly. **Confirm this assumption before
> relying on it.***

**Confirmed as substantially true, and false in three specific ways.** The brief was right to flag
it. Two of the three are serious; one changes the definition of the study's primary covariate.

| # | Finding | Severity |
|---|---|---|
| 1 | The pipeline **imports code from two other project directories** via absolute `sys.path` — it is not self-contained | **high** |
| 2 | **`condition_number` is not in Paper 4's codebase**, and computes something narrower than Paper 5's documents currently say | **high — amends D-019** |
| 3 | `stage4_adapter.py` **hardcodes GELU** with no parameter — silently wrong on ReLU/SiLU backbones | medium |

Everything else in the "downstream is array-level" claim holds up. The Stage-4 and Stage-5 modules do
operate on cached `.npy` arrays and never touch a backbone.

---

## 2. Finding 1 — the pipeline is not self-contained

Four Stage-4/5 modules bootstrap themselves onto **other papers' codebases** with hardcoded absolute
paths:

```python
sys.path.insert(0, "/Users/cubo/Research/paper-3/scripts")   # Paper 3
sys.path.insert(0, "/Users/cubo/Research/CSG-SKin")          # Paper 2
```

and then import:

| Symbol | Lives in | Used by |
|---|---|---|
| `geometry_diagnostics.condition_number` | **paper-3** | `stage4_geometry_completion.py`, `stage3_2_reference_arm_baseline.py` |
| `src.utils.ood_metrics.compute_mahalanobis_params_from_arrays` | **CSG-SKin** | `stage4_geometry_completion.py`, `stage4_reliability_scorers.py`, `stage5_c3_canonical.py` |

Consequences, in order of how much they will hurt:

1. **The primary covariate and the primary estimator both live outside the repository being
   published.** A reproducibility check that clones Paper 5's repo gets neither.
2. **The paths are machine-specific.** `/Users/cubo/…` works on exactly one Mac. Any server run —
   which is now the plan, D-003 — fails at import.
3. **No version pinning.** `sys.path.insert` takes whatever is on disk. If `paper-3` changes,
   Paper 5's κ changes silently, with no error and no record.

**Recommendation: vendor, do not re-implement, and do not keep the path hack.**

Copy both functions into `src/geometry/` and `src/estimators/` with a header recording the source
repository, file, and **the commit they were taken from** — both source repos are under git and
currently sit at `paper-3 @ 5fedcb3` and `CSG-SKin @ 1233898`. Pin those SHAs in the vendored header
and in `results/manifest.jsonl`.

Vendoring rather than re-implementing is deliberate: these functions produced Paper 4's published
numbers, and re-implementing them would silently break the replication this paper claims to be.
The PanDerm regression run (D-027) is what proves the vendored copies are faithful.

---

## 3. Finding 2 — what `condition_number` actually computes

This is the one that changes a decision already accepted.

**D-019 as written says** condition number is `λ_max/λ_min` of "the embedding covariance."
**What Paper 4 actually computes** is narrower:

```
compute_mahalanobis_params_from_arrays(features, labels, num_classes=8, reg_eps=1e-5)
    cov  = pooled WITHIN-CLASS covariance          # Σ_W, not the marginal covariance
    cov  = cov + reg_eps * I                       # absolute regularization
    precision = inv(cov)

condition_number(precision) = λ_max/λ_min over positive eigenvalues
```

Two corrections follow.

**(a) It is the pooled within-class covariance `Σ_W`, not the marginal covariance.** If Paper 5
computed κ on the marginal covariance it would not be replicating Paper 4's quantity, and the whole
replication claim would rest on a quantity that was never measured. D-019 must be amended to name
`Σ_W` explicitly.

*The dimension argument in D-019 survives this correction unchanged* — `Σ_W` is still `d×d`,
estimated from `N−K` samples, and its trailing eigenvalues still shrink as `d` grows. The confound is
real; only the matrix's name was wrong.

*(Taking κ from the precision rather than inverting back is correct and well-reasoned in Paper 3's
own docstring: κ is invariant under inversion, since inversion reciprocates every eigenvalue and
swaps which is max and min without changing the ratio. Keep it.)*

**(b) The regularizer is absolute, and this is a second confound — invisible, and worse than the
first.**

`cov + 1e-5 · I` floors the smallest eigenvalue at ε, so `κ ≤ λ_max/ε`. Because ε is **absolute**,
its effective strength depends on the scale of the embeddings — and embedding norms differ
substantially across backbones (a CLIP pre-projection vector, a ResNet GAP output and a DINOv2 CLS
token are not on a common scale). So ε=1e-5 is heavy regularization for one backbone and negligible
for another, and κ is comparing quantities regularized to **different degrees**.

This never surfaced in Paper 4 because there was one backbone and therefore one scale.

**Recommendation — decouple the scorer from the covariate:**

| | Definition | Purpose |
|---|---|---|
| **κ_paper4** | exactly Paper 4's: full-`d`, `Σ_W + 1e-5·I`, absolute ε | Direct replication comparison. Reported for PanDerm at minimum; validated by the D-027 regression run against Paper 4's published value |
| **κ_primary** | top-k eigenvalues of `Σ_W` computed on **scale-normalized** embeddings, k fixed across backbones | The cross-backbone covariate (D-019) |

The Mahalanobis **estimator** keeps ε=1e-5 absolute, unchanged, so the scorer matches Paper 4 exactly.
Only the **geometry descriptor** is made commensurable. This is legitimate precisely because κ is not
part of the scorer — it is a description of the representation, and a description that is not
comparable across the things being compared is not doing its job.

**Precise proposed definition of κ_primary**, so this is implementable without further judgement:
take the eigenvalues of `Σ_W` in descending order, and set `κ_k = λ_1 / λ_k` for fixed k across all
backbones. This is the condition number restricted to the leading k-dimensional eigen-subspace of the
same matrix Paper 4 used — not a different quantity, a truncation of the same one.

---

## 4. Finding 3 — the adapter hardcodes GELU

`stage4_adapter.py` is 56 lines and is otherwise in good shape. `dim` and `num_classes` are already
constructor parameters with defaults (`EMBED_DIM = 1024`, `NUM_CLASSES = 8`), so a 768-d backbone
works by passing `dim=768` — and a caller who forgets gets a **loud** shape error from `nn.Linear`.

The activation does not have that property:

```python
self.act = nn.GELU(approximate="none")   # matches PanDerm's own activation, SS1
```

No parameter. On ResNet-50 (ReLU) or EfficientNet-B3 (SiLU) this runs **without error** and produces
an adapter that violates the protocol's explicit per-backbone activation rule. Silent, not loud —
which makes it more dangerous than the dimension default.

**Recommendation.** Add an `act` constructor parameter defaulting to GELU, resolved from the
backbone config. Three lines. Also change `EMBED_DIM`'s default to `None` and raise if unset, so that
dimension becomes explicit rather than inherited from PanDerm by accident.

---

## 5. Module categorization

### 5.1 Reusable without modification

| Module | Note |
|---|---|
| `lid_spectral_decay.py` | Pure NumPy. `features`/`labels`/`num_classes`/`k` all parameters, no paths, no torch. The cleanest module in the audit |
| `geometry_diagnostics.condition_number` *(paper-3)* | Pure. **Vendor it** — Finding 1 |
| `ood_metrics.compute_mahalanobis_params_from_arrays` *(CSG-SKin)* | Pure. **Vendor it** — Finding 1 |
| `stage4_adapter.orthogonality_loss`, `task_loss` | Pure tensor functions, no backbone assumptions |

### 5.2 Reusable with a wrapper

All of these are array-level — the brief's claim holds. They need config injection (paths, `dim`,
`r`) rather than surgery.

| Module | What the wrapper must supply |
|---|---|
| `stage4_adapter.py` | `act` parameter (§4); make `dim` explicit |
| `stage4_train_intervention.py` | Per-backbone `TRAIN_DIR`, embed dim. PanDerm appears only in a docstring |
| `stage4_alpha_ladder.py` | Per-backbone artifact paths |
| `compute_nuisance_direction.py` | Currently pinned to one `assets/reference_embeddings/…` pair. Paths are script-relative (`Path(__file__).parents[1]`), **not** absolute — good. Needs a per-backbone artifact dir |
| `stage4_reliability_scorers.py` | `R = 16` is a module constant, and `ARMS` hardcodes directory names. Also carries a `sys.path` hack |
| `stage4_geometry_completion.py` | Two `sys.path` hacks; otherwise array-level |
| `stage4_canonical_intervention.py` / `_conventional.py` / `_adaptation.py` | Arm drivers; path + hyperparameter injection |
| `groupB_mahalanobis_family.py` | Scorer family; path injection |
| `stage5_c3_spectral_tempering_check.py` | Needed for D-013. The `d=1024` occurrence is in a timing **comment**, not code |
| `stage1_3_rerun_preregistered_split.py` | Not needed to regenerate the split — needed to **verify** it is unchanged, which the protocol requires at Step 0 |

### 5.3 Rewrite

| Module | Why |
|---|---|
| `stage2_1_extract_reference_embeddings.py` | PanDerm-specific: `get_encoder(args, 'PanDerm_Large_LP')` + `forward_features`. Already flagged in the brief. Needs one entrypoint per backbone family |
| `ea01_extract_reference_train_embeddings.py` | Same, for the training population |
| `stage1_2_encoder_forward_pass.py` | PanDerm forward-pass validation |
| `stage4_select_lambda.py` | `R_FIXED = 16  # Step 3 result` bakes **PanDerm's grid outcome** in as a constant. Since D-021 requires a fresh grid per backbone, this must become a function of the backbone, not a constant |

### 5.4 Does not exist — must be built

Named here because a portability audit that only looks at what exists will under-count the work.

| Component | Driven by |
|---|---|
| `κ_primary` — top-k, scale-normalized | D-019 as amended (§3) |
| **Gate 0-pre** — frozen-feature adequacy check | D-020 |
| **Family-level fixed-effect model + planned contrasts** | D-011. Paper 4 had one backbone, so no such code exists |
| Backbone-agnostic extraction dispatch | Brief §3 |
| Masked pooling + padded-fraction pre-check | D-026, MedSAM only |
| `results/manifest.jsonl` provenance writer | repo convention |

### 5.5 Obsolete for Paper 5

Paper 4 stage-specific; retained in Paper 4's repo, not carried over. Flagged at moderate confidence
from structure and naming — worth a skim before deletion, though nothing depends on them.

`stage0_retrospective.py` · `stage0_figure.py` · `stage3_0_artifact_audit.py` ·
`stage3_0b_duplicate_registry_and_issues.py` · `stage3_1_artifact_characterization.py` ·
`stage3_2_reference_arm_baseline.py` · `stage4_fix_adaptation_scorers.py` *(a one-time patch script)* ·
`stage1_3a_dataset_validation.py` · `stage1_3_batch_inference_validation.py` ·
`stage2_1b_finalize_embedding_artifact.py` · `build_master_metadata_with_provenance.py`

---

## 6. Effect on the plan

The brief's assumption held well enough that **no Stage-4/5 module needs rewriting for
backbone-agnosticism** — the array-level design was sound. The real work is different from what was
expected:

- **Not** "make the downstream backbone-agnostic" — it already is.
- **But** untangle the cross-project dependency, which nobody had costed (§2).
- **And** settle what κ actually means before any is computed (§3).

§3 is the item that would have been expensive to discover late: had Paper 5 computed κ on the
marginal covariance across ten backbones, every number would have been internally consistent,
superficially reasonable, and **not a replication of Paper 4** — and there is no point downstream at
which that would have announced itself.

## 7. Proposed decision-log entries

| ID | Decision |
|---|---|
| D-028 | **Vendor** `condition_number` (paper-3 @ `5fedcb3`) and `compute_mahalanobis_params_from_arrays` (CSG-SKin @ `1233898`) into `src/`, with source repo + commit recorded in-file and in the manifest. Remove all absolute `sys.path` hacks. Do not re-implement |
| D-029 | **Amend D-019**: κ is defined on the **pooled within-class covariance `Σ_W`**, not the marginal covariance. `κ_primary = λ_1/λ_k` over `Σ_W`'s descending eigenvalues, k fixed across backbones, on scale-normalized embeddings. `κ_paper4` (full-`d`, absolute ε=1e-5) retained and reported for direct replication comparison |
| D-030 | The Mahalanobis **estimator** keeps ε=1e-5 absolute unchanged, matching Paper 4. Only the geometry **covariate** is made scale-commensurable. Scorer and descriptor are decoupled deliberately |
| D-031 | `stage4_adapter.BottleneckAdapter` gains an `act` parameter resolved per backbone from config; `dim` becomes required rather than defaulting to 1024 |
