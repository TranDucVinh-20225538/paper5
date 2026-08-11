# Paper 5 — Implementation Brief

**Purpose of this file.** Single self-contained handoff for a coding session. Read this before
`00_Kickoff.md` if you only have time for one file — this one has the exact numbers; the kickoff
has the reasoning behind the design choices. Everything under "Reused from Paper 4" is a fact about
what was already built and validated, not a proposal.

---

## 1. Program lineage (context, one line each)

- **Paper 1**: Mahalanobis distance robust to acquisition shift (AUROC 0.97) when softmax
  confidence collapses — ISIC 2018 → PAD-UFES-20, CNN backbones.
- **Paper 2 (CSG-Skin)**: after shortcut-removal via domain-adversarial disentanglement, Mahalanobis
  AUROC on the *same* shift collapses to ≈0.40 — apparent robustness in Paper 1 may have been
  shortcut-driven.
- **Paper 3 (DST-Skin)**: confirms the collapse is not loss of information — a probe still decodes
  domain at 0.72–0.81 AUROC from the same embeddings that defeat 8 different distance/density
  scorers, across a disentanglement dose-response ladder (λ_orth = 0, 1, 5).
- **Paper 4**: preregistered causal test of *why*, on one frozen foundation model (PanDerm
  ViT-L/16). Condition number is the only geometry metric Holm-significantly associated with
  estimator performance; LID and spectral-decay slope are not. The probe/distance gap nearly
  vanishes on this backbone (≈0.002 vs. ≈0.32–0.41 in Paper 3).
- **Paper 5 (this project)**: Paper 4's own Prediction 3 — dropped before execution for resource
  reasons, not logical ones — revived as a **multi-backbone causal replication study**: does the
  condition-number → AUROC relationship hold after accounting for backbone identity, across
  deliberately sampled representation families, each run through Paper 4's full causal ladder
  (not a cheaper frozen-snapshot design)?

Full reasoning and open decisions: `00_Kickoff.md` in this folder.

---

## 2. Reused from Paper 4 — exact protocol (do not re-derive, re-implement per backbone)

### 2.1 Intervention mechanism (the "ladder")

Backbone stays **frozen** throughout — the intervention is a small trainable adapter bolted onto
the frozen output embedding, never fine-tuning the backbone itself.

- Adapter: `z' = z + W2 · GELU(W1 · z)`, bottleneck width `r = 16`, `W2` zero-initialized (identity
  at init). GELU chosen to match PanDerm's own activation — **for other backbones, match each
  backbone's own activation function, don't hardcode GELU.**
- Loss: `L = L_task + λ_proj · L_orth(z + Δz, w)`, `λ_proj = 0.1`.
- Nuisance direction `w = unit(μ_ISIC − μ_PAD)` — closed-form, fixed, computed once from
  domain-conditional means, **never learned/adversarial**. Must be recomputed per backbone (the
  direction lives in that backbone's embedding space).
- `L_orth = cos(z', w)²`.
- Optimizer AdamW, lr=1e-3, batch=512, 100 epochs, 5 seeds `{42, 52, 62, 72, 82}`.
- **Dose ladder is post-hoc interpolation from one trained adapter, not five separate training
  runs**: `z'(α) = z + α·Δz`, `α ∈ {0, 0.25, 0.5, 0.75, 1.0}`.
- Controls:
  - **Conventional arm**: identical architecture/schedule/seeds, `L_orth` term structurally
    absent (not λ=0 — the term itself is removed).
  - **Adaptation arm**: 3 capacity rungs (linear probe / no adapter; partial FT r=8; full adapter
    r=16), task-loss only, isolates capacity effects from the orthogonality objective.
- `r` and `λ_proj` were chosen by a pre-committed grid ({16,32,64,128} × {0.1,1,5}), smallest value
  passing Gate 0 + Gate 1, **never tuned against outcome results**. Re-run this grid-search
  procedure fresh per backbone — don't assume PanDerm's winning values transfer.
- **Abandoned alternative, don't reuse**: a β-scaled ℓ2 normalization knob for geometry
  manipulation was tried and dropped (moved condition number but not LID/spectral-decay).
  Replaced by **spectral tempering**: `z' = μ + U·diag(λᵢ^(−γ/2))·Uᵀ(z−μ)`, γ=0 identity, γ=1 full
  whitening, fit on ID training covariance. Use spectral tempering, not the ℓ2 knob, if a
  geometry-manipulation check is needed per backbone.

### 2.2 Embedding extraction

- PanDerm: checkpoint `panderm_ll_data6_checkpoint-499.pth`, loaded via
  `models.get_encoder(args, 'PanDerm_Large_LP')`, forward pass =
  `model.forward_features(x, is_train=False)` → 1024-d vector, no CLS/pooling ambiguity.
- Intervention acts **only at output stage**, post-`forward_features()` — nothing upstream touched.
- Preprocessing frozen in `assets/reference_preprocessing.json` — **backbone-specific**: each new
  backbone (MedSAM, BiomedCLIP, DINOv3, SigLIP, OpenCLIP, ResNet-50, EfficientNet-B3) needs its own
  frozen preprocessing spec matching its own published eval transform, not PanDerm's.
- **Open per-backbone decision, not yet made**: which layer/pooling to use for non-ViT-classification
  architectures — MedSAM's encoder is segmentation-oriented (patch/mask embeddings, not a single
  pooled vector by default); BiomedCLIP/SigLIP/OpenCLIP are dual-encoder vision-language models
  (use the image tower's pooled output, analogous to CLIP's `image_embeds`). This needs a written
  decision per backbone before extraction, mirroring how PanDerm's `forward_features` choice was
  fixed and documented.

### 2.3 Reliability estimators (unchanged across backbones)

All four fit on the *adapted* training population, evaluated on held-out ID + shifted pool:

- **Mahalanobis**: class-conditional means + shared precision matrix, regularization ε=1e-5, score
  = min squared Mahalanobis distance to any class centroid.
- **Cosine-to-centroid**: max cosine similarity to class centroid.
- **k-NN**: k=10 (also report k=1/10/50 as in Paper 4's Group B baseline).
- **KDE**: class-conditional density estimate.
- Output metric: AUROC (ID vs. domain-shifted) + FPR@95TPR.
- **Explicitly excluded, same as Paper 4**: Energy, ViM, relative-Mahalanobis — classifier-head
  dependent, out of scope for a geometry-only hypothesis.

### 2.4 Gates and statistics (unchanged in spirit, reapply per backbone)

- **Seeds**: 5 per backbone, `{42, 52, 62, 72, 82}`.
- **Gate 0 (Implementation Integrity)**: domain-probe accuracy > majority-baseline + 0.05 (i.e.
  > 0.688+0.05 for this ISIC/PAD split) AND balanced accuracy > chance + 0.10, both via 5-seed
  ℓ2-regularized logistic regression on standardized features. Recompute the majority-baseline
  number if the eval pool composition changes per backbone (it shouldn't — same dataset split is
  reused, see §2.5 — but confirm, don't assume).
- **Gate 1 (Manipulation Check)**: at least one preregistered geometry metric (LID and/or
  spectral-decay slope) shows reproducible dose-dependence across the α-ladder — no monotonicity
  assumed. **Gate 1 failure = "not testable" for that backbone, never counted as a falsification of
  the hypothesis.** This distinction matters for the Outcome-taxonomy scoring in `00_Kickoff.md` —
  a Gate-1 failure on one backbone is a different outcome than a Gate-1 pass with a non-significant
  result.
- **Statistics**: Kendall's τ (exact) for geometry-vs-AUROC association per backbone; paired
  t-test + Wilcoxon signed-rank for arm comparisons; Holm–Bonferroni across scorer × metric × arm
  family — for Paper 5, the correction family also grows to include **× backbone**, which changes
  the correction denominator and therefore the power calculation. This must be redone, not
  inherited from Paper 4's number.
- **Primary Paper-5-specific analysis** (per `00_Kickoff.md`): backbone as a **fixed effect** with
  planned family-level contrasts (CNN vs. medical-SSL vs. medical-VLM vs. general-SSL vs.
  general-VLM), not a random effect — 7 backbones is too few groups for a stable mixed-model
  variance-component estimate. Mixed model still fit as a secondary, explicitly-labeled-exploratory
  check.

### 2.5 Dataset (unchanged, reuse as-is)

- ISIC 2019 (25,331 dermoscopy) + PAD-UFES-20 (2,298 smartphone) = 27,629 total.
- Split seed=42: ISIC test 20% (n=5,067, ID eval), val 20% (n=4,053, unused), train n=16,211.
  OOD = full PAD-UFES-20 (n=2,298). Eval pool n=7,365.
- Existing frozen embedding paths (PanDerm only, so far):
  `Paper4/PhaseB/assets/reference_embeddings/ReferenceEmbedding/` (eval, frozen),
  `.../ReferenceTrainEmbedding/` (n=16,211, fitting population).
  **These need to be regenerated per new backbone** — the extraction scripts exist (§3) but the
  embeddings themselves are backbone-specific and don't currently exist for MedSAM/BiomedCLIP/
  DINOv3/SigLIP/OpenCLIP/ResNet-50/EfficientNet-B3.

---

## 3. Reusable code (Paper 4, verified to exist)

All under `/Users/cubo/Research/Paper4/PhaseB/`:

- `analysis/` — ~30 scripts covering the full pipeline: `lid_spectral_decay.py`,
  `stage4_adapter.py`, `stage4_train_intervention.py`,
  `stage4_canonical_intervention/conventional/adaptation.py`, `stage4_alpha_ladder.py`,
  `stage4_geometry_completion.py`, `stage4_reliability_scorers.py`, `stage5_c3_*.py`,
  `compute_nuisance_direction.py`, `stage2_1_extract_reference_embeddings.py`.
- `InterventionArm_Canonical/` — 5 trained adapter checkpoints (`.pt`) + adapted embeddings
  (`.npy`) per seed, PanDerm only.
- `AlphaLadder/`, `analysis/` outputs — JSON results, not raw dumps.

**Adaptation needed for Paper 5**: the extraction script
(`stage2_1_extract_reference_embeddings.py`) is PanDerm-specific (calls
`models.get_encoder(args, 'PanDerm_Large_LP')` and PanDerm's `forward_features`). It needs a
backbone-agnostic rewrite — one extraction entrypoint per backbone family (timm/HF loader for
ResNet-50/EfficientNet-B3, PanDerm loader as-is, MedSAM's own repo loader, open_clip/HF loaders for
BiomedCLIP/SigLIP/OpenCLIP, DINOv3's own loader) — everything downstream
(`stage4_adapter.py` onward) should be backbone-agnostic already since it operates on saved
embedding arrays, not on the backbone directly. Confirm this assumption before relying on it.

---

## 4. What's new for Paper 5 (not in Paper 4, must be built)

1. **Backbone loop**: run §2.1–2.4 independently for each of the 7 sampled backbones (see
   `00_Kickoff.md` §"Representation-family sampling strategy" for the current list and its two
   unresolved singleton cells — medical-VLM and general-SSL each need a second instance sourced
   before this loop can be considered complete).
2. **Backbone-agnostic extraction wrapper** (§3 above).
3. **Per-backbone preprocessing spec** (§2.2) and **per-backbone layer/pooling decision** (§2.2,
   especially MedSAM and the CLIP-style dual encoders).
4. **Fresh grid-search for `r`/`λ_proj`** per backbone (§2.1) — do not reuse PanDerm's values.
5. **Family-level fixed-effect statistical model** (§2.4) — new analysis code, doesn't exist in
   Paper 4's `analysis/` folder since Paper 4 only ever had one backbone.
6. **Architecture-portability pilot**: before committing all 7, run the full ladder on one
   non-ViT-classification backbone (MedSAM suggested in `00_Kickoff.md`) as a feasibility check —
   confirms the adapter/nuisance-direction/Gate-0/Gate-1 machinery actually works on a structurally
   different encoder before spending compute on all 7.

---

## 5. Blockers before full-scale runs start (from `00_Kickoff.md`, repeated here for the coding
session's benefit)

1. Compute/server access and multi-day job duration — confirmed or still assumed?
2. Second instance for medical-VLM family (candidates: PLIP, MedCLIP, Quilt-1M-CLIP — not chosen).
3. Second instance for general-SSL family (candidate: DINOv2 — not chosen).
4. Outcome-taxonomy thresholds (`00_Kickoff.md`) are drafts — need real numbers before any result
   is reported as "preregistered."

---

## 6. File index

- `00_Kickoff.md` — design reasoning, lineage, outcome taxonomy, open decisions.
- `01_Implementation_Brief.md` — this file, exact reusable protocol + what's new.
- `/Users/cubo/Research/Paper4/PhaseB/` — source of all reused code and specs above.
- `/Users/cubo/Research/Paper4/PhaseB/assets/reference_embeddings/` — existing PanDerm embeddings
  (reusable as one backbone's data directly, no rerun needed for PanDerm itself).
