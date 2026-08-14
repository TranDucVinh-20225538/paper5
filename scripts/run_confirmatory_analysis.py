#!/usr/bin/env python3
"""
Confirmatory analysis — implements docs/preregistration.md, frozen at a83c74c.

This script is a reconstruction. The original run was reported at commit da8b775,
which is not present on any branch (D-053), so the analysis was rewritten from the
frozen specification and re-run against the committed intermediate outputs. Those
inputs — results/csv/<backbone>/{geometry_metrics,reliability_scorers}.json — are
under version control and unchanged, so the reconstruction is checkable: it either
reproduces the reported table or it does not, and either outcome is informative.

Nothing here is a choice. Every number that could have been decided was decided in
the preregistration, before any cross-backbone association existed. Where this file
looks rigid, that is the point.

    python3 scripts/run_confirmatory_analysis.py
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FREEZE = "a83c74c3273189d9c9a5573b5635fb555279b468"

# D-017: the ten family members. MedSAM is a probe (D-016) and appears nowhere below.
FAMILIES = {
    "cnn": ["resnet50", "efficientnet_b3"],
    "medical_ssl": ["panderm", "uni"],
    "medical_vlm": ["biomedclip", "monet"],
    "general_ssl": ["dinov3", "mocov3"],
    "general_vlm": ["openclip", "siglip"],
}
BACKBONES = [b for fam in FAMILIES.values() for b in fam]

LADDER_ALPHAS = (0.25, 0.5, 0.75, 1.0)   # D-034
RUNGS = ("linear-probe", "partial-FT")   # D-034: full-adapter-FT excluded
ALPHA_FWER = 0.05                        # preregistration §4
OUTCOME = "maha_auroc"                   # D-037: Mahalanobis is the confirmatory scorer
KAPPA_PRIMARY = "condition_number_primary"
SECONDARY_SCORERS = ("cosine_auroc", "knn_k10_auroc", "kde_auroc")


# --------------------------------------------------------------------------- io


def _load(backbone: str, name: str) -> dict:
    return json.loads((REPO / "results" / "csv" / backbone / f"{name}.json").read_text())


def build_population(backbone: str, kappa_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The n=30 population fixed by D-034. Returns (kappa, auroc, seed).

    20 rows from the intervention arm at four α values × five seeds, plus five
    linear-probe and five partial-FT rows. The rungs carry no α: they are capacity
    points, not dose points, so each seed contributes exactly one row per rung.
    """
    geom = _load(backbone, "geometry_metrics")["results"]
    scor = _load(backbone, "reliability_scorers")["results"]

    gk = {("c", r["seed"], r["alpha"]): r[kappa_key] for r in geom["canonical"]}
    gk.update({("a", r["seed"], r["rung"]): r[kappa_key] for r in geom.get("adaptation", [])})

    kappa, auroc, seed = [], [], []
    for r in scor["canonical"]:
        if r["alpha"] in LADDER_ALPHAS:
            kappa.append(gk[("c", r["seed"], r["alpha"])])
            auroc.append(r[OUTCOME])
            seed.append(r["seed"])
    for r in scor.get("adaptation", []):
        if r["rung"] in RUNGS:
            kappa.append(gk[("a", r["seed"], r["rung"])])
            auroc.append(r[OUTCOME])
            seed.append(r["seed"])

    return np.array(kappa, float), np.array(auroc, float), np.array(seed, int)


# ------------------------------------------------------------------ inference


def seed_permutation_test(
    kappa: np.ndarray, auroc: np.ndarray, seed: np.ndarray
) -> tuple[float, float, int]:
    """Exact seed-level permutation test for Kendall's τ (D-039).

    Whole seeds are permuted, never individual rows. Rows sharing a seed share an
    adapter, a training trajectory and an initialisation, so they are not
    exchangeable at the row level; permuting rows would reintroduce precisely the
    independence assumption this test exists to avoid.

    With five seeds the permutation group has 5! = 120 elements, so p-values are
    quantised at multiples of 1/120 ≈ 0.0083. That resolution is a property of the
    design and is reported rather than worked around.
    """
    seeds = sorted(set(seed.tolist()))
    # Rows are grouped by seed in a fixed within-seed order, so a seed permutation
    # relabels blocks while preserving the condition alignment inside each block.
    blocks = [np.where(seed == s)[0] for s in seeds]
    sizes = {len(b) for b in blocks}
    if len(sizes) != 1:
        raise ValueError(f"unbalanced seed blocks: {sizes}")

    tau_obs = kendalltau(kappa, auroc).statistic
    perms = list(itertools.permutations(range(len(seeds))))
    count = 0
    for p in perms:
        permuted = np.concatenate([auroc[blocks[i]] for i in p])
        ordered = np.concatenate([kappa[b] for b in blocks])
        t = kendalltau(ordered, permuted).statistic
        if abs(t) >= abs(tau_obs) - 1e-12:
            count += 1
    return float(tau_obs), count / len(perms), len(perms)


def holm(pvals: list[float], alpha: float = ALPHA_FWER) -> tuple[list[float], list[bool]]:
    """Holm–Bonferroni step-down. Returns (adjusted p, rejected)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adj[idx] = running
    return adj.tolist(), [a <= alpha for a in adj]


def tier1(s: int, t: int) -> str:
    """D-032. Defined on the count S alone; no family-pattern condition here."""
    if s >= t:
        return "A"
    if s >= int(np.ceil(0.6 * t)):
        return "B"
    if s >= 2:
        return "C"
    return "D"


# ----------------------------------------------------------------------- main


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE, "HEAD"], cwd=REPO)
    if anc.returncode != 0:
        print(f"ABORT: freeze commit {FREEZE[:7]} is not an ancestor of HEAD", file=sys.stderr)
        return 1

    out_dir = REPO / "results" / "csv" / "confirmatory"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Frozen protocol : {FREEZE[:7]} (v0.2-protocol-frozen)")
    print(f"Analysis commit : {head[:7]}\n")

    # --- Step 0: testability gate (D-033), before anything else ----------------
    testable = []
    for b in BACKBONES:
        g1 = _load(b, "gate1_ea03")["results"]["canonical"]
        if any(r.get("gate1_pass") for r in g1):
            testable.append(b)
    T = len(testable)
    complete_cells = sum(all(x in testable for x in m) for m in FAMILIES.values())
    print(f"Testability gate: T = {T}, complete family cells = {complete_cells}")
    if T < 5 or complete_cells < 2:
        print("STOP — feasibility study; neither tier scored (D-033).")
        return 0

    # --- Steps 2-3: per-backbone association ----------------------------------
    rows = []
    for b in testable:
        k, a, s = build_population(b, KAPPA_PRIMARY)
        assert len(k) == 30, f"{b}: population is {len(k)}, expected 30"
        tau, p_perm, n_perm = seed_permutation_test(k, a, s)
        res = kendalltau(k, a)
        rows.append({
            "backbone": b,
            "n": len(k),
            "tau": tau,
            "p_perm": p_perm,
            "n_perm": n_perm,
            "p_param": float(res.pvalue),
            "auroc_min": float(a.min()),
            "auroc_max": float(a.max()),   # D-038: never report τ without this
        })

    # --- Step 4: Holm on the permutation p-values, family of 10 (D-037) -------
    adj, rej = holm([r["p_perm"] for r in rows])
    for r, pa, rj in zip(rows, adj, rej):
        r["p_holm"], r["significant"] = pa, rj

    # --- Step 5: taxonomy (D-032) ---------------------------------------------
    sig = [r for r in rows if r["significant"]]
    if sig:
        direction = np.sign(np.median([r["tau"] for r in sig]))
        S = sum(1 for r in sig if np.sign(r["tau"]) == direction)
    else:
        S = 0
    t1 = tier1(S, T)

    fam_sig = {f: sum(1 for b in m if any(r["backbone"] == b and r["significant"] for r in rows))
               for f, m in FAMILIES.items()}
    fam_tau = {f: float(np.mean([r["tau"] for r in rows if r["backbone"] in m]))
               for f, m in FAMILIES.items()}
    reversed_fams = [f for f, v in fam_tau.items() if v < 0]
    if any(v == 2 for v in fam_sig.values()) and any(v == 0 for v in fam_sig.values()):
        t2 = "family-specific"          # contrast significance assessed separately
    elif not reversed_fams:
        t2 = "consistent"
    else:
        t2 = "heterogeneous"

    # --- report ---------------------------------------------------------------
    print(f"\n{'backbone':<18}{'tau':>9}{'p_perm':>9}{'p_param':>11}{'p_holm':>9}{'sig':>5}"
          f"   {'AUROC range'}")
    print("-" * 78)
    for r in rows:
        print(f"{r['backbone']:<18}{r['tau']:>+9.4f}{r['p_perm']:>9.4f}{r['p_param']:>11.2e}"
              f"{r['p_holm']:>9.4f}{'yes' if r['significant'] else 'no':>5}"
              f"   {r['auroc_min']:.4f}–{r['auroc_max']:.4f}")
    print(f"\nS = {S}   Tier 1 = {t1}   Tier 2 = {t2}   ->  Outcome {t1}, {t2}")
    print(f"families reversed (mean τ < 0): {', '.join(reversed_fams) or 'none'}")

    disagree = [r for r in rows if (r["p_perm"] <= 0.05) != (r["p_param"] <= 0.05)]
    print(f"\npermutation vs parametric disagreements at unadjusted 0.05: {len(disagree)}/{len(rows)}")
    for r in disagree:
        print(f"  {r['backbone']:<18} τ={r['tau']:+.4f}  p_perm={r['p_perm']:.4f}  "
              f"p_param={r['p_param']:.4g}")

    # --- sensitivity (preregistered) ------------------------------------------
    print("\nsensitivity")
    sens = {}
    for key, label in [("condition_number_primary_k128", "k=128"),
                       (KAPPA_PRIMARY, "k=256"),
                       ("condition_number_primary_k512", "k=512"),
                       ("condition_number", "kappa_paper4")]:
        ps, taus = [], []
        for b in testable:
            k, a, s = build_population(b, key)
            tau, p, _ = seed_permutation_test(k, a, s)
            ps.append(p); taus.append(tau)
        a_, r_ = holm(ps)
        s_ = sum(1 for t, rr in zip(taus, r_) if rr and np.sign(t) > 0) if any(r_) else 0
        sens[label] = {"S": s_, "tier1": tier1(s_, T)}
        print(f"  {label:<14} S = {s_}   Tier 1 = {tier1(s_, T)}")

    # --- replication check (D-027) --------------------------------------------
    k, a, s = build_population("panderm", "condition_number")
    tau_p4, p_p4, _ = seed_permutation_test(k, a, s)
    print(f"\nreplication (panderm, kappa_paper4): τ = {tau_p4:+.4f}  "
          f"(published +0.5576, Δ = {abs(tau_p4 - 0.5576):.4f})   p_perm = {p_p4:.4f}")

    # --- secondary scorers, outside the correction (D-037) --------------------
    secondary = []
    for b in testable:
        for sc in SECONDARY_SCORERS:
            geom = _load(b, "geometry_metrics")["results"]
            scor = _load(b, "reliability_scorers")["results"]
            gk = {("c", r["seed"], r["alpha"]): r[KAPPA_PRIMARY] for r in geom["canonical"]}
            gk.update({("a", r["seed"], r["rung"]): r[KAPPA_PRIMARY]
                       for r in geom.get("adaptation", [])})
            kk, aa = [], []
            for r in scor["canonical"]:
                if r["alpha"] in LADDER_ALPHAS and sc in r:
                    kk.append(gk[("c", r["seed"], r["alpha"])]); aa.append(r[sc])
            for r in scor.get("adaptation", []):
                if r["rung"] in RUNGS and sc in r:
                    kk.append(gk[("a", r["seed"], r["rung"])]); aa.append(r[sc])
            if len(kk) == 30:
                res = kendalltau(kk, aa)
                secondary.append({"backbone": b, "scorer": sc,
                                  "tau": float(res.statistic), "p_param": float(res.pvalue),
                                  "auroc_min": min(aa), "auroc_max": max(aa)})

    # --- write ----------------------------------------------------------------
    import csv
    with (out_dir / "per_backbone_confirmatory.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with (out_dir / "secondary_scorers.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(secondary[0]))
        w.writeheader(); w.writerows(secondary)
    (out_dir / "summary.json").write_text(json.dumps({
        "freeze_commit": FREEZE, "analysis_commit": head,
        "T": T, "complete_cells": complete_cells, "S": S,
        "tier1": t1, "tier2": t2, "outcome": f"{t1}, {t2}",
        "families_significant": fam_sig, "families_mean_tau": fam_tau,
        "sensitivity": sens,
        "replication_panderm_kappa_paper4": {"tau": tau_p4, "p_perm": p_p4,
                                            "published": 0.5576},
        "permutation_resolution": f"1/{rows[0]['n_perm']}",
    }, indent=2))
    print(f"\nwritten -> {out_dir.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
