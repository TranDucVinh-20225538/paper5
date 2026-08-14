#!/usr/bin/env python3
"""Figures for the manuscript. Reads only committed confirmatory outputs."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "csv" / "confirmatory"
OUT = REPO / "results" / "figures"
FAM = {"resnet50":"CNN","efficientnet_b3":"CNN","panderm":"med-SSL","uni":"med-SSL",
       "biomedclip":"med-VLM","monet":"med-VLM","dinov3":"gen-SSL","mocov3":"gen-SSL",
       "openclip":"gen-VLM","siglip":"gen-VLM"}
COL = {"CNN":"#4C72B0","med-SSL":"#DD8452","med-VLM":"#55A868",
       "gen-SSL":"#C44E52","gen-VLM":"#8172B3"}

rows = list(csv.DictReader((SRC/"per_backbone_confirmatory.csv").open()))
for r in rows:
    for k in ("tau","p_perm","p_param","auroc_min","auroc_max"): r[k]=float(r[k])
rows.sort(key=lambda r: r["tau"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios":[1.15,1]})

# ---- Panel A: tau per backbone, ordered ------------------------------------
y = np.arange(len(rows))
for i, r in enumerate(rows):
    c = COL[FAM[r["backbone"]]]
    ax[0].plot([0, r["tau"]], [i, i], color=c, lw=1.4, alpha=.55, zorder=1)
    ax[0].scatter(r["tau"], i, s=58, color=c, zorder=3, edgecolor="white", lw=.8)
ax[0].axvline(0, color="0.35", lw=1, zorder=2)
ax[0].set_yticks(y); ax[0].set_yticklabels([r["backbone"] for r in rows], fontsize=9)
ax[0].set_xlabel("Kendall's τ  (κ$_{primary}$ vs Mahalanobis AUROC,  n = 30)")
ax[0].set_xlim(-0.72, 0.72)
ax[0].set_title("A   Association per backbone", loc="left", fontsize=11, fontweight="bold")
ax[0].spines[["top","right"]].set_visible(False)
handles=[plt.Line2D([],[],marker="o",ls="",color=c,label=f) for f,c in COL.items()]
ax[0].legend(handles=handles, fontsize=7.5, loc="lower right", frameon=False, ncol=1)

# ---- Panel B: permutation vs parametric ------------------------------------
# Both axes are log. On a linear y-axis the permutation floor (1/120 = 0.0083)
# and Holm's first threshold (0.005) are visually identical, which is precisely
# the comparison this panel exists to show.
FLOOR = 1/120
for r in rows:
    c = COL[FAM[r["backbone"]]]
    ax[1].scatter(max(r["p_param"],1e-6), r["p_perm"], s=58, color=c,
                  edgecolor="white", lw=.8, zorder=4)
ax[1].axhspan(1e-3, .005, color="#F2DEDE", zorder=0)          # unreachable by Holm
ax[1].axhline(.005,  color="#B03A2E", lw=1.3, zorder=2)
ax[1].axhline(FLOOR, color="0.25", ls=":", lw=1.4, zorder=2)
ax[1].axvline(.05,   color="0.65", ls="--", lw=1, zorder=1)
ax[1].axhline(.05,   color="0.65", ls="--", lw=1, zorder=1)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlim(5e-7, 1.6); ax[1].set_ylim(1.4e-3, 1.6)
ax[1].set_xlabel("p  (parametric, log scale)")
ax[1].set_ylabel("p  (seed-level permutation, log scale)")
ax[1].set_title("B   The two procedures disagree", loc="left", fontsize=11, fontweight="bold")
ax[1].annotate("permutation floor  1/120 = 0.0083", xy=(6e-7, FLOOR), xytext=(6e-7, .0155),
               fontsize=7.4, color="0.2")
ax[1].annotate("Holm threshold  α/m = 0.005", xy=(6e-7, .005), xytext=(6e-7, .0034),
               fontsize=7.4, color="#B03A2E")
ax[1].annotate("no permutation p can enter this band", xy=(6e-7, .0022),
               fontsize=7.0, color="#B03A2E", style="italic")
ax[1].annotate("unadjusted 0.05", xy=(0.62, .057), fontsize=7.0, color="0.5")
ax[1].spines[["top","right"]].set_visible(False)

fig.tight_layout()
for ext in ("pdf","png"):
    fig.savefig(OUT/f"fig1_confirmatory.{ext}", dpi=300, bbox_inches="tight")

# ---- Figure 2: AUROC ranges (D-038 made visual) -----------------------------
fig2, ax2 = plt.subplots(figsize=(6.2, 4.0))
rr = sorted(rows, key=lambda r: r["auroc_min"])
for i, r in enumerate(rr):
    c = COL[FAM[r["backbone"]]]
    ax2.plot([r["auroc_min"], r["auroc_max"]], [i, i], color=c, lw=4, solid_capstyle="round")
    ax2.text(r["auroc_max"]+.004, i, f"Δ={r['auroc_max']-r['auroc_min']:.4f}",
             fontsize=7.2, va="center", color="0.3")
ax2.set_yticks(range(len(rr))); ax2.set_yticklabels([r["backbone"] for r in rr], fontsize=9)
ax2.set_xlabel("Mahalanobis AUROC across the n = 30 analysis population")
ax2.set_xlim(.77, 1.03)
ax2.set_title("Outcome variation is narrow on every backbone",
              loc="left", fontsize=11, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
fig2.tight_layout()
for ext in ("pdf","png"):
    fig2.savefig(OUT/f"fig2_auroc_range.{ext}", dpi=300, bbox_inches="tight")
print("wrote fig1_confirmatory and fig2_auroc_range (pdf + png)")
