#!/usr/bin/env python3
"""
Relative depth alignment for IOI heads (Pythia → GPT-2).
Produces:
  - scatter: src_rel_depth vs tgt_rel_depth
  - hist: |Δ relative depth|
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.fig_style import set_icml_style, savefig_icml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    set_icml_style()

    path = "results/relative_depth_alignment_ioi.csv"
    df = pd.read_csv(path)
    print("[INFO] Loaded relative-depth alignment from", path)
    print("[INFO] Columns:", list(df.columns))

    for col in ["src_rel_depth", "tgt_rel_depth", "rel_depth_diff"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in {list(df.columns)}")

    # Scatter
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(df["src_rel_depth"], df["tgt_rel_depth"], s=14, alpha=0.8)
    ax_scatter.set_xlabel("Source relative depth (Pythia-410M)")
    ax_scatter.set_ylabel("Target relative depth (GPT-2-Medium)")
    ax_scatter.set_title("Relative layer alignment (IOI heads)")
    savefig_icml(fig_scatter, "paper/figs/relative_depth_alignment_ioi_scatter")

    # Histogram of |Δ rel depth|
    fig_hist, ax_hist = plt.subplots()
    rel_abs = np.abs(df["rel_depth_diff"])
    ax_hist.hist(rel_abs, bins=10)
    ax_hist.set_xlabel("|Δ relative depth|")
    ax_hist.set_ylabel("# of heads")
    ax_hist.set_title("Distribution of relative depth mismatch (IOI)")
    savefig_icml(fig_hist, "paper/figs/relative_depth_alignment_ioi_hist")


if __name__ == "__main__":
    main()

