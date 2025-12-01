#!/usr/bin/env python3
"""
Barplot of top IOI heads + MLP blocks for GPT-2-medium.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.fig_style import set_icml_style, savefig_icml
import pandas as pd
import matplotlib.pyplot as plt


def main():
    set_icml_style()

    heads_path = "results/global_ioi_gpt2medium_top_heads.csv"
    mlps_path = "results/global_ioi_gpt2medium_top_mlps.csv"

    heads = pd.read_csv(heads_path)
    mlps = pd.read_csv(mlps_path)

    print("[INFO] Heads columns:", list(heads.columns))
    print("[INFO] MLP columns:", list(mlps.columns))

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), sharey=True)

    # Heads
    heads_sorted = heads.sort_values("abs_effect", ascending=False)
    head_labels = [f"L{l}H{h}" for l, h in zip(heads_sorted["layer"], heads_sorted["head"])]
    axes[0].bar(range(len(heads_sorted)), heads_sorted["abs_effect"])
    axes[0].set_xticks(range(len(heads_sorted)))
    axes[0].set_xticklabels(head_labels, rotation=45, ha="right")
    axes[0].set_ylabel("|Δ IOI margin|")
    axes[0].set_title("Top IOI attention heads")

    # MLPs
    mlps_sorted = mlps.sort_values("abs_effect", ascending=False)
    mlp_labels = [f"MLP L{l}" for l in mlps_sorted["layer"]]
    axes[1].bar(range(len(mlps_sorted)), mlps_sorted["abs_effect"])
    axes[1].set_xticks(range(len(mlps_sorted)))
    axes[1].set_xticklabels(mlp_labels, rotation=45, ha="right")
    axes[1].set_title("Top IOI MLP blocks")

    fig.tight_layout()
    savefig_icml(fig, "paper/figs/gpt2medium_global_ioi_circuit")


if __name__ == "__main__":
    main()

