#!/usr/bin/env python3
"""
Scatter: tokenization mismatch vs IOI distortion.
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

    path = "paper/tables/tokenization_mismatch_ioi.csv"
    print(f"[INFO] Loaded tokenization mismatch table from {path}")
    df = pd.read_csv(path)
    print("[INFO] Columns:", list(df.columns))

    x_col = "abs_delta_tokens"
    y_col = "abs_ioi_diff"
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Expected columns '{x_col}', '{y_col}' missing.")

    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col], s=12, alpha=0.8)
    ax.set_xlabel("|Δ #tokens between Pythia and GPT-2|")
    ax.set_ylabel("|Δ IOI logit-diff|")
    ax.set_title("Tokenization mismatch vs IOI distortion")

    savefig_icml(fig, "paper/figs/tokenization_mismatch_scatter")


if __name__ == "__main__":
    main()

