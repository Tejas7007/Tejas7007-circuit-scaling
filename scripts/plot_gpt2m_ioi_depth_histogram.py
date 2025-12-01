#!/usr/bin/env python3
"""
Histogram of layer indices for top-K IOI circuit components (GPT-2-M).
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

    path = "paper/tables/gpt2medium_ioi_circuit_components.csv"
    df = pd.read_csv(path)
    print("[INFO] Sample rows:")
    print(df.head())

    fig, ax = plt.subplots()
    ax.hist(df["layer"], bins=24, range=(0, 23), align="left")
    ax.set_xlabel("Layer index (GPT-2-Medium)")
    ax.set_ylabel("# of top IOI units")
    ax.set_title("Depth distribution of IOI circuit components (top-K)")

    fig.tight_layout()
    savefig_icml(fig, "paper/figs/gpt2m_ioi_depth_hist_topK")


if __name__ == "__main__":
    main()

