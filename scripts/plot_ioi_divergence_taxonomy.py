#!/usr/bin/env python3
"""
Bar chart of divergence types in IOI transfer Pythia-410M → GPT-2-M.
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

    path = "results/transfer/ioi_divergence_pythia410m_to_gpt2medium.csv"
    df = pd.read_csv(path)
    print("[INFO] Loaded divergence taxonomy from", path)
    print("[INFO] Columns:", list(df.columns))

    if "divergence_type" not in df.columns:
        raise ValueError(
            f"Expected 'divergence_type' column missing; columns are {list(df.columns)}"
        )

    counts = df["divergence_type"].value_counts().sort_values(ascending=False)
    print(counts)

    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("# of IOI-like heads")
    ax.set_title("IOI transfer divergence taxonomy\n(Pythia-410M → GPT-2-Medium)")
    ax.set_xticklabels(counts.index, rotation=30, ha="right")

    fig.tight_layout()
    savefig_icml(fig, "paper/figs/ioi_divergence_taxonomy_pythia410m_to_gpt2medium")


if __name__ == "__main__":
    main()

