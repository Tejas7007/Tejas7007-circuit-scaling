#!/usr/bin/env python3
"""
Horizontal bar plot of top IOI units by |patch_effect|.
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

    df_sorted = df.sort_values("abs_patch_effect", ascending=True)
    labels = df_sorted["unit_label"]
    values = df_sorted["abs_patch_effect"]

    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    ax.barh(labels, values)
    ax.set_xlabel("|patch effect on IOI margin|")
    ax.set_title("Top GPT-2-M IOI circuit components (path patching)")
    ax.invert_yaxis()  # strongest at top

    fig.tight_layout()
    savefig_icml(fig, "paper/figs/gpt2m_ioi_path_patching_top_units")


if __name__ == "__main__":
    main()

