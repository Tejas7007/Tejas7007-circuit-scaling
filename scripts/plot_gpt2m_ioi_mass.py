#!/usr/bin/env python3
"""
Count how many top IOI units are attention vs MLP for GPT-2-M.
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
    counts = df["type"].value_counts()
    print(counts)

    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("# of top IOI units")
    ax.set_title("IOI mass in GPT-2-Medium\n(attention vs MLP)")

    fig.tight_layout()
    savefig_icml(fig, "paper/figs/gpt2m_ioi_mass_attention_vs_mlp")


if __name__ == "__main__":
    main()

