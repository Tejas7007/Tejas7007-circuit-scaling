#!/usr/bin/env python3
"""
plot_divergence_taxonomy_expanded.py

Reads results/ioi_divergence_taxonomy_expanded.csv and makes
a simple bar plot of counts per taxonomy class.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "results" / "ioi_divergence_taxonomy_expanded.csv"
OUT_PATH = ROOT / "figs" / "ioi_divergence_taxonomy_expanded_bar.png"

def main():
    df = pd.read_csv(IN_PATH)
    counts = df["taxonomy"].value_counts().sort_index()

    plt.figure(figsize=(5, 3))
    counts.plot(kind="bar")
    plt.ylabel("Number of Pythia-410M IOI heads")
    plt.xlabel("Divergence category (Pythia-410M → GPT-2-Medium)")
    plt.title("Expanded divergence taxonomy for IOI heads")
    plt.tight_layout()

    OUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300)
    print(f"[OUT] Saved expanded divergence taxonomy bar plot to {OUT_PATH}")

if __name__ == "__main__":
    main()

