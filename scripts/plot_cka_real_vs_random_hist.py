#!/usr/bin/env python3
"""
Overlay histogram: real head-to-head CKA vs random baseline.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.fig_style import set_icml_style, savefig_icml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    set_icml_style()

    real_path = "results/cka_alignment_pythia410m_gpt2medium.csv"
    rand_path = "results/cka_random_baseline_pythia-410m_gpt2-medium.npz"

    print(f"[INFO] Loaded real CKA from {real_path}")
    cka = pd.read_csv(real_path)
    print("[INFO] Real CKA columns:", list(cka.columns))

    if "cka" not in cka.columns:
        raise ValueError(f"No 'cka' column found in {real_path}")

    cka_vals = cka["cka"].to_numpy()

    rand_npz = np.load(rand_path)
    print("[INFO] Loaded random baseline from", rand_path)
    print("[INFO] NPZ keys:", list(rand_npz.keys()))

    if "cka_vals" not in rand_npz:
        raise ValueError(f"'cka_vals' not found in {rand_path}")
    rand_vals = rand_npz["cka_vals"]

    fig, ax = plt.subplots()
    ax.hist(
        rand_vals,
        bins=40,
        alpha=0.7,
        label="Random baseline",
        density=True,
    )
    ax.hist(
        cka_vals,
        bins=40,
        alpha=0.7,
        label="Pythia ↔ GPT-2 heads",
        density=True,
    )
    ax.set_xlabel("CKA similarity")
    ax.set_ylabel("Density")
    ax.set_title("Head-to-head CKA: real vs random baseline")
    ax.legend()

    savefig_icml(fig, "paper/figs/cka_real_vs_random_hist")


if __name__ == "__main__":
    main()

