#!/usr/bin/env python
"""
Analyze joint IOI vs anti-repeat table for GPT-2 family.

Reads:
    results/joint_ioi_anti_repeat_gpt2.csv

Prints, for several thresholds τ:
  - head counts by category (IOI-only, Anti-only, Shared, Weak/none)
  - Pearson corr(delta_ioi, delta_anti) per model
"""

import os

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
JOINT_PATH = os.path.join(RESULTS_DIR, "joint_ioi_anti_repeat_gpt2.csv")


def classify_heads(df_model: pd.DataFrame, tau: float):
    """
    Return category counts for a given model and threshold tau.
    We consider a head "strong" if |delta| >= tau.
    """
    d_ioi = df_model["delta_ioi"].values
    d_anti = df_model["delta_anti"].values

    strong_ioi = np.abs(d_ioi) >= tau
    strong_anti = np.abs(d_anti) >= tau

    ioi_only = np.logical_and(strong_ioi, ~strong_anti)
    anti_only = np.logical_and(strong_anti, ~strong_ioi)
    shared = np.logical_and(strong_ioi, strong_anti)
    weak = np.logical_and(~strong_ioi, ~strong_anti)

    counts = {
        "IOI-only": int(ioi_only.sum()),
        "Anti-only": int(anti_only.sum()),
        "Shared": int(shared.sum()),
        "Weak/none": int(weak.sum()),
        "Total": len(df_model),
    }
    return counts


def main():
    if not os.path.exists(JOINT_PATH):
        raise FileNotFoundError(f"Joint GPT-2 table not found at {JOINT_PATH}")

    df = pd.read_csv(JOINT_PATH)
    models = sorted(df["model"].unique())
    taus = [0.03, 0.05, 0.07, 0.10]

    for tau in taus:
        print()
        print(f"========== τ = {tau:.2f} ==========")
        for model in models:
            df_m = df[df["model"] == model].copy()
            if len(df_m) == 0:
                continue

            corr = df_m["delta_ioi"].corr(df_m["delta_anti"])
            counts = classify_heads(df_m, tau)

            print(f"Model: {model}")
            print(f"  heads in joint table: {counts['Total']}")
            print(
                f"  IOI-only: {counts['IOI-only']}, "
                f"Anti-only: {counts['Anti-only']}, "
                f"Shared: {counts['Shared']}, "
                f"Weak/none: {counts['Weak/none']}"
            )
            print(f"  Pearson corr(Δ_ioi, Δ_anti): {corr:.3f}")
            print()


if __name__ == "__main__":
    main()

