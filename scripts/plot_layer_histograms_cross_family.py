#!/usr/bin/env python
"""
Plot layer distributions of IOI-only vs Anti-only heads for each model.

Reads:
    results/joint_ioi_anti_repeat_all.csv

Saves:
    results/layer_hist_<model>.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
JOINT_PATH = os.path.join(RESULTS_DIR, "joint_ioi_anti_repeat_all.csv")

TAU = 0.05  # threshold for "strong" head


def classify_heads(df_model: pd.DataFrame, tau: float):
    d_ioi = df_model["delta_ioi"].values
    d_anti = df_model["delta_anti"].values

    strong_ioi = np.abs(d_ioi) >= tau
    strong_anti = np.abs(d_anti) >= tau

    ioi_only_mask = np.logical_and(strong_ioi, ~strong_anti)
    anti_only_mask = np.logical_and(strong_anti, ~strong_ioi)

    ioi_only_layers = df_model["layer"].values[ioi_only_mask]
    anti_only_layers = df_model["layer"].values[anti_only_mask]

    return ioi_only_layers, anti_only_layers


def main():
    if not os.path.exists(JOINT_PATH):
        raise FileNotFoundError(f"Joint table not found at {JOINT_PATH}")

    df = pd.read_csv(JOINT_PATH)

    for (family, model), df_m in df.groupby(["family", "model"]):
        ioi_layers, anti_layers = classify_heads(df_m, TAU)

        if len(ioi_layers) == 0 and len(anti_layers) == 0:
            print(f"[INFO] No strong heads for {family}/{model} at τ={TAU}, skipping.")
            continue

        plt.figure(figsize=(6, 4))
        bins = range(int(df_m["layer"].min()), int(df_m["layer"].max()) + 2)

        if len(ioi_layers) > 0:
            plt.hist(
                ioi_layers,
                bins=bins,
                alpha=0.6,
                label="IOI-only",
            )
        if len(anti_layers) > 0:
            plt.hist(
                anti_layers,
                bins=bins,
                alpha=0.6,
                label="Anti-only",
            )

        plt.xlabel("Layer")
        plt.ylabel("Head count")
        plt.title(f"{family}/{model} – strong heads by layer (τ={TAU})")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(RESULTS_DIR, f"layer_hist_{family}_{model}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

