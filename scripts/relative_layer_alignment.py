#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

JOINT_PATH = "paper/tables/joint_ioi_anti_repeat_all.csv"
OUT_TABLE = "paper/tables/ioi_relative_layer_positions.csv"
OUT_FIG = "paper/figs/ioi_relative_layer_violin.png"

# Threshold for considering a head "IOI-like"
IOI_THRESH = 0.15


def main():
    # Load joint head metrics
    df = pd.read_csv(JOINT_PATH)

    required_cols = {"family", "model", "layer", "head", "delta_ioi"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {JOINT_PATH}: {missing}")

    # Infer number of layers per (family, model) from max(layer)+1
    df["n_layers"] = df.groupby(["family", "model"])["layer"].transform(
        lambda s: int(s.max()) + 1
    )

    # Relative layer depth in (0, 1]
    df["relative_layer"] = (df["layer"] + 0.5) / df["n_layers"]

    # Filter IOI-like heads
    ioi_heads = df[df["delta_ioi"].abs() >= IOI_THRESH].copy()
    if ioi_heads.empty:
        raise ValueError(
            f"No IOI-like heads found with |delta_ioi| >= {IOI_THRESH}. "
            "Try lowering IOI_THRESH in scripts/relative_layer_alignment.py."
        )

    # Save table for further analysis
    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
    ioi_heads.to_csv(OUT_TABLE, index=False)
    print(f"[TABLE] Saved IOI relative layer positions to {OUT_TABLE}")

    # Prepare data for plot: one distribution per model
    models = sorted(ioi_heads["model"].unique())
    data = [ioi_heads[ioi_heads["model"] == m]["relative_layer"].values for m in models]

    positions = np.arange(len(models))

    plt.figure(figsize=(max(8, len(models) * 0.7), 5))

    plt.violinplot(
        data,
        positions=positions,
        showmeans=True,
        showextrema=False,
    )

    plt.xticks(positions, models, rotation=45, ha="right")
    plt.ylabel("Relative layer (0 = early, 1 = late)")
    plt.title(f"Relative depth of IOI-like heads (|Δ_ioi| ≥ {IOI_THRESH})")
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()
    print(f"[FIG] Saved violin plot to {OUT_FIG}")


if __name__ == "__main__":
    main()

