import os
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Approx parameter counts in millions (for plotting on x-axis)
MODEL_PARAMS_M: Dict[Tuple[str, str], float] = {
    ("pythia", "pythia-70m"): 70.0,
    ("pythia", "pythia-160m"): 160.0,
    ("pythia", "pythia-410m"): 410.0,
    ("pythia", "pythia-1b"): 1000.0,

    ("gpt-neo", "gpt-neo-125M"): 125.0,

    ("gpt2", "gpt2"): 124.0,
    ("gpt2", "gpt2-medium"): 355.0,
    ("gpt2", "gpt2-large"): 774.0,

    ("opt", "opt-125m"): 125.0,
}


def main():
    joint_path = os.path.join("results", "joint_ioi_anti_repeat_all.csv")
    if not os.path.exists(joint_path):
        raise FileNotFoundError(f"Missing {joint_path}. Run joint_ioi_anti_repeat_all.py first.")

    joint = pd.read_csv(joint_path)
    print("joint_ioi_anti_repeat_all columns:", list(joint.columns))

    if joint.empty:
        print("Joint table is empty; cannot compute correlations.")
        return

    # ------------------------------------------------------------------
    # Build summary: per (family, model), correlation between Δ_ioi & Δ_anti
    # ------------------------------------------------------------------
    rows: List[Dict] = []

    for (family, model), group in joint.groupby(["family", "model"]):
        if len(group) < 2:
            # not enough heads to compute correlation
            continue

        corr = group[["delta_ioi", "delta_anti"]].corr().iloc[0, 1]
        params = MODEL_PARAMS_M.get((family, model), None)

        rows.append(
            {
                "family": family,
                "model": model,
                "params_m": params,
                "corr": corr,
            }
        )

    summary = pd.DataFrame(rows, columns=["family", "model", "params_m", "corr"])
    print("summary columns:", list(summary.columns))
    print(summary)

    out_csv = os.path.join("results", "corr_vs_scale_cross_family.csv")
    os.makedirs("results", exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    if summary.empty:
        print("Summary is empty; skipping plot generation.")
        return

    # ------------------------------------------------------------------
    # Plot correlation vs scale for each family
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 4))

    families = sorted(summary["family"].unique())
    for fam in families:
        fam_df = summary[summary["family"] == fam].copy()
        # drop models without param estimates
        fam_df = fam_df.dropna(subset=["params_m"])
        if fam_df.empty:
            continue

        fam_df = fam_df.sort_values("params_m")
        x = fam_df["params_m"].values
        y = fam_df["corr"].values
        labels = fam_df["model"].values

        plt.plot(x, y, marker="o", label=fam)
        for xv, yv, name in zip(x, y, labels):
            plt.text(
                xv,
                yv,
                name,
                fontsize=7,
                ha="center",
                va="bottom",
            )

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Model size (approx params, millions)")
    plt.ylabel("Pearson corr(Δ_ioi, Δ_anti)")
    plt.title("IOI vs anti-repeat correlation across model families")
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join("results", "corr_vs_scale_cross_family.png")
    plt.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

