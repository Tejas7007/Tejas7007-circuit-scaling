import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Recompute correlation between Δ_ioi and Δ_anti for each (family, model),
and plot correlation vs. model scale (approx parameter count in millions).

Reads:
  results/joint_ioi_anti_repeat_all.csv

Writes:
  results/corr_vs_scale_cross_family.csv
  results/corr_vs_scale_cross_family.png
"""

# Rough parameter sizes in millions
PARAMS_M = {
    # Pythia
    ("pythia", "pythia-70m"): 70,
    ("pythia", "pythia-160m"): 160,
    ("pythia", "pythia-410m"): 410,
    ("pythia", "pythia-1b"): 1000,

    # GPT-2 family (approx)
    ("gpt2", "gpt2"): 124,
    ("gpt2", "gpt2-medium"): 355,
    ("gpt2", "gpt2-large"): 774,

    # OPT
    ("opt", "opt-125m"): 125,

    # GPT-Neo
    ("gpt-neo", "gpt-neo-125M"): 125,
}


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson corr, returning NaN if undefined."""
    if len(x) < 2:
        return math.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def main():
    # 1. Load joint table
    joint_path = os.path.join("results", "joint_ioi_anti_repeat_all.csv")
    if not os.path.exists(joint_path):
        raise FileNotFoundError(f"{joint_path} not found. Run joint_ioi_anti_repeat_all.py first.")

    df = pd.read_csv(joint_path)
    print("joint_ioi_anti_repeat_all columns:", df.columns.tolist())

    required_cols = {"family", "model", "delta_ioi", "delta_anti"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in joint table: {missing}")

    # 2. Compute correlation per (family, model)
    rows = []
    for (family, model), g in df.groupby(["family", "model"]):
        corr = safe_corr(g["delta_ioi"].to_numpy(), g["delta_anti"].to_numpy())
        n_heads = len(g)
        params_m = PARAMS_M.get((family, model), math.nan)

        rows.append(
            {
                "family": family,
                "model": model,
                "params_m": params_m,
                "corr_delta_ioi_anti": corr,
                "n_heads": n_heads,
            }
        )

    summary = pd.DataFrame(rows)
    print("summary columns:", summary.columns.tolist())
    print(summary)

    # 3. Save summary CSV
    out_csv = os.path.join("results", "corr_vs_scale_cross_family.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # 4. Make scatter plot: correlation vs. params, colored by family
    plt.figure(figsize=(7, 5))

    families = sorted(summary["family"].unique())
    for fam in families:
        sub = summary[summary["family"] == fam]
        plt.scatter(
            sub["params_m"],
            sub["corr_delta_ioi_anti"],
            label=fam,
        )
        for _, row in sub.iterrows():
            if not math.isnan(row["params_m"]):
                short_name = (
                    row["model"]
                    .replace("pythia-", "")
                    .replace("gpt2-", "")
                    .replace("-125M", "")
                    .replace("gpt-neo-", "")
                    .replace("opt-", "")
                )
                plt.text(
                    row["params_m"],
                    row["corr_delta_ioi_anti"],
                    short_name,
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

