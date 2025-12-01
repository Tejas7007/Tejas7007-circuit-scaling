#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import scipy.stats as stats

IN_PATH = "paper/tables/tokenization_mismatch_ioi.csv"
OUT_PATH = "paper/tables/architecture_control_ioi.csv"


def summarize_group(name: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "group": name,
            "n": 0,
            "mean_abs_delta_tokens": np.nan,
            "mean_abs_ioi_diff": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    # Raw strengths
    x = df["ioistrength_pythia"].values
    y = df["ioistrength_gpt2"].values

    # Differences
    abs_delta_tokens = df["abs_delta_tokens"].values
    abs_ioi_diff = df["abs_ioi_diff"].values

    if len(df) >= 2:
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        pearson_r, pearson_p = pearson.statistic, pearson.pvalue
        spearman_rho, spearman_p = spearman.statistic, spearman.pvalue
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = np.nan

    return {
        "group": name,
        "n": len(df),
        "mean_abs_delta_tokens": float(np.nanmean(abs_delta_tokens)),
        "mean_abs_ioi_diff": float(np.nanmean(abs_ioi_diff)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"{IN_PATH} not found. Run analyze_tokenization_mismatch.py first.")

    df = pd.read_csv(IN_PATH)

    required = {
        "name",
        "ioistrength_pythia",
        "ioistrength_gpt2",
        "abs_delta_tokens",
        "abs_ioi_diff",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {IN_PATH}: {missing}")

    # Split groups
    matched = df[df["abs_delta_tokens"] == 0].copy()
    mismatched = df[df["abs_delta_tokens"] > 0].copy()

    print(f"Total names: {len(df)}")
    print(f"  Matched-tokenization names:   {len(matched)}")
    print(f"  Mismatched-tokenization names:{len(mismatched)}")

    rows = []
    rows.append(summarize_group("all", df))
    rows.append(summarize_group("matched", matched))
    rows.append(summarize_group("mismatched", mismatched))

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print("\n=== Architecture-only IOI comparison ===")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

