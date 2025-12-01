#!/usr/bin/env python
"""
Generic transfer analysis for factual recall heads.

Inputs:
  - base_scores CSV
  - target_scores CSV

Outputs:
  results/factual_recall_transfer_{base}_to_{target}.csv
  results/factual_recall_transfer_{base}_to_{target}_summary.txt
"""

import os
import numpy as np
import pandas as pd

def load_scores(path):
    return pd.read_csv(path)

def main(base_scores_path, target_scores_path, top_k=20, tau=0.015):
    os.makedirs("results", exist_ok=True)

    base_df = load_scores(base_scores_path)
    tgt_df = load_scores(target_scores_path)

    base_model = base_df["model"].iloc[0]
    tgt_model = tgt_df["model"].iloc[0]

    base_df["strength"] = -base_df["delta_margin"]
    base_top = base_df.sort_values("strength", ascending=False).head(top_k)

    merged = base_top.merge(
        tgt_df[["layer", "head", "delta_margin"]]
            .rename(columns={"delta_margin": "target_delta_margin"}),
        on=["layer", "head"],
        how="left",
    )

    def classify(row):
        base = row["delta_margin"]
        tgt = row["target_delta_margin"]
        if pd.isna(tgt):
            return "missing"
        if abs(tgt) < tau:
            return "weak"
        if np.sign(tgt) == np.sign(base):
            return "same-sign"
        return "inverted"

    merged["outcome"] = merged.apply(classify, axis=1)

    tgt_df["strength"] = -tgt_df["delta_margin"]
    tgt_df = tgt_df.sort_values("strength", ascending=False).reset_index(drop=True)
    tgt_df["rank"] = np.arange(len(tgt_df)) + 1
    tgt_df["frac_rank"] = tgt_df["rank"] / len(tgt_df)

    merged = merged.merge(
        tgt_df[["layer", "head", "frac_rank"]],
        on=["layer", "head"],
        how="left",
    )

    base_s = base_model.replace("/", "_")
    tgt_s = tgt_model.replace("/", "_")
    csv_path = f"results/factual_recall_transfer_{base_s}_to_{tgt_s}.csv"
    merged.to_csv(csv_path, index=False)

    summary = {
        "total": len(merged),
        "missing": (merged["outcome"] == "missing").sum(),
        "weak": (merged["outcome"] == "weak").sum(),
        "same-sign": (merged["outcome"] == "same-sign").sum(),
        "inverted": (merged["outcome"] == "inverted").sum(),
        "mean_frac_rank": merged["frac_rank"].mean(skipna=True)
    }

    txt_path = f"results/factual_recall_transfer_{base_s}_to_{tgt_s}_summary.txt"
    with open(txt_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"Wrote transfer table: {csv_path}")
    print(f"Wrote summary: {txt_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_scores_path", type=str, required=True)
    parser.add_argument("--target_scores_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.015)
    args = parser.parse_args()
    main(args.base_scores_path, args.target_scores_path, args.top_k, args.tau)

