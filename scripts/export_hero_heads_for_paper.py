#!/usr/bin/env python
"""
Export top heads per model/category into a clean CSV for tables in the paper.

Reads:  results/joint_ioi_anti_repeat_heads.csv
Writes: results/hero_heads_for_paper.csv
"""

import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_csv",
        type=str,
        default=os.path.join("results", "joint_ioi_anti_repeat_heads.csv"),
        help="Per-head summary CSV.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=os.path.join("results", "hero_heads_for_paper.csv"),
        help="Where to write the filtered hero table.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Top-k per (family, model, category) to keep by strength.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.in_csv):
        raise FileNotFoundError(f"Cannot find {args.in_csv}")

    df = pd.read_csv(args.in_csv)

    required = {
        "family",
        "model",
        "layer",
        "head",
        "delta_ioi",
        "delta_anti",
        "strength",
        "category",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.in_csv}: {missing}")

    # limit to models for paper
    keep_models = [
        ("gpt2", "gpt2-large"),
        ("pythia", "pythia-160m"),
        ("pythia", "pythia-70m"),
    ]
    mask = False
    for fam, mod in keep_models:
        mask |= ((df["family"] == fam) & (df["model"] == mod))
    df = df[mask].copy()

    df = df.sort_values("strength", ascending=False)

    rows = []
    grouped = df.groupby(["family", "model", "category"], sort=True)
    for (family, model, category), g in grouped:
        top = g.head(args.topk)
        for row in top.itertuples():
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "category": category,
                    "layer": row.layer,
                    "head": row.head,
                    "delta_ioi": row.delta_ioi,
                    "delta_anti": row.delta_anti,
                    "strength": row.strength,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["family", "model", "category", "layer", "head"])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote hero head table to {args.out_csv}")

if __name__ == "__main__":
    main()

