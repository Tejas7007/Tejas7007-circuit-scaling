#!/usr/bin/env python
"""
Summarize IOI vs anti-repeat deltas per head, with categorical labels.

Reads:  results/joint_ioi_anti_repeat_all.csv
Writes: results/joint_ioi_anti_repeat_heads.csv
Also prints some top heads per model to stdout.
"""

import argparse
import os
import pandas as pd


def categorize_head(delta_ioi: float, delta_anti: float, tau: float) -> str:
    """Return a string label for the head category."""
    abs_ioi = abs(delta_ioi)
    abs_anti = abs(delta_anti)

    ioi_strong = abs_ioi >= tau
    anti_strong = abs_anti >= tau

    if ioi_strong and anti_strong:
        return "shared"
    elif ioi_strong and not anti_strong:
        return "ioi_only"
    elif anti_strong and not ioi_strong:
        return "anti_only"
    else:
        return "weak"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="Threshold |Δ| >= τ to call a head 'strong' for IOI or anti-repeat.",
    )
    parser.add_argument(
        "--in_csv",
        type=str,
        default=os.path.join("results", "joint_ioi_anti_repeat_all.csv"),
        help="Path to unified per-head CSV.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=os.path.join("results", "joint_ioi_anti_repeat_heads.csv"),
        help="Where to write the per-head summary CSV.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="How many top heads per category to print per model.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.in_csv):
        raise FileNotFoundError(f"Cannot find {args.in_csv}")

    df = pd.read_csv(args.in_csv)
    required_cols = {"family", "model", "layer", "head", "delta_ioi", "delta_anti"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    # Add derived columns
    df["abs_delta_ioi"] = df["delta_ioi"].abs()
    df["abs_delta_anti"] = df["delta_anti"].abs()
    df["strength"] = df["abs_delta_ioi"] + df["abs_delta_anti"]
    df["category"] = [
        categorize_head(row.delta_ioi, row.delta_anti, args.tau)
        for row in df.itertuples()
    ]

    # Write out full summary
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote per-head summary to {args.out_csv}")

    # Pretty-print some top heads per model
    grouped = df.groupby(["family", "model"])

    for (family, model), g in grouped:
        print()
        print(f"=== {family}/{model} (τ={args.tau}) ===")
        for cat in ["shared", "ioi_only", "anti_only"]:
            sub = g[g["category"] == cat].sort_values("strength", ascending=False)
            if sub.empty:
                print(f"  No {cat} heads.")
                continue

            print(f"  Top {cat} heads (up to {args.topk}):")
            for row in sub.head(args.topk).itertuples():
                print(
                    f"    L{row.layer}H{row.head}  "
                    f"Δ_ioi={row.delta_ioi:+.4f}  Δ_anti={row.delta_anti:+.4f}  "
                    f"strength={row.strength:.4f}"
                )


if __name__ == "__main__":
    main()

