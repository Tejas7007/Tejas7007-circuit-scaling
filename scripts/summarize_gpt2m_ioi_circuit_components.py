#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize GPT-2 Medium IOI circuit components from exhaustive "
            "path-patching results."
        )
    )
    parser.add_argument(
        "--in_csv",
        type=str,
        required=True,
        help="Input path-patching CSV (e.g. results/gpt2medium_ioi_path_patching.csv)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV for top-k components "
             "(e.g. paper/tables/gpt2medium_ioi_circuit_components.csv)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=32,
        help="Number of strongest units (by |patch_effect|) to keep.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading path-patching CSV from {args.in_csv}")
    df = pd.read_csv(args.in_csv)

    # -------------------------------------------------------------------------
    # Ensure we have abs_patch_effect
    # -------------------------------------------------------------------------
    if "abs_patch_effect" not in df.columns:
        if "patch_effect" not in df.columns:
            raise ValueError(
                f"{args.in_csv} is missing both 'patch_effect' and "
                f"'abs_patch_effect'. Columns: {list(df.columns)}"
            )
        print("[INFO] 'abs_patch_effect' not found; creating it from |patch_effect|.")
        df["abs_patch_effect"] = df["patch_effect"].abs()

    # -------------------------------------------------------------------------
    # Basic sanity check on required columns
    # -------------------------------------------------------------------------
    required = [
        "type",
        "layer",
        "head",
        "baseline_mean_margin_corrupt",
        "clean_mean_margin",
        "patched_mean_margin",
        "total_clean_corrupt_gap",
        "patch_effect",
        "abs_patch_effect",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in {args.in_csv}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"[INFO] Loaded {len(df)} rows.")
    print("[INFO] Example columns:", list(df.columns))

    # -------------------------------------------------------------------------
    # Sort by absolute patch effect and take top_k units
    # -------------------------------------------------------------------------
    df_sorted = df.sort_values("abs_patch_effect", ascending=False).reset_index(drop=True)
    top_k = min(args.top_k, len(df_sorted))
    top = df_sorted.iloc[:top_k].copy()
    print(f"[INFO] Selecting top {top_k} units by |patch_effect|.")

    # -------------------------------------------------------------------------
    # Add a human-readable label for each unit
    #   attn: "L{layer}H{head}"
    #   mlp:  "MLP_L{layer}"
    # -------------------------------------------------------------------------
    def make_label(row):
        if row["type"] == "attn":
            return f"L{int(row['layer'])}H{int(row['head'])}"
        elif row["type"] == "mlp":
            return f"MLP_L{int(row['layer'])}"
        else:
            return f"{row['type']}_L{int(row['layer'])}H{int(row['head'])}"

    top["unit_label"] = top.apply(make_label, axis=1)

    # -------------------------------------------------------------------------
    # Re-order / trim columns for a nice paper table
    # -------------------------------------------------------------------------
    cols_out = [
        "type",
        "layer",
        "head",
        "unit_label",
        "baseline_mean_margin_corrupt",
        "clean_mean_margin",
        "patched_mean_margin",
        "total_clean_corrupt_gap",
        "patch_effect",
        "abs_patch_effect",
    ]
    # Keep only the columns that actually exist (in case some extras are missing)
    cols_out = [c for c in cols_out if c in top.columns]
    top = top[cols_out]

    # -------------------------------------------------------------------------
    # Save and print preview
    # -------------------------------------------------------------------------
    top.to_csv(args.out_csv, index=False)
    print(f"[OUT] Wrote GPT-2-M IOI circuit components to {args.out_csv}")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()

