#!/usr/bin/env python
"""
threshold_sensitivity_ioi_transfer.py

Recompute IOI transfer stats for multiple thresholds tau on target_abs_delta_ioi.

Inputs:
    paper/tables/ioi_transfer_generic.csv

Outputs:
    results/ioi_transfer_threshold_sensitivity.csv

Each row: base_family, base_model, tgt_family, tgt_model, tau,
          n_heads, n_missing, frac_missing,
          mean_tgt_abs_delta_ioi, std_tgt_abs_delta_ioi,
          mean_tgt_frac_rank, std_tgt_frac_rank
"""

import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "paper" / "tables" / "ioi_transfer_generic.csv"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "ioi_transfer_threshold_sensitivity.csv"

# These thresholds are what we'll report in the paper
TAUS = [0.010, 0.015, 0.020, 0.030]


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Could not find input CSV at: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    required_cols = [
        "base_family",
        "base_model",
        "tgt_family",
        "tgt_model",
        "layer",
        "head",
        "tgt_abs_delta_ioi",
        "tgt_frac_rank",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"Expected column '{c}' in {IN_PATH}, "
                f"but available columns are: {list(df.columns)}"
            )

    # missing_in_target_index might or might not be present / bool / int
    has_missing_col = "missing_in_target_index" in df.columns
    if has_missing_col:
        # Normalize to bool
        miss_raw = df["missing_in_target_index"]
        if miss_raw.dtype == bool:
            df["missing_arch"] = miss_raw
        else:
            # treat non-zero as True
            df["missing_arch"] = miss_raw.fillna(0).astype(int) != 0
    else:
        df["missing_arch"] = False

    rows = []

    group_cols = ["base_family", "base_model", "tgt_family", "tgt_model"]
    grouped = df.groupby(group_cols, dropna=False)

    for (base_fam, base_model, tgt_fam, tgt_model), g in grouped:
        n_heads = len(g)
        if n_heads == 0:
            continue

        for tau in TAUS:
            # missing because of architecture (no aligned head)
            missing_arch = g["missing_arch"].values

            # missing because target_abs_delta_ioi is below threshold
            below_tau = g["tgt_abs_delta_ioi"].values < tau

            missing = missing_arch | below_tau

            n_missing = int(missing.sum())
            frac_missing = n_missing / n_heads

            mean_abs = g["tgt_abs_delta_ioi"].mean()
            std_abs = g["tgt_abs_delta_ioi"].std(ddof=0)

            mean_frac_rank = g["tgt_frac_rank"].mean()
            std_frac_rank = g["tgt_frac_rank"].std(ddof=0)

            rows.append(
                {
                    "base_family": base_fam,
                    "base_model": base_model,
                    "tgt_family": tgt_fam,
                    "tgt_model": tgt_model,
                    "tau": tau,
                    "n_heads": n_heads,
                    "n_missing": n_missing,
                    "frac_missing": frac_missing,
                    "mean_tgt_abs_delta_ioi": mean_abs,
                    "std_tgt_abs_delta_ioi": std_abs,
                    "mean_tgt_frac_rank": mean_frac_rank,
                    "std_tgt_frac_rank": std_frac_rank,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.sort_values(
        ["base_family", "base_model", "tgt_family", "tgt_model", "tau"],
        inplace=True,
    )

    out_df.to_csv(OUT_PATH, index=False)
    print(f"[OUT] Wrote IOI transfer threshold sensitivity to {OUT_PATH}")

    # Also print the key Pythia-410M -> GPT-2-Medium block as a sanity check
    mask = (
        (out_df["base_family"] == "pythia")
        & (out_df["base_model"] == "pythia-410m")
        & (out_df["tgt_family"] == "gpt2")
        & (out_df["tgt_model"] == "gpt2-medium")
    )
    subset = out_df[mask]
    if len(subset) > 0:
        print("\n[HIGHLIGHT] Pythia-410M -> GPT-2-Medium threshold sweep:")
        print(subset.to_string(index=False))
    else:
        print(
            "\n[WARN] No Pythia-410M -> GPT-2-Medium rows found. "
            "Check base/tgt family/model names in the input CSV."
        )


if __name__ == "__main__":
    main()

