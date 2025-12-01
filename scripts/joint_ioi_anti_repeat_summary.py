#!/usr/bin/env python
"""
Summarize joint IOI vs anti-repeat behavior for each model.

This script expects one or more CSVs of the form:
    results/joint_ioi_anti_repeat_*.csv

Each CSV should contain at least:
    model, layer, head, delta_ioi, delta_anti

It will:
  * Classify each head into:
        - ioi_only
        - anti_only
        - shared (both non-zero, same sign)
        - opposite_sign (both non-zero, opposite sign)
        - silent (both ~0)
  * Write a per-head table with the new "joint_role" column.
  * Write a per-model summary table with counts and fractions.
"""

import os
from glob import glob

import numpy as np
import pandas as pd
from pathlib import Path


JOINT_HEADS_OUT = Path("results/joint_ioi_anti_repeat_all_heads_with_roles.csv")
SUMMARY_OUT = Path("paper/tables/joint_ioi_anti_repeat_summary.csv")


def classify_head(delta_ioi: float, delta_anti: float, eps: float = 1e-6) -> str:
    """Assign a simple joint role label based on IOI / anti deltas."""
    ioi_on = abs(delta_ioi) > eps
    anti_on = abs(delta_anti) > eps

    if not ioi_on and not anti_on:
        return "silent"
    if ioi_on and not anti_on:
        return "ioi_only"
    if not ioi_on and anti_on:
        return "anti_only"

    # Both non-zero
    if delta_ioi * delta_anti > 0:
        return "shared"
    else:
        return "opposite_sign"


def load_all_joint_tables() -> pd.DataFrame:
    """Load all joint IOI/anti CSVs and concatenate them."""
    paths = sorted(glob("results/joint_ioi_anti_repeat_*.csv"))
    if not paths:
        raise FileNotFoundError(
            "No joint IOI/anti tables found under "
            "'results/joint_ioi_anti_repeat_*.csv'. "
            "Run scripts/joint_ioi_anti_repeat_gpt2.py or similar first."
        )

    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        print(f"[INFO] Loaded {path} with shape {df.shape}")
        # Basic sanity check
        required_cols = {"model", "layer", "head", "delta_ioi", "delta_anti"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}. "
                f"Columns present: {list(df.columns)}"
            )
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("paper/tables", exist_ok=True)

    df = load_all_joint_tables()
    print("[INFO] Combined joint IOI/anti table shape:", df.shape)
    print("[INFO] Columns:", list(df.columns))

    # Classify each head
    df["joint_role"] = [
        classify_head(di, da) for di, da in zip(df["delta_ioi"], df["delta_anti"])
    ]

    # Save per-head table with roles
    JOINT_HEADS_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(JOINT_HEADS_OUT, index=False)
    print(f"[OUT] Wrote per-head joint roles to {JOINT_HEADS_OUT}")

    # Per-model summary
    grouped = (
        df.groupby(["model", "joint_role"])
        .size()
        .reset_index(name="count")
        .sort_values(["model", "count"], ascending=[True, False])
    )

    # Add total + fraction per model
    totals = grouped.groupby("model")["count"].transform("sum")
    grouped["frac"] = grouped["count"] / totals

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(SUMMARY_OUT, index=False)
    print(f"[OUT] Wrote joint IOI/anti summary to {SUMMARY_OUT}")
    print("\n=== Joint IOI vs Anti-repeat summary ===")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()

