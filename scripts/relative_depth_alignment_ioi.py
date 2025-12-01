#!/usr/bin/env python
"""
Relative-depth IOI alignment for Pythia-410M → GPT-2-medium.

Reads:  paper/tables/ioi_transfer_pythia410m_to_gpt2medium.csv
Writes: results/relative_depth_alignment_ioi.csv

Table schema (in your repo):

  layer, head,
  base_family, base_model, base_delta_ioi, base_abs_delta_ioi, base_category,
  target_family, target_model, target_delta_ioi, target_abs_delta_ioi, target_category

There is only one layer column ("layer"), which is the Pythia layer index.
Since Pythia-410M and GPT-2-medium both have 24 layers, relative depth and
coordinate depth are effectively the same here. We therefore treat "layer"
as both src_layer and tgt_layer for the purpose of this baseline.
"""

import os
from typing import List

import numpy as np
import pandas as pd

IOI_TRANSFER_PATH = "paper/tables/ioi_transfer_pythia410m_to_gpt2medium.csv"
OUT_PATH = "results/relative_depth_alignment_ioi.csv"


def pick_column(df: pd.DataFrame, logical_name: str, candidates: List[str]) -> str:
    """
    Try several possible column names and return the first that exists.
    Raise a clear error if none are found.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a column for {logical_name}. "
        f"Tried candidates: {candidates}. Available columns: {list(df.columns)}"
    )


def load_ioi_transfer() -> pd.DataFrame:
    """
    Load IOI transfer results for Pythia-410M -> GPT-2-medium.
    """
    if not os.path.exists(IOI_TRANSFER_PATH):
        raise FileNotFoundError(
            f"Expected IOI transfer CSV at {IOI_TRANSFER_PATH}, but it does not exist."
        )

    df = pd.read_csv(IOI_TRANSFER_PATH)
    print(f"[INFO] Loaded IOI transfer table with shape {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Filter by family/model if present (but your file is already specific)
    if "base_family" in df.columns and "target_family" in df.columns:
        mask = (df["base_family"] == "pythia") & (df["target_family"] == "gpt2")
        before = df.shape[0]
        df = df[mask]
        print(
            f"[INFO] Filtered by base_family=pythia, target_family=gpt2: "
            f"{before} → {df.shape[0]} rows"
        )

    if df.empty:
        raise ValueError(
            "After filtering for Pythia-410M → GPT-2-medium IOI, the table is empty."
        )

    return df


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df = load_ioi_transfer()

    # Source layer: your table uses "layer" for the Pythia layer index
    src_layer_col = pick_column(
        df,
        logical_name="src_layer",
        candidates=["src_layer", "base_layer", "layer_src", "layer_A", "layer"],
    )

    # Target layer: there is no explicit target layer column in your table,
    # so we treat the same layer index as the target relative depth as well.
    try:
        tgt_layer_col = pick_column(
            df,
            logical_name="tgt_layer",
            candidates=[
                "tgt_layer",
                "target_layer",
                "layer_tgt",
                "layer_B",
                "layer",
            ],
        )
    except KeyError:
        print(
            "[INFO] No explicit target layer column; using source layer column "
            "as a proxy for target layer."
        )
        tgt_layer_col = src_layer_col

    print(f"[INFO] Using src_layer column: {src_layer_col}")
    print(f"[INFO] Using tgt_layer column: {tgt_layer_col}")

    src_layers = df[src_layer_col].astype(float)
    tgt_layers = df[tgt_layer_col].astype(float)

    src_max = float(src_layers.max())
    tgt_max = float(tgt_layers.max())

    if src_max <= 0 or tgt_max <= 0:
        raise ValueError(
            f"Invalid layer maxima: src_max={src_max}, tgt_max={tgt_max}. "
            "Expected positive maxima for layer indices."
        )

    # Relative depth in [0, 1]
    df["src_rel_depth"] = src_layers / src_max
    df["tgt_rel_depth"] = tgt_layers / tgt_max
    df["rel_depth_diff"] = (df["src_rel_depth"] - df["tgt_rel_depth"]).abs()
    df["coord_match"] = src_layers == tgt_layers

    # Try to grab a quality signal if present
    quality_col = None
    for cand in ["frac_rank", "frac_rank_ioi", "frac_good", "is_good"]:
        if cand in df.columns:
            quality_col = cand
            break

    if quality_col is not None:
        print(f"[INFO] Using quality column: {quality_col}")
        quality = df[quality_col].astype(float)
        corr_coord = np.corrcoef(df["coord_match"].astype(float), quality)[0, 1]
        corr_rel = np.corrcoef(-df["rel_depth_diff"], quality)[0, 1]
        print(
            f"[METRIC] Corr(coord_match, {quality_col}) = {corr_coord:.3f} "
            "(higher = better coordinate alignment)"
        )
        print(
            f"[METRIC] Corr(-rel_depth_diff, {quality_col}) = {corr_rel:.3f} "
            "(higher = better relative-depth alignment)"
        )
    else:
        print(
            "[WARN] No obvious quality column found "
            "(tried frac_rank, frac_rank_ioi, frac_good, is_good). "
            "Only writing structural relative-depth info."
        )

    # Save enriched table
    df.to_csv(OUT_PATH, index=False)
    print(f"[OUT] Saved relative depth alignment table to {OUT_PATH}")

    coord_match = df["coord_match"]
    rel_depth_diff = df["rel_depth_diff"]
    print("\n[SUMMARY]")
    print(f"  Rows: {df.shape[0]}")
    print(f"  Coord-match fraction: {coord_match.mean():.3f}")
    print(
        f"  Mean |src_rel_depth - tgt_rel_depth|: "
        f"{rel_depth_diff.mean():.3f}  (median={rel_depth_diff.median():.3f})"
    )


if __name__ == "__main__":
    main()

