#!/usr/bin/env python
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "results" / "global_ioi_circuit_gpt2-medium.csv"
OUT_HEADS = ROOT / "results" / "global_ioi_gpt2medium_top_heads.csv"
OUT_MLPS = ROOT / "results" / "global_ioi_gpt2medium_top_mlps.csv"

TOP_K = 10

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Could not find {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    print(f"[INFO] Loaded {IN_PATH}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Use |delta_margin| as abs_effect
    if "delta_margin" not in df.columns:
        raise ValueError("No delta_margin column found in global scan CSV")

    df["abs_effect"] = df["delta_margin"].abs()

    # unit type column is "type"
    if "type" not in df.columns:
        raise ValueError("Expected a column named 'type' indicating attention/mlp")
    unit_col = "type"

    # layer column (exists already)
    if "layer" not in df.columns:
        raise ValueError("Expected a 'layer' column")
    layer_col = "layer"

    # Detect attention heads vs mlps
    heads_mask = df[unit_col].str.contains("att", case=True) | df[unit_col].str.contains("head", case=True)
    mlp_mask   = df[unit_col].str.contains("mlp", case=True)

    df_heads = df[heads_mask].copy()
    df_mlps  = df[mlp_mask].copy()

    if df_heads.empty:
        print("[WARN] No attention heads found in global scan (type contains 'att' or 'head').")
    if df_mlps.empty:
        print("[WARN] No mlp units found in global scan (type contains 'mlp').")

    # Sort by magnitude of effect
    df_heads_top = df_heads.sort_values("abs_effect", ascending=False).head(TOP_K)
    df_mlps_top  = df_mlps.sort_values("abs_effect", ascending=False).head(TOP_K)

    OUT_HEADS.parent.mkdir(parents=True, exist_ok=True)
    df_heads_top.to_csv(OUT_HEADS, index=False)
    df_mlps_top.to_csv(OUT_MLPS, index=False)

    print(f"[OUT] Wrote top-{TOP_K} attention heads to {OUT_HEADS}")
    print(f"[OUT] Wrote top-{TOP_K} mlp blocks to {OUT_MLPS}")

    print("\n=== TOP ATTENTION HEADS ===")
    print(df_heads_top.to_string(index=False))

    print("\n=== TOP MLP BLOCKS ===")
    print(df_mlps_top.to_string(index=False))

if __name__ == "__main__":
    main()

