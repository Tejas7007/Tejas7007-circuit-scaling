#!/usr/bin/env python
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "paper" / "tables" / "ioi_transfer_generic.csv"
OUT_PATH = ROOT / "results" / "ioi_what_transfers_summary.csv"

TAU = 0.015  # IOI-like threshold, synced with rest of paper

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Could not find {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    required_cols = {
        "base_family", "base_model", "tgt_family", "tgt_model",
        "layer", "head", "base_abs_delta_ioi", "tgt_abs_delta_ioi",
        "tgt_rank", "tgt_frac_rank"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {IN_PATH}: {missing}")

    # Add a boolean: does this head still look IOI-like in the target?
    df["tgt_is_ioi_like"] = df["tgt_abs_delta_ioi"] >= TAU

    groups = []
    for (bfam, bmod, tfam, tmod), g in df.groupby(
        ["base_family", "base_model", "tgt_family", "tgt_model"]
    ):
        n_heads = len(g)
        n_good = int(g["tgt_is_ioi_like"].sum())
        frac_good = n_good / n_heads if n_heads > 0 else 0.0

        groups.append({
            "base_family": bfam,
            "base_model": bmod,
            "tgt_family": tfam,
            "tgt_model": tmod,
            "n_heads": n_heads,
            "n_good_tgt_ioi_like": n_good,
            "frac_good_tgt_ioi_like": frac_good,
            "mean_tgt_abs_delta_ioi": g["tgt_abs_delta_ioi"].mean(),
            "std_tgt_abs_delta_ioi": g["tgt_abs_delta_ioi"].std(ddof=0),
            "mean_tgt_frac_rank": g["tgt_frac_rank"].mean(),
            "std_tgt_frac_rank": g["tgt_frac_rank"].std(ddof=0),
        })

    out_df = pd.DataFrame(groups)
    out_df = out_df.sort_values(
        ["base_family", "base_model", "tgt_family", "tgt_model"]
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"[OUT] Wrote IOI 'what transfers' summary to {OUT_PATH}")

    # Highlight within-family vs cross-family in stdout
    print("\n[HIGHLIGHT] Within-family (Pythia -> Pythia):")
    in_family = out_df[
        (out_df["base_family"] == "pythia")
        & (out_df["tgt_family"] == "pythia")
    ]
    if not in_family.empty:
        print(in_family.to_string(index=False))
    else:
        print("  [INFO] No Pythia -> Pythia rows found.")

    print("\n[HIGHLIGHT] Cross-family (Pythia -> non-Pythia):")
    cross = out_df[
        (out_df["base_family"] == "pythia")
        & (out_df["tgt_family"] != "pythia")
    ]
    if not cross.empty:
        print(cross.to_string(index=False))
    else:
        print("  [INFO] No Pythia -> non-Pythia rows found.")

if __name__ == "__main__":
    main()

