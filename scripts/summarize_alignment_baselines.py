#!/usr/bin/env python
"""
Summarize different alignment baselines:

1. Coordinate IOI transfer (per-head table)
2. Relative-depth alignment
3. CKA alignment vs random baseline

Outputs:
    paper/tables/alignment_baselines_summary.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    os.makedirs("paper/tables", exist_ok=True)

    # ---------- 1) Coordinate IOI transfer (per-head table) ----------
    ioi_path = "results/transfer/ioi_transfer_generic.csv"
    ioi = pd.read_csv(ioi_path)
    print("[INFO] Loaded IOI generic transfer:", ioi_path)
    print("[INFO] IOI columns:", list(ioi.columns))

    # Expect columns:
    #   base_family, base_model, tgt_family, tgt_model,
    #   layer, head, base_abs_delta_ioi, tgt_abs_delta_ioi,
    #   tgt_rank, tgt_frac_rank, missing_in_target_index

    mask = (
        (ioi["base_family"] == "pythia")
        & (ioi["base_model"] == "pythia-410m")
        & (ioi["tgt_family"] == "gpt2")
        & (ioi["tgt_model"] == "gpt2-medium")
    )

    if not mask.any():
        raise ValueError(
            "Could not find rows for pythia-410m → gpt2-medium in "
            "results/transfer/ioi_transfer_generic.csv"
        )

    coord_rows = ioi.loc[mask].copy()
    print(f"[INFO] Coordinate IOI subset rows: {len(coord_rows)}")

    # You can decide whether to exclude missing heads; here we just include all
    coord_mean_frac_rank = coord_rows["tgt_frac_rank"].mean()
    coord_mean_abs_delta = coord_rows["tgt_abs_delta_ioi"].mean()

    # ---------- 2) Relative depth alignment ----------
    depth_path = "results/relative_depth_alignment_ioi.csv"
    depth = pd.read_csv(depth_path)
    print("[INFO] Loaded relative depth:", depth_path)
    print("[INFO] Depth columns:", list(depth.columns))

    # Depth table already has rel_depth_diff from src/tgt layers
    mean_rel_diff = depth["rel_depth_diff"].mean()
    std_rel_diff = depth["rel_depth_diff"].std()
    print(f"[INFO] Relative depth: mean={mean_rel_diff:.4f}, std={std_rel_diff:.4f}")

    # ---------- 3) CKA alignment vs random baseline ----------
    cka_path = "results/cka_alignment_pythia410m_gpt2medium.csv"
    cka = pd.read_csv(cka_path)
    print("[INFO] Loaded CKA alignment:", cka_path)
    print("[INFO] CKA columns:", list(cka.columns))

    cka_mean = cka["cka"].mean()
    cka_std = cka["cka"].std()
    print(f"[INFO] CKA real: mean={cka_mean:.4f}, std={cka_std:.4f}")

    rand_npz_path = "results/cka_random_baseline_pythia-410m_gpt2-medium.npz"
    rand_npz = np.load(rand_npz_path)
    rand_vals = rand_npz["cka_vals"]
    rand_mean = rand_vals.mean()
    rand_std = rand_vals.std()
    print(f"[INFO] CKA random: mean={rand_mean:.6f}, std={rand_std:.6f}")

    # ---------- Assemble summary table ----------
    summary = pd.DataFrame(
        [
            {
                "alignment_baseline": "coordinate_ioi_transfer",
                "mean_frac_rank": coord_mean_frac_rank,
                "mean_abs_delta_ioi": coord_mean_abs_delta,
                "extra_metric": np.nan,  # not used for this row
            },
            {
                "alignment_baseline": "relative_depth",
                "mean_frac_rank": np.nan,
                "mean_abs_delta_ioi": np.nan,
                "extra_metric": mean_rel_diff,  # mean |Δ rel depth|
            },
            {
                "alignment_baseline": "cka_real_heads",
                "mean_frac_rank": np.nan,
                "mean_abs_delta_ioi": np.nan,
                "extra_metric": cka_mean,  # mean CKA
            },
            {
                "alignment_baseline": "cka_random_baseline",
                "mean_frac_rank": np.nan,
                "mean_abs_delta_ioi": np.nan,
                "extra_metric": rand_mean,  # baseline CKA
            },
        ]
    )

    out_path = Path("paper/tables/alignment_baselines_summary.csv")
    summary.to_csv(out_path, index=False)
    print("[OUT] Wrote alignment baselines summary to", out_path)
    print(summary)


if __name__ == "__main__":
    main()

