#!/usr/bin/env python
import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd

JOINT_PATH = "paper/tables/joint_ioi_anti_repeat_all.csv"
OUT_DIR = "results/divergence"


def compute_relative_layers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_layers"] = df.groupby(["family", "model"])["layer"].transform(
        lambda s: int(s.max()) + 1
    )
    df["relative_layer"] = (df["layer"] + 0.5) / df["n_layers"]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True,
                        help="e.g. 'pythia-410m'")
    parser.add_argument("--target_model", type=str, required=True,
                        help="e.g. 'gpt2-medium'")
    parser.add_argument("--ioi_thresh", type=float, default=0.15,
                        help="Abs Δ_ioi threshold to treat a head as IOI-like")
    parser.add_argument("--strong_thresh", type=float, default=0.25,
                        help="Abs Δ_ioi threshold for 'strong' IOI heads")
    parser.add_argument("--moderate_thresh", type=float, default=0.10,
                        help="Abs Δ_ioi threshold for 'moderate' IOI heads")
    parser.add_argument("--rel_layer_tol", type=float, default=0.15,
                        help="Max |Δ relative_layer| to call a head 'relocated'")
    parser.add_argument("--top_k_base", type=int, default=40,
                        help="Number of top base IOI heads to consider")
    args = parser.parse_args()

    df = pd.read_csv(JOINT_PATH)
    required = {"family", "model", "layer", "head", "delta_ioi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {JOINT_PATH}: {missing}")

    df = compute_relative_layers(df)

    base = df[df["model"] == args.base_model].copy()
    target = df[df["model"] == args.target_model].copy()

    if base.empty:
        raise ValueError(f"No rows for base_model={args.base_model}")
    if target.empty:
        raise ValueError(f"No rows for target_model={args.target_model}")

    # Precompute IOI magnitudes and signs
    base["abs_delta_ioi"] = base["delta_ioi"].abs()
    base["sign_delta_ioi"] = np.sign(base["delta_ioi"]).astype(int)
    target["abs_delta_ioi"] = target["delta_ioi"].abs()
    target["sign_delta_ioi"] = np.sign(target["delta_ioi"]).astype(int)

    # Sort base IOI heads by IOI strength, take top_k_base
    base_ioi = base[base["abs_delta_ioi"] >= args.ioi_thresh].copy()
    base_ioi = base_ioi.sort_values("abs_delta_ioi", ascending=False).head(args.top_k_base)

    if base_ioi.empty:
        raise ValueError(
            f"No base IOI-like heads with |Δ_ioi| >= {args.ioi_thresh} for model={args.base_model}"
        )

    # Build quick lookup for target
    target_by_pos = target.set_index(["layer", "head"])

    # Global target stats
    target_strong = target[target["abs_delta_ioi"] >= args.strong_thresh].copy()
    target_moderate = target[
        (target["abs_delta_ioi"] >= args.moderate_thresh)
        & (target["abs_delta_ioi"] < args.strong_thresh)
    ].copy()

    # If target has basically no IOI-ish heads at all, most things will be missing_in_target
    target_has_any_moderate = not target_moderate.empty
    target_has_any_strong = not target_strong.empty

    records = []

    for _, row in base_ioi.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])
        base_abs = float(row["abs_delta_ioi"])
        base_sign = int(row["sign_delta_ioi"])
        base_rel = float(row["relative_layer"])

        # Default classification
        category = "unknown"
        detail = ""

        # Try aligned head
        aligned = None
        if (layer, head) in target_by_pos.index:
            aligned = target_by_pos.loc[(layer, head)]
            aligned_abs = float(aligned["abs_delta_ioi"])
            aligned_sign = int(aligned["sign_delta_ioi"])
        else:
            aligned_abs = 0.0
            aligned_sign = 0

        # Helper: find candidate relocated heads (strong, same sign, near in depth)
        relocated_candidates = target_strong[
            (target_strong["sign_delta_ioi"] == base_sign)
            & (target_strong["relative_layer"].sub(base_rel).abs() <= args.rel_layer_tol)
        ]

        # Helper: count moderate same-sign heads anywhere
        moderate_same_sign = target_moderate[target_moderate["sign_delta_ioi"] == base_sign]

        # 1) If target has basically no IOI-ish heads
        if not target_has_any_moderate and not target_has_any_strong:
            category = "missing_in_target"
            detail = "target_has_no_ioi_heads"

        # 2) If aligned head is strong
        elif aligned_abs >= args.strong_thresh:
            if aligned_sign == base_sign:
                category = "conserved"
                detail = "aligned_strong_same_sign"
            else:
                category = "inverted"
                detail = "aligned_strong_opposite_sign"

        # 3) If aligned head is weak, but a nearby strong same-sign head exists
        elif aligned_abs < args.moderate_thresh and not relocated_candidates.empty:
            category = "relocated"
            detail = f"nearest_strong_at_L{int(relocated_candidates.iloc[0]['layer'])}H{int(relocated_candidates.iloc[0]['head'])}"

        # 4) If IOI is spread across multiple moderate heads
        elif moderate_same_sign.shape[0] >= 3:
            category = "diffused"
            detail = f"{moderate_same_sign.shape[0]}_moderate_heads_same_sign"

        # 5) Otherwise: target has some IOI-ish structure but this base head's role is lost
        else:
            category = "lost"
            detail = "no_aligned_no_relocated_no_diffused"

        records.append(
            {
                "base_model": args.base_model,
                "target_model": args.target_model,
                "layer": layer,
                "head": head,
                "base_delta_ioi": row["delta_ioi"],
                "base_abs_delta_ioi": base_abs,
                "base_sign": base_sign,
                "aligned_delta_ioi": aligned["delta_ioi"] if aligned is not None else 0.0,
                "aligned_abs_delta_ioi": aligned_abs,
                "aligned_sign": aligned_sign,
                "base_relative_layer": base_rel,
                "category": category,
                "detail": detail,
            }
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_name = f"divergence_taxonomy_{args.base_model.replace('/', '-')}_to_{args.target_model.replace('/', '-')}.csv"
    out_path = os.path.join(OUT_DIR, out_name)
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)

    print(f"[OUT] Saved divergence taxonomy to {out_path}")
    print("\nCategory counts:")
    print(out_df["category"].value_counts())


if __name__ == "__main__":
    main()

