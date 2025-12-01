#!/usr/bin/env python
import pandas as pd
import numpy as np
import os

INPUT_CSV = "paper/tables/joint_ioi_anti_repeat_heads.csv"
OUT_CSV = "paper/tables/anti_transfer_generic.csv"

# Pairs we want to analyze (base → target)
PAIRS = [
    ("pythia-70m", "pythia-160m"),
    ("pythia-160m", "pythia-410m"),
    ("pythia-410m", "pythia-1b"),
    ("pythia-160m", "gpt-neo-125m"),
    ("pythia-160m", "opt-125m"),
    ("pythia-410m", "gpt2-medium"),
]

TOP_K = 20   # same as IOI transfer


def fractional_ranks(values):
    """
    Convert absolute values to fractional ranks in [0,1].
    Lower rank means stronger magnitude (top of list).
    """
    absvals = np.abs(values)
    order = absvals.argsort()  # low to high
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks / (len(values) - 1)


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    rows = []

    for base, target in PAIRS:
        df_base = df[df["model"] == base]
        df_target = df[df["model"] == target]

        if df_base.empty or df_target.empty:
            print(f"[WARN] Missing model(s): {base} or {target}")
            continue

        # 1. Pick top-K anti heads in base model
        df_sorted = df_base.sort_values(by="delta_anti")
        # anti-repeat is negative; stronger = more negative magnitude
        df_top = df_sorted.head(TOP_K).copy()

        # 2. Fractional ranks for target model
        target_frac = fractional_ranks(df_target["delta_anti"].values)

        # Build index for lookup by (layer, head)
        key = list(zip(df_target["layer"], df_target["head"]))
        frac_dict = dict(zip(key, target_frac))

        missing = 0
        frac_list = []

        for _, row in df_top.iterrows():
            key = (row["layer"], row["head"])
            if key not in frac_dict:
                missing += 1
            else:
                frac_list.append(frac_dict[key])

        mean_frac_rank = np.mean(frac_list) if frac_list else np.nan

        rows.append(
            {
                "base": base,
                "target": target,
                "missing": missing,
                "top_k": TOP_K,
                "mean_frac_rank": mean_frac_rank,
            }
        )

        print(f"{base} → {target}: missing={missing}/{TOP_K}, mean_frac_rank={mean_frac_rank:.3f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[OUT] Saved anti-repeat transfer results to {OUT_CSV}")


if __name__ == "__main__":
    main()

