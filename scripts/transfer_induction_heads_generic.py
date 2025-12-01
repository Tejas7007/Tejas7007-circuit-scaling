#!/usr/bin/env python
import pandas as pd
import numpy as np

K = 20  # top-K induction heads to track


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure required columns exist
    assert "layer" in df.columns and "head" in df.columns
    assert "delta_induction" in df.columns
    if "abs_delta_induction" not in df.columns:
        df["abs_delta_induction"] = df["delta_induction"].abs()
    return df


def compute_transfer(base_df: pd.DataFrame, tgt_df: pd.DataFrame) -> dict:
    # Rank all heads in target by abs_delta_induction (descending)
    tgt_df = tgt_df.copy()
    tgt_df["tgt_rank"] = tgt_df["abs_delta_induction"].rank(
        ascending=False, method="average"
    )
    n_heads_tgt = len(tgt_df)
    tgt_df["tgt_frac_rank"] = (tgt_df["tgt_rank"] - 1) / (n_heads_tgt - 1)

    # Select top-K heads in base
    base_top = base_df.sort_values("abs_delta_induction", ascending=False).head(K)

    aligned_rows = []
    for _, row in base_top.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])

        # aligned head in target
        match = tgt_df[(tgt_df["layer"] == layer) & (tgt_df["head"] == head)]
        if match.empty:
            continue
        m = match.iloc[0]

        aligned_rows.append(
            dict(
                base_layer=layer,
                base_head=head,
                base_delta_induction=row["delta_induction"],
                base_abs_delta_induction=row["abs_delta_induction"],
                tgt_delta_induction=m["delta_induction"],
                tgt_abs_delta_induction=m["abs_delta_induction"],
                tgt_frac_rank=m["tgt_frac_rank"],
            )
        )

    if not aligned_rows:
        raise ValueError("No aligned heads found between base and target!")

    df_aligned = pd.DataFrame(aligned_rows)
    # "Missing" = not induction-like in target under some simple threshold
    # We reuse the same threshold tau = 0.015 by default
    tau = 0.015
    missing_mask = df_aligned["tgt_abs_delta_induction"] < tau
    n_missing = missing_mask.sum()
    n_total = len(df_aligned)

    summary = dict(
        n_heads=n_total,
        n_missing=n_missing,
        frac_missing=float(n_missing) / float(n_total),
        mean_tgt_abs_delta_induction=float(df_aligned["tgt_abs_delta_induction"].mean()),
        std_tgt_abs_delta_induction=float(df_aligned["tgt_abs_delta_induction"].std()),
        mean_tgt_frac_rank=float(df_aligned["tgt_frac_rank"].mean()),
        std_tgt_frac_rank=float(df_aligned["tgt_frac_rank"].std()),
    )

    return summary, df_aligned


def main():
    base_path = "results/induction_head_scores_pythia-160m.csv"
    tgt_path = "results/induction_head_scores_gpt2-medium.csv"

    print(f"[INFO] Loading base induction scores from {base_path}")
    base_df = load_scores(base_path)
    print(f"[INFO] Loading target induction scores from {tgt_path}")
    tgt_df = load_scores(tgt_path)

    summary, df_aligned = compute_transfer(base_df, tgt_df)

    out_summary = "results/induction_transfer_pythia160m_to_gpt2medium_summary.csv"
    out_aligned = "results/induction_transfer_pythia160m_to_gpt2medium_heads.csv"

    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    df_aligned.to_csv(out_aligned, index=False)

    print(f"[OUT] Wrote induction transfer summary to {out_summary}")
    print(f"[OUT] Wrote aligned head details to {out_aligned}")
    print("\n[HIGHLIGHT] Pythia-160M -> GPT-2-Medium induction transfer:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

