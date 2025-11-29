import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

INPUT_CSV = "results/transfer/ioi_transfer_generic.csv"
OUT_PNG = "results/transfer/ioi_transfer_generic_summary.png"


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = {
        "base_family",
        "base_model",
        "tgt_family",
        "tgt_model",
        "tgt_frac_rank",
        "missing_in_target_index",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {INPUT_CSV}: {missing}")

    group_cols = ["base_family", "base_model", "tgt_family", "tgt_model"]
    rows = []
    for key, df_pair in df.groupby(group_cols):
        base_family, base_model, tgt_family, tgt_model = key

        n_heads = len(df_pair)
        frac_missing = df_pair["missing_in_target_index"].mean()

        valid = df_pair[
            (~df_pair["missing_in_target_index"])
            & (df_pair["tgt_frac_rank"].notna())
        ]
        if not valid.empty:
            mean_frac_rank = float(valid["tgt_frac_rank"].mean())
        else:
            mean_frac_rank = float("nan")

        label = f"{base_family}/{base_model}\n→ {tgt_family}/{tgt_model}"
        rows.append(
            {
                "pair_label": label,
                "n_heads": n_heads,
                "frac_missing": frac_missing,
                "mean_frac_rank": mean_frac_rank,
            }
        )

    df_summary = pd.DataFrame(rows)

    # lower mean_frac_rank = better transfer
    df_summary = df_summary.sort_values("mean_frac_rank", ascending=True)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    x = np.arange(len(df_summary))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bar1 = ax.bar(
        x - width / 2,
        df_summary["mean_frac_rank"],
        width,
        label="Mean target frac rank\n(0 = best, 1 = worst)",
    )
    bar2 = ax.bar(
        x + width / 2,
        df_summary["frac_missing"],
        width,
        label="Fraction missing in target index",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_summary["pair_label"], rotation=30, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("IOI head transfer across scales and families")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    print(f"Wrote transfer summary figure to {OUT_PNG}")


if __name__ == "__main__":
    main()

