import os
import pandas as pd
import matplotlib.pyplot as plt

ABL_CSV = "results/ablation/hero_vs_random_ablation.csv"
OUT_SUMMARY_CSV = "results/ablation/hero_vs_random_ablation_summary.csv"
OUT_FIG_DIR = "results/ablation/figs"

os.makedirs("results/ablation", exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def main():
    if not os.path.exists(ABL_CSV):
        raise FileNotFoundError(f"{ABL_CSV} not found. Run ablate_hero_heads_vs_random.py first.")

    df = pd.read_csv(ABL_CSV)

    # Sanity: show columns
    print("Columns in ablation CSV:", list(df.columns))

    # Group by family/model/k/set_type and average metrics across trials
    group_cols = ["family", "model", "k", "set_type"]
    metric_cols = ["acc_ioi", "mean_diff_ioi", "acc_anti", "mean_diff_anti"]

    summary = (
        df.groupby(group_cols)[metric_cols]
        .mean()
        .reset_index()
        .sort_values(["family", "model", "k", "set_type"])
    )

    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"\nWrote ablation summary to {OUT_SUMMARY_CSV}\n")

    # Print a compact view like:
    # family model k set_type acc_ioi acc_anti
    compact = summary[["family", "model", "k", "set_type", "acc_ioi", "acc_anti"]]
    print("=== Compact view (IOI / Anti acc) ===")
    print(compact.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    # Now: for each (family, model), plot IOI and anti accuracy vs k for hero vs random
    for (fam, mdl), sub in summary.groupby(["family", "model"]):
        print(f"\nPlotting ablation curves for {fam}/{mdl}...")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        ks = sorted(sub["k"].unique())

        # Helper to safely extract series
        def get_series(set_type, metric):
            mask = (sub["set_type"] == set_type)
            return sub.loc[mask].set_index("k")[metric].reindex(ks)

        # IOI acc plot
        ax = axes[0]
        hero_ioi = get_series("hero", "acc_ioi")
        rand_ioi = get_series("random", "acc_ioi")
        base_ioi = get_series("baseline", "acc_ioi")

        if not base_ioi.isna().all():
            ax.plot(ks, base_ioi, marker="o", label="baseline")
        if not hero_ioi.isna().all():
            ax.plot(ks, hero_ioi, marker="o", label="hero")
        if not rand_ioi.isna().all():
            ax.plot(ks, rand_ioi, marker="o", label="random")

        ax.set_title("IOI accuracy vs # ablated heads")
        ax.set_xlabel("k (ablated heads)")
        ax.set_ylabel("IOI accuracy")
        ax.legend()

        # Anti acc plot
        ax = axes[1]
        hero_anti = get_series("hero", "acc_anti")
        rand_anti = get_series("random", "acc_anti")
        base_anti = get_series("baseline", "acc_anti")

        if not base_anti.isna().all():
            ax.plot(ks, base_anti, marker="o", label="baseline")
        if not hero_anti.isna().all():
            ax.plot(ks, hero_anti, marker="o", label="hero")
        if not rand_anti.isna().all():
            ax.plot(ks, rand_anti, marker="o", label="random")

        ax.set_title("Anti-repeat accuracy vs # ablated heads")
        ax.set_xlabel("k (ablated heads)")
        ax.set_ylabel("Anti-repeat accuracy")
        ax.legend()

        fig.suptitle(f"{fam}/{mdl} — hero vs random head ablations")
        fig.tight_layout()

        out_path = os.path.join(
            OUT_FIG_DIR, f"{fam}_{mdl}_hero_vs_random_ablation.png"
        )
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

