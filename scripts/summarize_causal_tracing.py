import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heads_csv",
        type=str,
        default="results/hero_heads_for_paper.csv",
        help="CSV of hero heads (family, model, category, layer, head, strength, delta_*).",
    )
    parser.add_argument(
        "--causal_json",
        type=str,
        default="results/causal_tracing/hero_head_causal_results.json",
        help="JSON file of causal tracing results.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/causal_tracing/hero_head_causal_summary.csv",
        help="Output CSV with merged results.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/causal_tracing",
        help="Directory to write plots into.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load hero heads
    heads_df = pd.read_csv(args.heads_csv)

    # Load causal tracing JSON
    with open(args.causal_json, "r") as f:
        causal_list = json.load(f)
    causal_df = pd.DataFrame(causal_list)

    # Merge on (family, model, category, layer, head)
    merged = heads_df.merge(
        causal_df,
        on=["family", "model", "category", "layer", "head"],
        how="inner",
        suffixes=("", "_causal"),
    )

    # Save merged CSV
    merged.to_csv(args.out_csv, index=False)
    print(f"Wrote merged summary to {args.out_csv}")
    print()
    print("Merged columns:", list(merged.columns))

    # Compute simple correlations
    def safe_corr(a, b):
        if merged[a].nunique() <= 1 or merged[b].nunique() <= 1:
            return float("nan")
        return merged[a].corr(merged[b])

    corr_ioi = safe_corr("strength", "mean_ioi_influence")
    corr_anti = safe_corr("strength", "mean_anti_influence")
    print(f"Corr(strength, mean_ioi_influence)  = {corr_ioi:.4f}")
    print(f"Corr(strength, mean_anti_influence) = {corr_anti:.4f}")

    # Also save these correlations to a tiny txt file for reference
    corr_txt_path = os.path.join(args.out_dir, "hero_head_causal_correlations.txt")
    with open(corr_txt_path, "w") as f:
        f.write(f"Corr(strength, mean_ioi_influence)  = {corr_ioi:.6f}\n")
        f.write(f"Corr(strength, mean_anti_influence) = {corr_anti:.6f}\n")
    print(f"Wrote correlations to {corr_txt_path}")

    # Scatter plots: strength vs causal influence (IOI + anti) colored by category
    for task, y_col, fname in [
        ("ioi", "mean_ioi_influence", "hero_head_causal_scatter_ioi.png"),
        ("anti", "mean_anti_influence", "hero_head_causal_scatter_anti.png"),
    ]:
        plt.figure(figsize=(5, 4))
        for cat in merged["category"].unique():
            sub = merged[merged["category"] == cat]
            plt.scatter(sub["strength"], sub[y_col], label=cat, alpha=0.8)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Head strength (|Δ_ioi| + |Δ_anti|)")
        plt.ylabel(f"Mean {task} causal influence")
        plt.title(f"Hero heads: strength vs {task} influence")
        plt.legend()
        out_path = os.path.join(args.out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

