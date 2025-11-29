import os
import pandas as pd
import matplotlib.pyplot as plt

DIVERGENCE_CSV = "results/transfer/ioi_divergence_pythia410m_to_gpt2medium.csv"
OUT_PNG = "results/transfer/ioi_divergence_pythia410m_to_gpt2medium_bar.png"

def main():
    if not os.path.exists(DIVERGENCE_CSV):
        raise FileNotFoundError(f"Cannot find divergence CSV at {DIVERGENCE_CSV}")

    df = pd.read_csv(DIVERGENCE_CSV)

    if "divergence_type" not in df.columns:
        raise ValueError(f"'divergence_type' column not found in {DIVERGENCE_CSV}. "
                         f"Columns present: {list(df.columns)}")

    total = len(df)
    counts = df["divergence_type"].value_counts().sort_index()
    frac = counts / total

    print(f"Total IOI-like heads in base model: {total}")
    print("\nDivergence type counts:")
    for t, c in counts.items():
        print(f"  {t:20s} n={c:3d} ({100.0 * c / total:.2f}%)")

    # --- Plot ---
    plt.figure(figsize=(5, 4))
    x = range(len(counts))
    labels = list(counts.index)
    plt.bar(x, frac.values)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Fraction of IOI-like heads")
    plt.title("IOI Divergence Types: Pythia-410M → GPT-2-Medium")
    plt.ylim(0, 1.0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200)
    print(f"\nWrote divergence bar plot to {OUT_PNG}")

if __name__ == "__main__":
    main()

