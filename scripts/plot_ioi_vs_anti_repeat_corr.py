import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

CSV_PATH = "results/joint_ioi_anti_repeat_pythia.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    print("Columns in joint CSV:", df.columns.tolist())

    # Adjust these names if they differ
    model_col = "model"
    delta_ioi_col = "delta_ioi"
    delta_anti_col = "delta_anti"

    models = sorted(df[model_col].unique(), key=lambda m: ["pythia-70m","pythia-160m","pythia-410m","pythia-1b"].index(m))
    corrs = []

    for m in models:
        sub = df[df[model_col] == m]
        x = sub[delta_ioi_col].values
        y = sub[delta_anti_col].values

        if len(sub) < 2:
            corr = float("nan")
        else:
            corr, _ = pearsonr(x, y)
        corrs.append(corr)
        print(f"{m}: Pearson corr(Δ_ioi, Δ_anti) = {corr:.3f}")

    # Plot
    plt.figure()
    plt.bar(models, corrs)
    plt.ylabel("Pearson corr(Δ_ioi, Δ_anti)")
    plt.title("IOI vs Anti-Repeat Copy Suppression Correlation (Pythia)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("results/pythia_ioi_anti_repeat_corr.png", dpi=200)
    print("[+] Wrote results/pythia_ioi_anti_repeat_corr.png")

if __name__ == "__main__":
    main()

