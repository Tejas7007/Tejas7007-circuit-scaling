import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    df = pd.read_csv("results/joint_ioi_anti_repeat_all.csv")
    os.makedirs("results/family_grid", exist_ok=True)

    families = df["family"].unique()
    plt.figure(figsize=(14,10))

    for i, fam in enumerate(families, 1):
        sub = df[df["family"] == fam]
        plt.subplot(2, 2, i)
        sns.scatterplot(
            data=sub,
            x="delta_ioi",
            y="delta_anti",
            s=20,
            alpha=0.7
        )
        plt.axhline(0, color="gray", linestyle="--")
        plt.axvline(0, color="gray", linestyle="--")
        plt.title(f"{fam} heads")
        plt.xlabel("Δ_ioi")
        plt.ylabel("Δ_anti")

    out = "results/family_grid/family_grid.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print("Wrote", out)

if __name__ == "__main__":
    main()

