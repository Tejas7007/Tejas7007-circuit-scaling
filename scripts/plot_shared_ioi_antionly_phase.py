import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    df = pd.read_csv("results/joint_ioi_anti_repeat_heads.csv")
    os.makedirs("results/phase_space", exist_ok=True)

    plt.figure(figsize=(7,7))
    sns.scatterplot(
        data=df,
        x="delta_ioi",
        y="delta_anti",
        hue="category",
        style="family",
        s=40,
        alpha=0.8
    )
    plt.axhline(0, color="black", linestyle="--")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Δ_ioi")
    plt.ylabel("Δ_anti")
    plt.title("Phase Space of Head Types Across All Model Families")

    out = "results/phase_space/all_heads_phase_space.png"
    plt.savefig(out, dpi=300)
    print("Wrote", out)

if __name__ == "__main__":
    main()

