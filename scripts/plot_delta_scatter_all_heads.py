import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    df = pd.read_csv("results/joint_ioi_anti_repeat_all.csv")
    os.makedirs("results/scatter_heads", exist_ok=True)

    for (fam, model), sub in df.groupby(["family", "model"]):
        plt.figure(figsize=(6,6))
        sns.scatterplot(
            data=sub,
            x="delta_ioi",
            y="delta_anti",
            s=30,
            alpha=0.7
        )
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.title(f"{fam}/{model}: Δ_ioi vs Δ_anti")
        plt.xlabel("Δ_ioi")
        plt.ylabel("Δ_anti")

        out = f"results/scatter_heads/{fam}_{model}_delta_scatter.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print("Wrote", out)

if __name__ == "__main__":
    main()

