import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    df = pd.read_csv("results/joint_ioi_anti_repeat_heads.csv")
    os.makedirs("results/head_scaling", exist_ok=True)

    grouped = df.groupby(["model", "family"])["strength"].count().reset_index()
    # Get model sizes from the summary we created
    scale = pd.read_csv("results/corr_vs_scale_cross_family.csv")[["model", "params_m"]]

    merged = grouped.merge(scale, on="model")

    plt.figure(figsize=(8,6))
    for fam, sub in merged.groupby("family"):
        plt.plot(sub["params_m"], sub["strength"], marker="o", label=fam)

    plt.xlabel("Model size (M params)")
    plt.ylabel("Number of strong heads")
    plt.title("Scaling of strong heads with model size")
    plt.legend()

    out = "results/head_scaling/strong_head_scaling.png"
    plt.savefig(out, dpi=300)
    print("Wrote", out)

if __name__ == "__main__":
    main()

