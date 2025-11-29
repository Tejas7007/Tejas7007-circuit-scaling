import os
import numpy as np
import pandas as pd


def classify_heads(df_family_model: pd.DataFrame, tau: float):
    """Classify heads into IOI-only, Anti-only, Shared, Weak/none."""

    # Negative delta means "suppressor" in our convention
    d_ioi = df_family_model["delta_ioi"].values
    d_anti = df_family_model["delta_anti"].values

    is_ioi = d_ioi <= -tau
    is_anti = d_anti <= -tau

    ioi_only = np.logical_and(is_ioi, ~is_anti)
    anti_only = np.logical_and(~is_ioi, is_anti)
    shared = np.logical_and(is_ioi, is_anti)
    weak = np.logical_not(np.logical_or(is_ioi, is_anti))

    return {
        "heads": len(df_family_model),
        "ioi_only": int(ioi_only.sum()),
        "anti_only": int(anti_only.sum()),
        "shared": int(shared.sum()),
        "weak": int(weak.sum()),
    }


def main():
    path = os.path.join("results", "joint_ioi_anti_repeat_all.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run joint_ioi_anti_repeat_all.py first.")

    df = pd.read_csv(path)
    if df.empty:
        print("joint_ioi_anti_repeat_all.csv is empty; nothing to analyze.")
        return

    taus = [0.03, 0.05, 0.07, 0.10]

    for tau in taus:
        print(f"\n========== τ = {tau:.2f} ==========")
        for (family, model), group in df.groupby(["family", "model"]):
            stats = classify_heads(group, tau)
            corr = group[["delta_ioi", "delta_anti"]].corr().iloc[0, 1]

            print(f"Family: {family:<8} Model: {model}")
            print(
                f"  heads: {stats['heads']:3d}  "
                f"IOI-only: {stats['ioi_only']:3d}, "
                f"Anti-only: {stats['anti_only']:3d}, "
                f"Shared: {stats['shared']:3d}, "
                f"Weak/none: {stats['weak']:3d}"
            )
            print(f"  Pearson corr(Δ_ioi, Δ_anti): {corr:6.3f}")


if __name__ == "__main__":
    main()
