import os
import pandas as pd

def main():
    in_csv = os.path.join("results", "joint_ioi_anti_repeat_heads.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Missing {in_csv}, run list_strong_heads.py first.")

    df = pd.read_csv(in_csv)

    # Sanity: ensure categories are present
    assert "family" in df.columns
    assert "model" in df.columns
    assert "category" in df.columns

    # Count heads per (family, model, category)
    counts = (
        df
        .groupby(["family", "model", "category"])
        .size()
        .reset_index(name="n_heads")
    )

    # Total heads per (family, model)
    totals = (
        df
        .groupby(["family", "model"])
        .size()
        .reset_index(name="total_heads")
    )

    summary = counts.merge(totals, on=["family", "model"], how="left")
    summary["fraction"] = summary["n_heads"] / summary["total_heads"]

    # Pivot to wide format: one row per (family, model)
    pivot_counts = summary.pivot_table(
        index=["family", "model"],
        columns="category",
        values="n_heads",
        fill_value=0
    )

    pivot_frac = summary.pivot_table(
        index=["family", "model"],
        columns="category",
        values="fraction",
        fill_value=0.0
    )

    # Flatten column names
    pivot_counts.columns = [f"n_{c}" for c in pivot_counts.columns]
    pivot_frac.columns = [f"frac_{c}" for c in pivot_frac.columns]

    final = pivot_counts.join(pivot_frac, how="left").reset_index()

    out_dir = os.path.join("results", "summaries")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "head_category_summary.csv")
    final.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    print("\n=== Compact view ===")
    print(final.to_string(index=False))

if __name__ == "__main__":
    main()

