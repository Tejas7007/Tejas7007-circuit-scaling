from __future__ import annotations

from pathlib import Path
import pandas as pd


# We use the per-head summary table that includes `category`
HEADS_TABLE = Path("paper/tables/joint_ioi_anti_repeat_heads.csv")

OUT_DIR = Path("results/transfer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Base (source) and target models for transfer analysis
BASE_FAMILY = "pythia"
BASE_MODEL = "pythia-410m"

TARGET_FAMILY = "gpt2"
TARGET_MODEL = "gpt2-medium"

# How many top IOI heads to treat as "hero" heads in the base model
TOP_K = 20


def load_heads(path: Path) -> pd.DataFrame:
    """Load the per-head table and sanity check required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find heads table at {path}")

    df = pd.read_csv(path)

    required = {
        "family",
        "model",
        "layer",
        "head",
        "delta_ioi",
        "delta_anti",
        "abs_delta_ioi",
        "abs_delta_anti",
        "strength",
        "category",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    return df


def main() -> None:
    df = load_heads(HEADS_TABLE)

    # Slice to base and target models
    base = df[(df["family"] == BASE_FAMILY) & (df["model"] == BASE_MODEL)].copy()
    target = df[
        (df["family"] == TARGET_FAMILY) & (df["model"] == TARGET_MODEL)
    ].copy()

    if base.empty:
        raise ValueError(f"No rows found for base model {BASE_FAMILY}/{BASE_MODEL}")
    if target.empty:
        raise ValueError(f"No rows found for target model {TARGET_FAMILY}/{TARGET_MODEL}")

    # Restrict to IOI-related heads in base: ioi_only + shared
    base_ioi = base[base["category"].isin(["ioi_only", "shared"])].copy()
    if base_ioi.empty:
        raise ValueError(
            f"No IOI-related heads (ioi_only/shared) found for {BASE_FAMILY}/{BASE_MODEL}"
        )

    # Rank by |Δ_ioi| and take top-K as "hero IOI heads"
    base_ioi.sort_values("abs_delta_ioi", ascending=False, inplace=True)
    hero = base_ioi.head(TOP_K).copy()

    records = []
    for _, row in hero.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])

        # Look up the *same index* head in the target model
        match = target[(target["layer"] == layer) & (target["head"] == head)]

        if match.empty:
            target_delta = None
            target_abs = None
            target_cat = None
        else:
            m = match.iloc[0]
            target_delta = float(m["delta_ioi"])
            target_abs = float(m["abs_delta_ioi"])
            # pandas Series supports .get(key, default)
            target_cat = m.get("category", None)

        records.append(
            {
                "layer": layer,
                "head": head,
                "base_family": BASE_FAMILY,
                "base_model": BASE_MODEL,
                "base_delta_ioi": float(row["delta_ioi"]),
                "base_abs_delta_ioi": float(row["abs_delta_ioi"]),
                "base_category": row["category"],
                "target_family": TARGET_FAMILY,
                "target_model": TARGET_MODEL,
                "target_delta_ioi": target_delta,
                "target_abs_delta_ioi": target_abs,
                "target_category": target_cat,
            }
        )

    out_df = pd.DataFrame.from_records(records)
    out_path = OUT_DIR / "ioi_transfer_pythia410m_to_gpt2medium.csv"
    out_df.to_csv(out_path, index=False)

    # Quick text summary
    print(f"Wrote transfer table to {out_path}")
    print()
    print(f"Base hero IOI heads (top K)  : {len(out_df)}")
    print(
        "Mean |Δ_ioi| in base model    : "
        f"{out_df['base_abs_delta_ioi'].mean():.3f}"
    )

    if out_df["target_abs_delta_ioi"].notnull().any():
        print(
            "Mean |Δ_ioi| in target model  : "
            f"{out_df['target_abs_delta_ioi'].dropna().mean():.3f}"
        )
    else:
        print("No matching heads found in target model at same indices.")


if __name__ == "__main__":
    main()

