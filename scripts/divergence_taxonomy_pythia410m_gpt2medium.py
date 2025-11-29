from __future__ import annotations

from pathlib import Path
import pandas as pd

HEADS_TABLE = Path("paper/tables/joint_ioi_anti_repeat_heads.csv")
OUT_DIR = Path("results/transfer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_FAMILY = "pythia"
BASE_MODEL = "pythia-410m"

TARGET_FAMILY = "gpt2"
TARGET_MODEL = "gpt2-medium"


def load_heads(path: Path) -> pd.DataFrame:
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


def classify_divergence(base_cat: str, target_cat: str | None) -> str:
    """
    Simple divergence taxonomy from the perspective of IOI-like heads in the base model.
    """
    ioi_like = {"ioi_only", "shared"}

    if target_cat is None:
        return "missing_in_target"

    if base_cat in ioi_like and target_cat in ioi_like:
        return "conserved_ioi_like"

    if base_cat in ioi_like and target_cat == "anti_only":
        return "inverted"

    if base_cat in ioi_like and target_cat == "weak":
        return "lost"

    if base_cat in ioi_like:
        return "other_change"

    return "non_ioi_base"  # should not appear if we pre-filter base to IOI-like


def main() -> None:
    df = load_heads(HEADS_TABLE)

    base = df[(df["family"] == BASE_FAMILY) & (df["model"] == BASE_MODEL)].copy()
    target = df[
        (df["family"] == TARGET_FAMILY) & (df["model"] == TARGET_MODEL)
    ].copy()

    if base.empty:
        raise ValueError(f"No rows for base model {BASE_FAMILY}/{BASE_MODEL}")
    if target.empty:
        raise ValueError(f"No rows for target model {TARGET_FAMILY}/{TARGET_MODEL}")

    # Restrict to IOI-like heads in the base model
    base_ioi = base[base["category"].isin(["ioi_only", "shared"])].copy()
    if base_ioi.empty:
        raise ValueError(
            f"No IOI-like heads (ioi_only/shared) found for {BASE_FAMILY}/{BASE_MODEL}"
        )

    records = []
    for _, row in base_ioi.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])
        base_cat = row["category"]

        match = target[(target["layer"] == layer) & (target["head"] == head)]

        if match.empty:
            target_cat = None
            target_delta_ioi = None
            target_abs_delta_ioi = None
        else:
            m = match.iloc[0]
            target_cat = m["category"]
            target_delta_ioi = float(m["delta_ioi"])
            target_abs_delta_ioi = float(m["abs_delta_ioi"])

        div_type = classify_divergence(base_cat, target_cat)

        records.append(
            {
                "layer": layer,
                "head": head,
                "base_family": BASE_FAMILY,
                "base_model": BASE_MODEL,
                "base_category": base_cat,
                "base_delta_ioi": float(row["delta_ioi"]),
                "base_abs_delta_ioi": float(row["abs_delta_ioi"]),
                "target_family": TARGET_FAMILY,
                "target_model": TARGET_MODEL,
                "target_category": target_cat,
                "target_delta_ioi": target_delta_ioi,
                "target_abs_delta_ioi": target_abs_delta_ioi,
                "divergence_type": div_type,
            }
        )

    out_df = pd.DataFrame.from_records(records)
    out_path = OUT_DIR / "ioi_divergence_pythia410m_to_gpt2medium.csv"
    out_df.to_csv(out_path, index=False)

    # Compact summary
    print(f"Wrote divergence table to {out_path}")
    print(f"Total IOI-like heads in base model: {len(out_df)}\n")

    counts = out_df["divergence_type"].value_counts().sort_index()
    print("Divergence type counts:")
    for div_type, cnt in counts.items():
        frac = cnt / len(out_df)
        print(f"  {div_type:20s}  n={cnt:3d}  ({frac:.2%})")


if __name__ == "__main__":
    main()

