#!/usr/bin/env python
import os
import pandas as pd

JOINT_PATH = "paper/tables/joint_ioi_anti_repeat_all.csv"
OUT_PATH = "results/case_study_heads_pythia410m_gpt2medium.csv"

PYTHIA_MODEL = "pythia-410m"
GPT2_MODEL = "gpt2-medium"


def main():
    df = pd.read_csv(JOINT_PATH)

    # Only require the columns we actually need
    required_cols = {"family", "model", "layer", "head", "delta_ioi", "delta_anti"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {JOINT_PATH}: {missing}")

    # Filter for the two models
    pythia = df[df["model"] == PYTHIA_MODEL]
    gpt2 = df[df["model"] == GPT2_MODEL]

    if pythia.empty:
        raise ValueError(f"No rows found for model={PYTHIA_MODEL} in {JOINT_PATH}")
    if gpt2.empty:
        raise ValueError(f"No rows found for model={GPT2_MODEL} in {JOINT_PATH}")

    # Align by (layer, head)
    merged = (
        pythia[["layer", "head", "delta_ioi", "delta_anti"]]
        .rename(
            columns={
                "delta_ioi": "delta_ioi_pythia",
                "delta_anti": "delta_anti_pythia",
            }
        )
        .merge(
            gpt2[["layer", "head", "delta_ioi", "delta_anti"]],
            on=["layer", "head"],
            how="inner",
        )
        .rename(
            columns={
                "delta_ioi": "delta_ioi_gpt2",
                "delta_anti": "delta_anti_gpt2",
            }
        )
    )

    if merged.empty:
        raise ValueError(
            "No overlapping (layer, head) positions between Pythia-410M and GPT-2-medium."
        )

    # IOI strengths
    merged["abs_delta_ioi_pythia"] = merged["delta_ioi_pythia"].abs()
    merged["abs_delta_ioi_gpt2"] = merged["delta_ioi_gpt2"].abs()
    merged["ioi_drop"] = merged["abs_delta_ioi_pythia"] - merged["abs_delta_ioi_gpt2"]

    # Heuristic thresholds
    strong_pythia_thresh = 0.20
    weak_gpt2_thresh = 0.03

    candidates = merged[
        (merged["abs_delta_ioi_pythia"] >= strong_pythia_thresh)
        & (merged["abs_delta_ioi_gpt2"] <= weak_gpt2_thresh)
    ].copy()

    if candidates.empty:
        print("No candidates found under current thresholds; try relaxing thresholds.")
        # As a fallback, just pick top 3 by largest IOI drop
        merged_sorted = merged.sort_values("ioi_drop", ascending=False)
        top3 = merged_sorted.head(3).copy()
        print("\nUsing fallback: top 3 heads by IOI drop:")
    else:
        candidates = candidates.sort_values("ioi_drop", ascending=False)
        top3 = candidates.head(3).copy()
        print("Selected case-study heads (filtered by thresholds):")

    top3.insert(0, "base_model", PYTHIA_MODEL)
    top3.insert(1, "target_model", GPT2_MODEL)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    top3.to_csv(OUT_PATH, index=False)

    print(top3[["layer", "head", "delta_ioi_pythia", "delta_ioi_gpt2", "ioi_drop"]])
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()

