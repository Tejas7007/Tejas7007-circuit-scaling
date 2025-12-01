#!/usr/bin/env python3
"""
learned_head_alignment.py

Learns a simple linear map that aligns Pythia -> GPT-2 heads
based on functional features (IOI, Anti-Repeat).
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import json
import os

IN_PATH = "results/joint_ioi_anti_repeat_all.csv"
OUT_PATH = "results/learned_head_alignment_results.json"


def build_feature_matrix(df: pd.DataFrame):
    """
    Build a simple functional feature matrix.

    The CSV only contains:
        delta_ioi, delta_anti
    So we compute:
        abs_delta_ioi = abs(delta_ioi)
        abs_delta_anti = abs(delta_anti)
    """
    df = df.copy()
    df["abs_delta_ioi"] = df["delta_ioi"].abs()
    df["abs_delta_anti"] = df["delta_anti"].abs()

    return df[["abs_delta_ioi", "abs_delta_anti"]].values


def main():
    print(f"[INFO] Loaded: {IN_PATH}")
    df = pd.read_csv(IN_PATH)

    # Build features
    X = build_feature_matrix(df)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Coordinates are (layer, head)
    coords = df[["layer", "head"]].values

    # Split Pythia vs GPT-2 rows
    is_pythia = df["family"].str.contains("pythia")
    is_gpt2 = df["family"].str.contains("gpt2")

    X_A = X_scaled[is_pythia]
    coords_A = coords[is_pythia]

    X_B = X_scaled[is_gpt2]
    coords_B = coords[is_gpt2]

    if len(X_A) == 0 or len(X_B) == 0:
        raise ValueError("No Pythia or GPT-2 rows found in table. Check your CSV.")

    print(f"[INFO] Pythia heads: {len(X_A)}, GPT-2 heads: {len(X_B)}")

    # Learn mapping from Pythia-feature -> GPT-2 coordinate
    knn = KNeighborsRegressor(
        n_neighbors=3,
        weights="distance",
    )
    knn.fit(X_A, coords_A)

    # Predict GPT-2 alignment for each GPT-2 head
    preds = knn.predict(X_B)

    results = []
    for target_coord, pred_coord in zip(coords_B, preds):
        results.append({
            "gpt2_layer": int(target_coord[0]),
            "gpt2_head": int(target_coord[1]),
            "pred_layer": float(pred_coord[0]),
            "pred_head": float(pred_coord[1])
        })

    # Save JSON
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[OUT] Wrote learned alignment results to {OUT_PATH}")
    print("[DONE] Learned head mapping complete.")


if __name__ == "__main__":
    main()

