#!/usr/bin/env python
"""
Compare CKA of circuit / selected heads vs random-head baseline.

Assumes:
  results/cka_alignment_pythia410m_gpt2medium.csv      # has column 'cka'
  results/cka_random_baseline_pythia-410m_gpt2-medium.npz  # has array 'cka_vals'
"""

import numpy as np
import pandas as pd
import os


def summarize_array(name, arr):
    arr = np.asarray(arr)
    print(f"== {name} ==")
    print(f"n      = {len(arr)}")
    print(f"mean   = {arr.mean():.6f}")
    print(f"std    = {arr.std():.6f}")
    print(f"p05    = {np.quantile(arr, 0.05):.6f}")
    print(f"p50    = {np.quantile(arr, 0.50):.6f}")
    print(f"p95    = {np.quantile(arr, 0.95):.6f}")
    print()


def main():
    # Paths are hard-coded for now; you can add CLI later if needed
    random_path = "results/cka_random_baseline_pythia-410m_gpt2-medium.npz"
    cka_path = "results/cka_alignment_pythia410m_gpt2medium.csv"

    if not os.path.exists(random_path):
        raise FileNotFoundError(random_path)
    if not os.path.exists(cka_path):
        raise FileNotFoundError(cka_path)

    # Load random baseline
    data = np.load(random_path)
    if "cka_vals" in data:
        random_vals = data["cka_vals"]
    else:
        # In case we ever save under a different key
        random_vals = list(data.values())[0]

    # Load structured circuit/head CKA
    df = pd.read_csv(cka_path)
    if "cka" in df.columns:
        circuit_vals = df["cka"].values
    else:
        # Fallback: assume last numeric column is CKA
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in CKA CSV.")
        circuit_vals = df[numeric_cols[-1]].values

    print("### CKA random baseline vs circuit/head alignment ###\n")
    summarize_array("Random baseline", random_vals)
    summarize_array("Circuit / selected heads", circuit_vals)


if __name__ == "__main__":
    main()

