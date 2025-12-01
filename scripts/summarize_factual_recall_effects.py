#!/usr/bin/env python
"""
Summarize factual recall head scores for Pythia-410M and GPT-2-medium.

Reads:
  results/factual_recall_head_scores_pythia-410m.csv
  results/factual_recall_head_scores_gpt2-medium.csv

For each numeric column, prints:
  - mean, std, min, max for each model
  - Cohen's d between Pythia and GPT-2
"""

import os
import numpy as np
import pandas as pd


PYTHIA_PATH = "results/factual_recall_head_scores_pythia-410m.csv"
GPT2_PATH = "results/factual_recall_head_scores_gpt2-medium.csv"


def summarize_df(df: pd.DataFrame, label: str):
    print(f"=== {label} ===")
    num_cols = df.select_dtypes(include=[float, int]).columns
    if len(num_cols) == 0:
        print("  [no numeric columns found]")
        print()
        return num_cols

    for col in num_cols:
        series = df[col].dropna()
        print(f"  {col}: mean={series.mean():.6f}, "
              f"std={series.std():.6f}, "
              f"min={series.min():.6f}, "
              f"max={series.max():.6f}")
    print()
    return num_cols


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for two independent samples."""
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sx2 = x.var(ddof=1)
    sy2 = y.var(ddof=1)
    sp2 = ((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)
    if sp2 <= 0:
        return 0.0
    return float((x.mean() - y.mean()) / np.sqrt(sp2))


def main():
    if not os.path.exists(PYTHIA_PATH):
        raise FileNotFoundError(PYTHIA_PATH)
    if not os.path.exists(GPT2_PATH):
        raise FileNotFoundError(GPT2_PATH)

    df_p = pd.read_csv(PYTHIA_PATH)
    df_g = pd.read_csv(GPT2_PATH)

    # Basic summaries
    cols_p = summarize_df(df_p, "Pythia-410M factual recall heads")
    cols_g = summarize_df(df_g, "GPT-2-medium factual recall heads")

    common_cols = [c for c in cols_p if c in cols_g]

    print("=== Cohen's d: Pythia vs GPT-2 (per numeric column) ===")
    for col in common_cols:
        x = df_p[col].values
        y = df_g[col].values
        d = cohens_d(x, y)
        print(f"  {col}: d = {d:.3f}")
    print()


if __name__ == "__main__":
    main()

