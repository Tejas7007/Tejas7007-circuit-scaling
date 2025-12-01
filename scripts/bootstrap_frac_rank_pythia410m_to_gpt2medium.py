#!/usr/bin/env python

import os
import csv
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import math


def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)


def load_tgt_frac_ranks_for_pair(
    path: str,
    base_family: str,
    base_model: str,
    tgt_family: str,
    tgt_model: str,
    colname: str = "tgt_frac_rank",
) -> np.ndarray:
    """
    Load tgt_frac_rank values for a specific (base_family, base_model, tgt_family, tgt_model)
    from paper/tables/ioi_transfer_generic.csv.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find CSV file at: {path}")

    vals: List[float] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        required = ["base_family", "base_model", "tgt_family", "tgt_model", colname]
        for col in required:
            if col not in header:
                raise ValueError(
                    f"Required column '{col}' not found in {path}. "
                    f"Available columns: {header}"
                )

        for row in reader:
            if (
                row["base_family"] == base_family
                and row["base_model"] == base_model
                and row["tgt_family"] == tgt_family
                and row["tgt_model"] == tgt_model
            ):
                try:
                    v = float(row[colname])
                except (ValueError, TypeError):
                    continue
                vals.append(v)

    if len(vals) == 0:
        raise ValueError(
            f"No rows found in {path} for "
            f"(base_family={base_family}, base_model={base_model}, "
            f"tgt_family={tgt_family}, tgt_model={tgt_model})."
        )

    return np.array(vals, dtype=float)


def plot_bootstrap_distribution(
    samples: np.ndarray,
    sample_mean: float,
    ci_low: float,
    ci_high: float,
    title: str,
    out_path: str,
):
    """
    Plot histogram of bootstrap means with CI and sample mean.
    """
    plt.figure(figsize=(7, 4))
    plt.hist(samples, bins=60, density=True, alpha=0.7, label="Bootstrap means")
    plt.axvline(sample_mean, linestyle="-", linewidth=2, label=f"Mean = {sample_mean:.3f}")
    plt.axvline(ci_low, linestyle="--", linewidth=1.5, label=f"95% CI low = {ci_low:.3f}")
    plt.axvline(ci_high, linestyle="--", linewidth=1.5, label=f"95% CI high = {ci_high:.3f}")

    plt.title(title)
    plt.xlabel("Mean target fractional rank")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ensure_dirs()

    # ------------------------------------------------------------------
    # Configuration: Pythia-410M -> GPT-2-Medium
    # ------------------------------------------------------------------
    IN_PATH = "paper/tables/ioi_transfer_generic.csv"
    BASE_FAMILY = "pythia"
    BASE_MODEL = "pythia-410m"
    TGT_FAMILY = "gpt2"
    TGT_MODEL = "gpt2-medium"
    COLNAME = "tgt_frac_rank"

    N_BOOTSTRAP = 100_000
    ALPHA = 0.05
    SEED = 314159

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    frac_ranks = load_tgt_frac_ranks_for_pair(
        IN_PATH,
        base_family=BASE_FAMILY,
        base_model=BASE_MODEL,
        tgt_family=TGT_FAMILY,
        tgt_model=TGT_MODEL,
        colname=COLNAME,
    )

    n = len(frac_ranks)
    sample_mean = float(np.mean(frac_ranks))
    print(f"[INFO] Loaded {n} fractional ranks for "
          f"{BASE_FAMILY}-{BASE_MODEL} -> {TGT_FAMILY}-{TGT_MODEL}")
    print(f"[INFO] Sample mean {COLNAME} = {sample_mean:.6f}")

    # ------------------------------------------------------------------
    # Bootstrap distribution of the mean
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    idx = rng.integers(low=0, high=n, size=(N_BOOTSTRAP, n))
    boot_means = frac_ranks[idx].mean(axis=1)

    ci_low, ci_high = np.percentile(boot_means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])

    print(
        f"[BOOTSTRAP] Mean({COLNAME}) = {sample_mean:.6f}, "
        f"{100 * (1 - ALPHA):.1f}% CI = [{ci_low:.6f}, {ci_high:.6f}] "
        f"(n={n}, boot={N_BOOTSTRAP})"
    )

    # ------------------------------------------------------------------
    # Analytic test vs Uniform(0,1) null (mean = 0.5)
    # For mean of n Uniform(0,1), var = 1 / (12 * n)
    # ------------------------------------------------------------------
    null_mean = 0.5
    null_var = 1.0 / (12.0 * n)
    null_std = math.sqrt(null_var)

    z = (sample_mean - null_mean) / null_std
    # two-sided p-value under normal approximation
    from math import erf, sqrt

    def normal_cdf(x: float) -> float:
        # standard normal CDF
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    p_two_sided = 2 * min(normal_cdf(z), 1 - normal_cdf(z))

    print(
        f"[NULL TEST] Null mean = {null_mean:.3f}, "
        f"null std of mean = {null_std:.6f}, z = {z:.3f}, "
        f"approx p_two_sided ≈ {p_two_sided:.4f}"
    )

    # ------------------------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------------------------
    out_csv = "results/frac_rank_pythia410m_to_gpt2medium_bootstrap_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "base_family",
                "base_model",
                "tgt_family",
                "tgt_model",
                "n_heads",
                "sample_mean_tgt_frac_rank",
                "ci_low",
                "ci_high",
                "n_bootstrap",
                "alpha",
                "null_mean",
                "null_std_mean",
                "z_score",
                "p_two_sided_normal_approx",
            ]
        )
        writer.writerow(
            [
                BASE_FAMILY,
                BASE_MODEL,
                TGT_FAMILY,
                TGT_MODEL,
                n,
                sample_mean,
                ci_low,
                ci_high,
                N_BOOTSTRAP,
                ALPHA,
                null_mean,
                null_std,
                z,
                p_two_sided,
            ]
        )

    print(f"[OUT] Wrote summary to {out_csv}")

    # ------------------------------------------------------------------
    # Save plot
    # ------------------------------------------------------------------
    fig_path = "figs/frac_rank_pythia410m_to_gpt2medium_bootstrap_mean.png"
    plot_bootstrap_distribution(
        boot_means,
        sample_mean,
        ci_low,
        ci_high,
        title="Bootstrap Mean Target Fractional Rank (Pythia-410M -> GPT-2-Medium)",
        out_path=fig_path,
    )
    print(f"[OUT] Saved bootstrap histogram to {fig_path}")


if __name__ == "__main__":
    main()

