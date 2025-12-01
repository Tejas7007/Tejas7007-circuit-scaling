#!/usr/bin/env python

import os
import json
import math
import random
import csv

import numpy as np
import matplotlib.pyplot as plt


def simulate_null_means(k: int, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate the null distribution of the mean of K fractional ranks,
    where each fractional rank is Uniform(0, 1).

    Returns an array of length n_trials with the simulated means.
    """
    # Shape: (n_trials, k) -> means over axis=1
    samples = rng.random(size=(n_trials, k))
    means = samples.mean(axis=1)
    return means


def compute_p_value(null_means: np.ndarray, observed_mean: float, tail: str = "two-sided") -> float:
    """
    Compute a p-value for the observed_mean under the null distribution.

    tail = "two-sided": 2 * min(P(mean <= obs), P(mean >= obs))
    tail = "left":      P(mean <= obs)
    tail = "right":     P(mean >= obs)
    """
    if tail not in {"two-sided", "left", "right"}:
        raise ValueError(f"Invalid tail: {tail}")

    n = len(null_means)
    less_eq = np.mean(null_means <= observed_mean)
    greater_eq = np.mean(null_means >= observed_mean)

    if tail == "left":
        return float(less_eq)
    elif tail == "right":
        return float(greater_eq)
    else:
        return float(2 * min(less_eq, greater_eq))


def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)


def main():
    """
    Simulate random baselines for the mean fractional rank of K heads and
    compare against the observed transfer results in the paper.

    You can tweak the OBS_* constants below if your actual numbers differ.
    """

    ensure_dirs()

    # ------------------------------------------------------------------
    # Configuration: edit these if you want to plug in updated numbers
    # ------------------------------------------------------------------
    K = 20               # number of transferred heads
    N_TRIALS = 100_000   # Monte Carlo samples for null distribution

    # Observed mean fractional ranks from the current paper draft
    # (Pythia-410M -> GPT-2-Medium; adjust if needed)
    OBS_IOI_MEAN = 0.57
    OBS_ANTI_MEAN = 0.67

    SEED = 12345
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # Simulate null distributions
    # ------------------------------------------------------------------
    print(f"[INFO] Simulating null distribution with K={K}, trials={N_TRIALS}...")
    null_means = simulate_null_means(K, N_TRIALS, rng)

    # Basic stats of the null
    null_mean = float(np.mean(null_means))
    null_std = float(np.std(null_means))
    ci_low, ci_high = np.percentile(null_means, [2.5, 97.5])

    print(f"[INFO] Null mean ≈ {null_mean:.4f}, std ≈ {null_std:.4f}")
    print(f"[INFO] 95% null CI for mean ≈ [{ci_low:.4f}, {ci_high:.4f}]")

    # ------------------------------------------------------------------
    # Compare IOI observed mean to null
    # ------------------------------------------------------------------
    p_ioi_two_sided = compute_p_value(null_means, OBS_IOI_MEAN, tail="two-sided")
    p_ioi_left = compute_p_value(null_means, OBS_IOI_MEAN, tail="left")
    p_ioi_right = compute_p_value(null_means, OBS_IOI_MEAN, tail="right")

    print(f"[IOI] Observed mean = {OBS_IOI_MEAN:.4f}")
    print(f"[IOI] p_two_sided = {p_ioi_two_sided:.4e}, "
          f"p_left = {p_ioi_left:.4e}, p_right = {p_ioi_right:.4e}")

    # ------------------------------------------------------------------
    # Compare anti-repeat observed mean to null
    # ------------------------------------------------------------------
    p_anti_two_sided = compute_p_value(null_means, OBS_ANTI_MEAN, tail="two-sided")
    p_anti_left = compute_p_value(null_means, OBS_ANTI_MEAN, tail="left")
    p_anti_right = compute_p_value(null_means, OBS_ANTI_MEAN, tail="right")

    print(f"[ANTI] Observed mean = {OBS_ANTI_MEAN:.4f}")
    print(f"[ANTI] p_two_sided = {p_anti_two_sided:.4e}, "
          f"p_left = {p_anti_left:.4e}, p_right = {p_anti_right:.4e}")

    # ------------------------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------------------------
    summary_path = "results/random_baseline_transfer_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metric",
            "K",
            "n_trials",
            "null_mean",
            "null_std",
            "null_ci_low",
            "null_ci_high",
            "observed_mean",
            "p_two_sided",
            "p_left",
            "p_right",
        ])
        writer.writerow([
            "ioi_fractional_rank",
            K,
            N_TRIALS,
            null_mean,
            null_std,
            ci_low,
            ci_high,
            OBS_IOI_MEAN,
            p_ioi_two_sided,
            p_ioi_left,
            p_ioi_right,
        ])
        writer.writerow([
            "antirepeat_fractional_rank",
            K,
            N_TRIALS,
            null_mean,
            null_std,
            ci_low,
            ci_high,
            OBS_ANTI_MEAN,
            p_anti_two_sided,
            p_anti_left,
            p_anti_right,
        ])

    print(f"[OUT] Saved random baseline summary to {summary_path}")

    # ------------------------------------------------------------------
    # Plot histogram of null means with observed markers
    # ------------------------------------------------------------------
    fig_path = "figs/random_baseline_fractional_rank.png"
    plt.figure(figsize=(7, 4))
    plt.hist(null_means, bins=60, density=True, alpha=0.7, label="Null mean distribution")

    # Vertical lines for IOI and anti-repeat observed means
    plt.axvline(OBS_IOI_MEAN, linestyle="--", linewidth=2, label=f"Observed IOI = {OBS_IOI_MEAN:.2f}")
    plt.axvline(OBS_ANTI_MEAN, linestyle=":", linewidth=2, label=f"Observed anti-repeat = {OBS_ANTI_MEAN:.2f}")

    plt.title(f"Null Distribution of Mean Fractional Rank (K={K})")
    plt.xlabel("Mean fractional rank")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[OUT] Saved histogram plot to {fig_path}")


if __name__ == "__main__":
    main()

