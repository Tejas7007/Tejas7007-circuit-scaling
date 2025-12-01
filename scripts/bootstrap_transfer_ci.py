#!/usr/bin/env python

import os
import csv
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)


def load_metric_column(path: str, colname: str) -> np.ndarray:
    """
    Load a numeric column from a CSV file.

    Assumes the file has a header row and that `colname` is one of the fields.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find CSV file at: {path}")

    values: List[float] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if colname not in reader.fieldnames:
            raise ValueError(
                f"Column '{colname}' not found in {path}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            try:
                val = float(row[colname])
                values.append(val)
            except (ValueError, TypeError):
                # Skip rows where the value is missing or non-numeric
                continue

    if len(values) == 0:
        raise ValueError(f"No valid numeric values found in column '{colname}' of {path}")

    return np.array(values, dtype=float)


def plot_bootstrap_distribution(
    samples: np.ndarray,
    sample_mean: float,
    ci_low: float,
    ci_high: float,
    title: str,
    out_path: str,
):
    """
    Plot a histogram of bootstrap means with vertical lines for
    the sample mean and confidence interval.
    """
    plt.figure(figsize=(7, 4))
    plt.hist(samples, bins=60, density=True, alpha=0.7, label="Bootstrap means")
    plt.axvline(sample_mean, linestyle="-", linewidth=2, label=f"Mean = {sample_mean:.4f}")
    plt.axvline(ci_low, linestyle="--", linewidth=1.5, label=f"95% CI low = {ci_low:.4f}")
    plt.axvline(ci_high, linestyle="--", linewidth=1.5, label=f"95% CI high = {ci_high:.4f}")

    plt.title(title)
    plt.xlabel("Mean metric value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def bootstrap_and_save(
    name: str,
    csv_path: str,
    colname: str,
    n_bootstrap: int,
    alpha: float,
    seed: int,
    writer: csv.writer,
):
    """
    Helper to:
      - load data
      - run bootstrap on the mean of `colname`
      - save summary row
      - save plot
    """
    print(f"[INFO] Bootstrapping for {name} from {csv_path} (column '{colname}')")

    data = load_metric_column(csv_path, colname=colname)
    n = len(data)
    rng = np.random.default_rng(seed)

    # Bootstrap resampling for the mean
    idx = rng.integers(low=0, high=n, size=(n_bootstrap, n))
    samples = data[idx].mean(axis=1)

    sample_mean = float(np.mean(data))
    ci_low, ci_high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    print(
        f"[RESULT] {name}: mean({colname}) = {sample_mean:.6f}, "
        f"{100 * (1 - alpha):.1f}% CI = [{ci_low:.6f}, {ci_high:.6f}] "
        f"(n={n}, boot={n_bootstrap})"
    )

    # Write summary row
    writer.writerow(
        [
            name,
            csv_path,
            colname,
            n,
            n_bootstrap,
            sample_mean,
            ci_low,
            ci_high,
            alpha,
            seed,
        ]
    )

    # Plot
    fig_name = f"figs/bootstrap_{name}_mean_{colname}.png"
    plot_bootstrap_distribution(
        samples,
        sample_mean,
        ci_low,
        ci_high,
        title=f"Bootstrap Mean {colname} ({name})",
        out_path=fig_name,
    )
    print(f"[OUT] Saved bootstrap plot for {name} to {fig_name}")


def main():
    ensure_dirs()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    N_BOOTSTRAP = 100_000
    ALPHA = 0.05
    SEED = 2025

    # We use the transfer metric that *does* exist in your IOI table:
    #   'target_abs_delta_ioi'
    METRIC_COL = "target_abs_delta_ioi"

    # Only IOI experiment for now
    experiments = [
        (
            "ioi_pythia410m_to_gpt2medium",
            "paper/tables/ioi_transfer_pythia410m_to_gpt2medium.csv",
            METRIC_COL,
        ),
    ]

    summary_path = "results/bootstrap_transfer_ci.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "csv_path",
                "column",
                "n_heads",
                "n_bootstrap",
                "sample_mean",
                "ci_low",
                "ci_high",
                "alpha",
                "seed",
            ]
        )

        for (name, csv_path, colname) in experiments:
            bootstrap_and_save(
                name=name,
                csv_path=csv_path,
                colname=colname,
                n_bootstrap=N_BOOTSTRAP,
                alpha=ALPHA,
                seed=SEED,
                writer=writer,
            )

    print(f"[OUT] Wrote bootstrap summary to {summary_path}")


if __name__ == "__main__":
    main()


