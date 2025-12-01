#!/usr/bin/env python

import os
import csv
from collections import defaultdict
from typing import Dict, Tuple, List
import math


def ensure_dirs():
    os.makedirs("results", exist_ok=True)


def summarize_ioi_transfer_generic(
    in_path: str,
    out_path: str,
    metric_col: str = "tgt_abs_delta_ioi",
):
    """
    Summarize IOI transfer results across all base/target family+model pairs.

    Input: paper/tables/ioi_transfer_generic.csv
    Expected columns include at least:
      - base_family
      - base_model
      - tgt_family
      - tgt_model
      - <metric_col> (e.g. 'tgt_abs_delta_ioi')

    Output: results/ioi_transfer_generic_summary.csv with aggregates per pair.
    """

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input CSV not found at: {in_path}")

    # group key: (base_family, base_model, tgt_family, tgt_model)
    groups: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)

    with open(in_path, "r") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if metric_col not in header:
            raise ValueError(
                f"Column '{metric_col}' not found in {in_path}. "
                f"Available columns: {header}"
            )

        required = ["base_family", "base_model", "tgt_family", "tgt_model"]
        for col in required:
            if col not in header:
                raise ValueError(
                    f"Required column '{col}' not found in {in_path}. "
                    f"Available columns: {header}"
                )

        for row in reader:
            try:
                value = float(row[metric_col])
            except (ValueError, TypeError):
                # skip non-numeric
                continue

            key = (
                row["base_family"],
                row["base_model"],
                row["tgt_family"],
                row["tgt_model"],
            )
            groups[key].append(value)

    # Write summary CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "base_family",
                "base_model",
                "tgt_family",
                "tgt_model",
                "n_heads",
                f"mean_{metric_col}",
                f"std_{metric_col}",
                f"min_{metric_col}",
                f"max_{metric_col}",
            ]
        )

        for key, vals in sorted(groups.items()):
            base_family, base_model, tgt_family, tgt_model = key
            n = len(vals)
            if n == 0:
                continue
            mean_val = sum(vals) / n
            var = sum((v - mean_val) ** 2 for v in vals) / n
            std_val = math.sqrt(var)
            min_val = min(vals)
            max_val = max(vals)
            writer.writerow(
                [
                    base_family,
                    base_model,
                    tgt_family,
                    tgt_model,
                    n,
                    mean_val,
                    std_val,
                    min_val,
                    max_val,
                ]
            )

    print(f"[OUT] Wrote IOI transfer summary to {out_path}")

    # Also print highlighted info about pythia<->gpt2 asymmetry
    print("\n[HIGHLIGHT] Pythia -> GPT-2 pairs:")
    for (bf, bm, tf, tm), vals in groups.items():
        if bf.lower().startswith("pythia") and tf.lower().startswith("gpt2"):
            n = len(vals)
            mean_val = sum(vals) / n
            print(
                f"  base=({bf}, {bm}) -> target=({tf}, {tm}): "
                f"n={n}, mean_{metric_col}={mean_val:.6f}"
            )

    print("\n[HIGHLIGHT] GPT-2 -> Pythia pairs:")
    any_gpt2_to_pythia = False
    for (bf, bm, tf, tm), vals in groups.items():
        if bf.lower().startswith("gpt2") and tf.lower().startswith("pythia"):
            any_gpt2_to_pythia = True
            n = len(vals)
            mean_val = sum(vals) / n
            print(
                f"  base=({bf}, {bm}) -> target=({tf}, {tm}): "
                f"n={n}, mean_{metric_col}={mean_val:.6f}"
            )

    if not any_gpt2_to_pythia:
        print("  [INFO] No GPT-2 -> Pythia entries found in this table.")


def main():
    ensure_dirs()

    in_path = "paper/tables/ioi_transfer_generic.csv"
    out_path = "results/ioi_transfer_generic_summary.csv"
    metric_col = "tgt_abs_delta_ioi"

    summarize_ioi_transfer_generic(
        in_path=in_path,
        out_path=out_path,
        metric_col=metric_col,
    )


if __name__ == "__main__":
    main()

