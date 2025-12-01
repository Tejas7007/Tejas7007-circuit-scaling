#!/usr/bin/env python
import pandas as pd
from pathlib import Path


RESULTS = Path("results")
INPUT_CSV = RESULTS / "ioi_what_transfers_summary.csv"
OUT_BASELINE = RESULTS / "pythia_internal_baseline.csv"
OUT_REPORT = RESULTS / "pythia_internal_vs_cross_family_summary.txt"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_CSV}. "
            "Make sure scripts/summarize_what_transfers.py has been run."
        )

    df = pd.read_csv(INPUT_CSV)

    # Basic sanity check
    required_cols = {
        "base_family",
        "base_model",
        "tgt_family",
        "tgt_model",
        "n_heads",
        "n_good_tgt_ioi_like",
        "frac_good_tgt_ioi_like",
        "mean_tgt_abs_delta_ioi",
        "std_tgt_abs_delta_ioi",
        "mean_tgt_frac_rank",
        "std_tgt_frac_rank",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {INPUT_CSV}: {missing}")

    # --- Within-family Pythia baseline ---
    df_within = df[(df["base_family"] == "pythia") & (df["tgt_family"] == "pythia")]
    if df_within.empty:
        print("[WARN] No within-family Pythia rows found in ioi_what_transfers_summary.csv")
    else:
        df_within.to_csv(OUT_BASELINE, index=False)
        print(f"[OUT] Wrote Pythia within-family baseline to {OUT_BASELINE}")

    # --- Cross-family comparison subset where base_family == 'pythia' and tgt_family != 'pythia' ---
    df_cross = df[(df["base_family"] == "pythia") & (df["tgt_family"] != "pythia")]

    # Aggregate stats for quick reporting
    def agg_stats(sub):
        if sub.empty:
            return None
        return {
            "n_pairs": len(sub),
            "mean_frac_good": sub["frac_good_tgt_ioi_like"].mean(),
            "std_frac_good": sub["frac_good_tgt_ioi_like"].std(),
            "mean_abs_delta": sub["mean_tgt_abs_delta_ioi"].mean(),
            "std_abs_delta": sub["mean_tgt_abs_delta_ioi"].std(),
            "mean_frac_rank": sub["mean_tgt_frac_rank"].mean(),
            "std_frac_rank": sub["mean_tgt_frac_rank"].std(),
        }

    stats_within = agg_stats(df_within)
    stats_cross = agg_stats(df_cross)

    # Save a human-readable summary you can quote directly in the paper
    lines = []
    lines.append("=== Pythia Within-Family Baseline vs Cross-Family Transfer ===\n")

    if stats_within:
        lines.append("Within-family (Pythia -> Pythia):\n")
        lines.append(f"  #pairs:              {stats_within['n_pairs']}\n")
        lines.append(
            f"  mean frac good IOI-like heads: "
            f"{stats_within['mean_frac_good']:.3f} ± {stats_within['std_frac_good']:.3f}\n"
        )
        lines.append(
            f"  mean |Δ IOI|:        "
            f"{stats_within['mean_abs_delta']:.3f} ± {stats_within['std_abs_delta']:.3f}\n"
        )
        lines.append(
            f"  mean target frac rank: "
            f"{stats_within['mean_frac_rank']:.3f} ± {stats_within['std_frac_rank']:.3f}\n"
        )
        lines.append("\n")
    else:
        lines.append("Within-family (Pythia -> Pythia): NO DATA\n\n")

    if stats_cross:
        lines.append("Cross-family (Pythia -> non-Pythia):\n")
        lines.append(f"  #pairs:              {stats_cross['n_pairs']}\n")
        lines.append(
            f"  mean frac good IOI-like heads: "
            f"{stats_cross['mean_frac_good']:.3f} ± {stats_cross['std_frac_good']:.3f}\n"
        )
        lines.append(
            f"  mean |Δ IOI|:        "
            f"{stats_cross['mean_abs_delta']:.3f} ± {stats_cross['std_abs_delta']:.3f}\n"
        )
        lines.append(
            f"  mean target frac rank: "
            f"{stats_cross['mean_frac_rank']:.3f} ± {stats_cross['std_frac_rank']:.3f}\n"
        )
        lines.append("\n")
    else:
        lines.append("Cross-family (Pythia -> non-Pythia): NO DATA\n\n")

    OUT_REPORT.write_text("".join(lines))
    print(f"[OUT] Wrote human-readable baseline summary to {OUT_REPORT}")
    print("\n=== SUMMARY (also saved to file) ===")
    print("".join(lines))


if __name__ == "__main__":
    main()

