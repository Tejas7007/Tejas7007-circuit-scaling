#!/usr/bin/env python
import os
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

PAIRS = [
    ("pythia-410m", "gpt2-medium"),
    ("pythia-410m", "gpt-neo-125M"),
]

def load_summary(src, tgt):
    path = RESULTS_DIR / f"factual_recall_transfer_{src}_to_{tgt}_summary.txt"
    if not path.exists():
        return None
    stats = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            # ints vs float
            if key in ["total", "missing", "weak", "same-sign", "inverted"]:
                stats[key] = int(float(val))
            elif key == "mean_frac_rank":
                stats[key] = float(val)
    stats["src_family"] = src
    stats["tgt_family"] = tgt
    return stats

def main():
    rows = []
    for src, tgt in PAIRS:
        rec = load_summary(src, tgt)
        if rec is None:
            print(f"[WARN] Missing summary for {src} → {tgt}")
            continue
        rows.append(rec)

    if not rows:
        print("[ERROR] No summaries found.")
        return

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "factual_recall_transfer_across_families.csv"
    df.to_csv(out_csv, index=False)
    print("[INFO] Wrote", out_csv)
    print()
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

