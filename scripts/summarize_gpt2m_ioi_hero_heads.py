import os
import json
import glob
import pandas as pd


def main():
    in_dir = "results/gpt2m_ioi_hero_analysis"
    out_path = "results/gpt2m_ioi_hero_summary.csv"

    if not os.path.isdir(in_dir):
        raise FileNotFoundError(
            f"{in_dir} not found. Run analyze_gpt2medium_ioi_hero_heads.py first."
        )

    rows = []
    for path in glob.glob(os.path.join(in_dir, "gpt2-medium_L*H*_summary.json")):
        with open(path, "r") as f:
            data = json.load(f)
        rows.append(data)

    if not rows:
        raise RuntimeError(f"No JSON summaries found in {in_dir}")

    df = pd.DataFrame(rows)

    # Sort by delta_margin (most negative = strongest pro-IOI)
    df_sorted = df.sort_values("delta_margin").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_sorted.to_csv(out_path, index=False)

    print(f"[OUT] Saved summary of GPT-2-M IOI hero heads to {out_path}")
    print(df_sorted[["layer", "head", "tag", "delta_margin", "base_mean_margin", "ablated_mean_margin"]])


if __name__ == "__main__":
    main()

