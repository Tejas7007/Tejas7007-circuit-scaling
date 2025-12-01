#!/usr/bin/env python
import os
import pandas as pd
from transformers import AutoTokenizer
import scipy.stats as stats

# Input: per-name IOI strengths for both models
# Expected columns: name, ioistrength_pythia, ioistrength_gpt2
NAME_STATS_PATH = "results/ioi_name_strengths_pythia_gpt2.csv"

OUT_PATH = "paper/tables/tokenization_mismatch_ioi.csv"

PYTHIA_CHECKPOINT = "EleutherAI/pythia-410m-deduped"
GPT2_CHECKPOINT = "gpt2-medium"


def main():
    if not os.path.exists(NAME_STATS_PATH):
        raise FileNotFoundError(
            f"{NAME_STATS_PATH} not found.\n"
            "Create it with columns: name, ioistrength_pythia, ioistrength_gpt2 "
            "(e.g. from your IOI evaluation script), then rerun."
        )

    df = pd.read_csv(NAME_STATS_PATH)

    required_cols = {"name", "ioistrength_pythia", "ioistrength_gpt2"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {NAME_STATS_PATH}: {missing}")

    tok_pythia = AutoTokenizer.from_pretrained(PYTHIA_CHECKPOINT)
    tok_gpt2 = AutoTokenizer.from_pretrained(GPT2_CHECKPOINT)

    def count_tokens(tokenizer, text: str) -> int:
        return len(tokenizer.tokenize(text))

    pythia_tokens = []
    gpt2_tokens = []

    for name in df["name"]:
        pythia_tokens.append(count_tokens(tok_pythia, name))
        gpt2_tokens.append(count_tokens(tok_gpt2, name))

    df["pythia_tokens"] = pythia_tokens
    df["gpt2_tokens"] = gpt2_tokens
    df["delta_tokens"] = df["pythia_tokens"] - df["gpt2_tokens"]
    df["abs_delta_tokens"] = df["delta_tokens"].abs()

    # IOI strength difference
    df["ioi_diff"] = df["ioistrength_pythia"] - df["ioistrength_gpt2"]
    df["abs_ioi_diff"] = df["ioi_diff"].abs()

    # Correlations between tokenization mismatch and IOI mismatch
    pearson = stats.pearsonr(df["abs_delta_tokens"], df["abs_ioi_diff"])
    spearman = stats.spearmanr(df["abs_delta_tokens"], df["abs_ioi_diff"])

    print("Correlation between |Δ tokens| and |Δ IOI strength|:")
    print(f"  Pearson r = {pearson.statistic:.3f}, p = {pearson.pvalue:.1e}")
    print(f"  Spearman ρ = {spearman.statistic:.3f}, p = {spearman.pvalue:.1e}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[TABLE] Saved tokenization mismatch table to {OUT_PATH}")


if __name__ == "__main__":
    main()

