#!/usr/bin/env python
import itertools
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
OUT_CSV = RESULTS / "tokenization_effects_ioi.csv"
OUT_SUMMARY = RESULTS / "tokenization_effects_summary.txt"


# Simple IOI-style templates
TEMPLATES = [
    "When {A} and {B} went to the store, {A} gave a book to",
    "{A} and {B} were talking because {A} wanted to help",
    "{A} met {B} after school, and later {A} called",
]

# A small name list – we only need enough variety to see tokenization patterns
NAMES = [
    "John",
    "Mary",
    "Alice",
    "Bob",
    "Charlie",
    "Diane",
    "Edward",
    "Grace",
    "Hannah",
    "Isaac",
]


MODEL_SPECS = {
    "pythia-410m": "EleutherAI/pythia-410m-deduped",
    "gpt2-medium": "gpt2-medium",
    "gpt-neo-125M": "EleutherAI/gpt-neo-125M",
    "opt-125m": "facebook/opt-125m",
}


def build_prompts(max_pairs=20):
    pairs = list(itertools.permutations(NAMES, 2))
    prompts = []
    count = 0
    for (A, B) in pairs:
        for template in TEMPLATES:
            text = template.format(A=A, B=B)
            prompts.append({"A": A, "B": B, "prompt": text})
            count += 1
            if count >= max_pairs:
                return prompts
    return prompts


def main():
    prompts = build_prompts(max_pairs=30)
    df_prompts = pd.DataFrame(prompts)
    print(f"[INFO] Built {len(df_prompts)} IOI-style prompts.")

    # Load tokenizers
    tokenizers = {}
    for short_name, hf_name in MODEL_SPECS.items():
        print(f"[INFO] Loading tokenizer for {short_name} ({hf_name})...")
        tok = AutoTokenizer.from_pretrained(hf_name)
        tokenizers[short_name] = tok

    rows = []
    for idx, row in df_prompts.iterrows():
        text = row["prompt"]
        for model_name, tok in tokenizers.items():
            # We do not add special tokens; we want the raw sequence
            ids = tok.encode(text, add_special_tokens=False)
            rows.append(
                {
                    "prompt_id": idx,
                    "A": row["A"],
                    "B": row["B"],
                    "prompt": text,
                    "model": model_name,
                    "n_tokens": len(ids),
                }
            )

    df_tokens = pd.DataFrame(rows)
    df_tokens.to_csv(OUT_CSV, index=False)
    print(f"[OUT] Wrote per-model tokenization stats to {OUT_CSV}")

    # Build a pivot table: rows = prompt_id, columns = model, values = n_tokens
    pivot = df_tokens.pivot_table(
        index=["prompt_id", "A", "B", "prompt"], columns="model", values="n_tokens"
    )

    # Per-model summary
    lines = []
    lines.append("=== Per-model token length statistics ===\n\n")
    for model in MODEL_SPECS.keys():
        if model not in pivot.columns:
            continue
        series = pivot[model]
        lines.append(f"Model: {model}\n")
        lines.append(f"  mean tokens: {series.mean():.2f}\n")
        lines.append(f"  std tokens:  {series.std():.2f}\n")
        lines.append(f"  min tokens:  {series.min():.0f}\n")
        lines.append(f"  max tokens:  {series.max():.0f}\n\n")

    # Pairwise mismatch rates (how often token lengths differ for same prompt)
    lines.append("=== Pairwise token length mismatch rates ===\n\n")
    models = list(MODEL_SPECS.keys())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            if m1 not in pivot.columns or m2 not in pivot.columns:
                continue
            s1 = pivot[m1]
            s2 = pivot[m2]
            mismatches = (s1 != s2).sum()
            total = len(pivot)
            frac_mismatch = mismatches / total if total > 0 else 0.0
            lines.append(f"{m1} vs {m2}:\n")
            lines.append(f"  mismatch count: {mismatches}/{total}\n")
            lines.append(f"  mismatch fraction: {frac_mismatch:.3f}\n\n")

    OUT_SUMMARY.write_text("".join(lines))
    print(f"[OUT] Wrote tokenization summary to {OUT_SUMMARY}")
    print("\n=== SUMMARY (also saved to file) ===")
    print("".join(lines))


if __name__ == "__main__":
    main()

