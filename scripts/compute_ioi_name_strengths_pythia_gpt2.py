#!/usr/bin/env python
import os
from collections import defaultdict
from typing import List, Tuple, Dict

import torch
import pandas as pd
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

OUT_PATH = "results/ioi_name_strengths_pythia_gpt2.csv"

PYTHIA_MODEL_NAME = "EleutherAI/pythia-410m-deduped"
GPT2_MODEL_NAME = "gpt2-medium"


# -------------------------------------------------
# IOI-style prompt generator
# -------------------------------------------------

NAMES = [
    "John", "Mary",
    "Alice", "Bob",
    "Tom", "Sarah",
    "David", "Emma",
    "James", "Olivia",
    "Michael", "Sophia",
    "Robert", "Isabella",
    "William", "Ava",
    "Joseph", "Mia",
    "Charles", "Emily",
]

TEMPLATE = "{A} and {B} went to the store. {A} gave a book to {B} because "


def build_ioi_dataset(names: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Build a set of IOI prompts over all ordered pairs (A, B) with A != B.
    Returns:
      prompts: list of strings
      pairs:   list of (A, B) for each prompt
    """
    prompts = []
    pairs = []
    for i, A in enumerate(names):
        for j, B in enumerate(names):
            if i == j:
                continue
            prompts.append(TEMPLATE.format(A=A, B=B))
            pairs.append((A, B))
    return prompts, pairs


# -------------------------------------------------
# Model helpers
# -------------------------------------------------

def load_model(model_name: str) -> HookedTransformer:
    print(f"Loading model {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    return model


def get_tokens(model: HookedTransformer, prompts: List[str]) -> torch.Tensor:
    return model.to_tokens(prompts, prepend_bos=True)


def get_last_token_id(model: HookedTransformer, text: str) -> int:
    toks = model.to_tokens(text, prepend_bos=False)
    return toks[0, -1].item()


def compute_name_strengths_for_model(
    model_name: str,
    names: List[str],
    prompts: List[str],
    pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    """
    For a given model:
      - Compute Δ = logit(B) - logit(A) at the last position
      - For each prompt with names (A, B):
           contribution to A: -Δ  (we want positive when model *discourages* wrong A)
           contribution to B: +Δ  (positive when model *encourages* correct B)
      - For each name, average all contributions over prompts it appears in.
    """
    model = load_model(model_name)
    tokens = get_tokens(model, prompts)
    target_pos = tokens.shape[1] - 1  # last position

    # Precompute last-token IDs for all names
    name_to_id = {name: get_last_token_id(model, name) for name in names}

    with torch.no_grad():
        logits = model(tokens)  # [batch, seq, d_vocab]
    logits_last = logits[:, target_pos, :]

    # Build per-name accumulators
    num = defaultdict(float)
    count = defaultdict(int)

    for idx, (A, B) in enumerate(pairs):
        logit_A = logits_last[idx, name_to_id[A]]
        logit_B = logits_last[idx, name_to_id[B]]
        delta = (logit_B - logit_A).item()

        # Contribution for B (correct)
        num[B] += delta
        count[B] += 1

        # Contribution for A (incorrect)
        num[A] += -delta
        count[A] += 1

    strengths = {}
    for name in names:
        if count[name] > 0:
            strengths[name] = num[name] / count[name]
        else:
            strengths[name] = float("nan")

    return strengths


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    prompts, pairs = build_ioi_dataset(NAMES)
    print(f"Built IOI dataset with {len(prompts)} prompts over {len(NAMES)} names.")

    # Compute strengths for each model
    strengths_pythia = compute_name_strengths_for_model(
        PYTHIA_MODEL_NAME, NAMES, prompts, pairs
    )
    strengths_gpt2 = compute_name_strengths_for_model(
        GPT2_MODEL_NAME, NAMES, prompts, pairs
    )

    # Build DataFrame
    rows = []
    for name in NAMES:
        rows.append(
            {
                "name": name,
                "ioistrength_pythia": strengths_pythia[name],
                "ioistrength_gpt2": strengths_gpt2[name],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"[OUT] Saved IOI name strengths for both models to {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()

