import os
import json
from typing import List, Tuple

import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer


# --------------------------
# IOI prompt construction
# --------------------------

NAMES = [
    "John",
    "Mary",
    "Alice",
    "Bob",
    "Tom",
    "Sarah",
    "David",
    "Emma",
    "James",
    "Olivia",
    "Michael",
    "Sophia",
    "Robert",
    "Isabella",
    "William",
    "Ava",
    "Joseph",
    "Mia",
    "Charles",
    "Emily",
]


def build_ioi_prompts_single_token(model: HookedTransformer) -> List[Tuple[str, str, str]]:
    """
    Build IOI prompts of the form:
      Name1 and Name2 went to the store. Name1 gave a book to Name2 because

    We keep only pairs where BOTH names are single-token under this model.

    Returns:
        List of (prompt, correct_name, wrong_name) tuples.
        correct_name = Name2 (indirect object)
        wrong_name   = Name1 (subject)
    """
    single_token_names = []
    for name in NAMES:
        try:
            _ = model.to_single_token(name)
            single_token_names.append(name)
        except AssertionError:
            # Not a single token; skip this name for GPT-2-Medium
            continue

    prompts: List[Tuple[str, str, str]] = []
    for name1 in single_token_names:
        for name2 in single_token_names:
            if name1 == name2:
                continue
            prompt = f"{name1} and {name2} went to the store. {name1} gave a book to {name2} because"
            # correct continuation is name2 (IOI), wrong is name1
            prompts.append((prompt, name2, name1))

    return prompts


# --------------------------
# IOI margin computation
# --------------------------

def compute_ioi_margins(
    logits: torch.Tensor,
    examples: List[Tuple[str, str, str]],
    model: HookedTransformer,
) -> np.ndarray:
    """
    Compute IOI margins for a batch of logits.

    Args:
        logits: [batch, seq, d_vocab]
        examples: list of (prompt, correct_name, wrong_name)
        model: HookedTransformer (for to_single_token)

    Returns:
        margins: np.ndarray of shape [batch], where
            margin = logit(correct_name) - logit(wrong_name)
            at the final position.
    """
    # logits for the next token at the final position of each prompt
    # shape: [batch, d_vocab]
    last_logits = logits[:, -1, :]

    margins = []
    for (prompt, correct_name, wrong_name), logit_vec in zip(examples, last_logits):
        correct_tok = model.to_single_token(correct_name)
        wrong_tok = model.to_single_token(wrong_name)
        margin = (logit_vec[correct_tok] - logit_vec[wrong_tok]).item()
        margins.append(margin)

    return np.array(margins, dtype=np.float64)


# --------------------------
# Head ablation hook (hook_z)
# --------------------------

def make_zero_head_hook(head_idx: int):
    """
    Returns a hook that zeros out a single attention head in hook_z.

    For hook_z, v has shape [batch, seq, n_heads, d_head].
    """
    def hook(v, hook):
        v = v.clone()
        v[:, :, head_idx, :] = 0.0
        return v

    return hook


# --------------------------
# Main analysis
# --------------------------

def main():
    hero_path = "results/ioi_hero_heads_gpt2-medium.csv"
    out_dir = "results/gpt2m_ioi_hero_analysis"

    if not os.path.exists(hero_path):
        raise FileNotFoundError(
            f"{hero_path} not found. Run select_ioi_heads_gpt2medium_from_global.py first."
        )

    os.makedirs(out_dir, exist_ok=True)

    # Load hero heads (top IOI heads from global scan)
    hero_df = pd.read_csv(hero_path)

    # Load model
    model_name = "gpt2-medium"
    print("`torch_dtype` is deprecated! Use `dtype` instead!")
    model = HookedTransformer.from_pretrained(model_name, device="cpu")

    # Build IOI prompts restricted to single-token names for this model
    examples = build_ioi_prompts_single_token(model)
    if not examples:
        raise RuntimeError("No IOI prompts with single-token names were constructed.")

    print(f"[INFO] Using {len(examples)} IOI prompts with single-token names.")

    # Tokenize all prompts once
    prompts = [p for (p, _, _) in examples]
    tokens = model.to_tokens(prompts)

    # Baseline logits
    with torch.no_grad():
        base_logits = model(tokens)

    base_margins = compute_ioi_margins(base_logits, examples, model)
    base_mean = float(base_margins.mean())
    print(f"[INFO] Baseline mean IOI margin (GPT-2-M): {base_mean:.4f}")

    # Analyze each hero head
    for _, row in hero_df.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])
        tag = row.get("tag", f"gpt2m_L{layer}H{head}")

        hook_name = f"blocks.{layer}.attn.hook_z"

        with torch.no_grad():
            abl_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, make_zero_head_hook(head))],
            )

        abl_margins = compute_ioi_margins(abl_logits, examples, model)
        abl_mean = float(abl_margins.mean())
        delta = abl_mean - base_mean

        summary = {
            "model": model_name,
            "layer": layer,
            "head": head,
            "tag": tag,
            "base_mean_margin": base_mean,
            "ablated_mean_margin": abl_mean,
            "delta_margin": delta,
            "n_prompts": len(examples),
        }

        out_name = f"gpt2-medium_L{layer}H{head}_summary.json"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"[HEAD] {tag}: delta_margin = {delta:.4f}, "
            f"ablated_mean = {abl_mean:.4f} "
            f"(saved {out_path})"
        )


if __name__ == "__main__":
    main()

