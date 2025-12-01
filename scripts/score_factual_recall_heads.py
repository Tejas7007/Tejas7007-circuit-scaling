#!/usr/bin/env python
"""
Score per-head influence on factual recall prompts.

Outputs:
  results/factual_recall_head_scores_{model}.csv
"""

import json
import os
from typing import List, Dict

import torch
import pandas as pd
from transformer_lens import HookedTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_prompts(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def get_first_token_id(model: HookedTransformer, text: str) -> int:
    """
    Return the ID of the FIRST token of `text` (ignoring BOS).
    We no longer require that the text be single-token.
    """
    toks = model.to_tokens(text, prepend_bos=False)[0]
    return int(toks[0].item())


def compute_factual_margin(
    model: HookedTransformer,
    prompts: List[Dict],
    head_mask: torch.Tensor = None,
) -> float:
    """
    head_mask: None or tensor [n_layers, n_heads] with 0/1 entries.
               If provided, heads with 0 are zeroed in hook_z for each layer.
    """
    margins = []
    n_layers = model.cfg.n_layers

    # Build per-layer hooks once per call
    if head_mask is not None:
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(value, hook, layer_idx=layer_idx):
                # value: [batch, seq, n_heads, d_head]
                mask = head_mask[layer_idx].view(1, 1, -1, 1)
                return value * mask
            return hook_fn

        for layer_idx in range(n_layers):
            hook_name = f"blocks.{layer_idx}.attn.hook_z"
            hooks.append((hook_name, make_hook(layer_idx)))
    else:
        hooks = None

    for ex in prompts:
        prompt = ex["prompt"]
        correct = ex["correct_object"]
        distractors = ex["distractors"]

        with torch.no_grad():
            toks = model.to_tokens(prompt).to(DEVICE)
            if hooks is None:
                logits = model(toks)
            else:
                logits = model.run_with_hooks(
                    toks,
                    fwd_hooks=hooks,
                )
            last_logits = logits[0, -1]  # [vocab_size]

        correct_id = get_first_token_id(model, correct)
        distractor_ids = [get_first_token_id(model, d) for d in distractors]

        correct_logit = last_logits[correct_id]
        distractor_logits = last_logits[distractor_ids]
        margin = float(correct_logit - distractor_logits.mean())
        margins.append(margin)

    return float(torch.tensor(margins).mean().item())


def main(
    model_name: str,
    prompts_path: str = "data/factual_recall_prompts.jsonl",
    output_path: str = None,
):
    os.makedirs("results", exist_ok=True)

    if output_path is None:
        safe_name = model_name.replace("/", "_")
        output_path = f"results/factual_recall_head_scores_{safe_name}.csv"

    print(f"[INFO] Loading model {model_name} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(model_name).to(DEVICE)
    prompts = load_prompts(prompts_path)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    print("[INFO] Computing baseline factual recall margin...")
    baseline = compute_factual_margin(model, prompts, head_mask=None)
    print(f"[INFO] Baseline margin: {baseline:.4f}")

    base_mask = torch.ones(n_layers, n_heads, device=DEVICE)
    rows = []

    for layer in range(n_layers):
        for head in range(n_heads):
            mask = base_mask.clone()
            mask[layer, head] = 0.0

            ablated = compute_factual_margin(model, prompts, head_mask=mask)
            delta = ablated - baseline

            rows.append({
                "model": model_name,
                "layer": layer,
                "head": head,
                "delta_margin": delta,
                "baseline_margin": baseline,
                "ablated_margin": ablated,
            })

            print(f"[L{layer}H{head}] Δmargin={delta:.5f}")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved head scores to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help='e.g. "pythia-410m", "gpt2-medium" (TransformerLens names)')
    parser.add_argument("--prompts_path", type=str,
                        default="data/factual_recall_prompts.jsonl")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    main(args.model_name, args.prompts_path, args.output_path)

