#!/usr/bin/env python
import os
from typing import List, Tuple, Dict

import torch
import pandas as pd
from transformer_lens import HookedTransformer

# -------------------- Device selection --------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

# -------------------- Config --------------------

MODEL_NAME = "gpt2-medium"
OUT_PATH = "results/global_ioi_circuit_gpt2medium.csv"

# --- IOI dataset (reuse same as compute_ioi_name_strengths_pythia_gpt2.py) ---

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
    prompts: List[str] = []
    pairs: List[Tuple[str, str]] = []
    for i, A in enumerate(names):
        for j, B in enumerate(names):
            if i == j:
                continue
            prompts.append(TEMPLATE.format(A=A, B=B))
            pairs.append((A, B))
    return prompts, pairs


# -------------------- Model helpers --------------------


def load_model() -> HookedTransformer:
    print(f"Loading model {MODEL_NAME} on {device}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    return model


def get_tokens(model: HookedTransformer, prompts: List[str]) -> torch.Tensor:
    # prepend_bos=True to match IOI setup
    return model.to_tokens(prompts, prepend_bos=True)


def last_token_id(model: HookedTransformer, text: str) -> int:
    toks = model.to_tokens(text, prepend_bos=False)
    return toks[0, -1].item()


# -------------------- IOI margin computation --------------------


def compute_ioi_margin_from_logits(
    logits: torch.Tensor,
    pairs: List[Tuple[str, str]],
    name_to_id: Dict[str, int],
    target_pos: int,
) -> torch.Tensor:
    """
    logits: [batch, seq, vocab]
    Returns IOI margins per example:
      margin = logit(correct_name_last_token) - logit(wrong_name_last_token)
    """
    logits_last = logits[:, target_pos, :]
    margins: List[torch.Tensor] = []
    for i, (A, B) in enumerate(pairs):
        logit_correct = logits_last[i, name_to_id[B]]
        logit_wrong = logits_last[i, name_to_id[A]]
        margins.append(logit_correct - logit_wrong)
    return torch.stack(margins)


# -------------------- Main scan --------------------


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    prompts, pairs = build_ioi_dataset(NAMES)
    print(f"Built IOI dataset with {len(prompts)} prompts.")

    model = load_model()
    tokens = get_tokens(model, prompts)
    target_pos = tokens.shape[1] - 1

    # Precompute name → vocab id once
    names = set(n for pair in pairs for n in pair)
    name_to_id = {name: last_token_id(model, name) for name in names}

    # Baseline margins
    with torch.no_grad():
        base_logits = model(tokens)
    base_margins = compute_ioi_margin_from_logits(
        base_logits, pairs, name_to_id, target_pos
    )
    base_mean = base_margins.mean().item()
    print(f"Baseline mean IOI margin ({MODEL_NAME}): {base_mean:.4f}")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    rows = []

    # --- Attention heads ---
    print("Scanning attention heads...")
    for layer in range(n_layers):
        for head in range(n_heads):
            hook_name = f"blocks.{layer}.attn.hook_z"

            def zero_head(z, hook, h=head):
                z = z.clone()
                z[:, :, h, :] = 0.0
                return z

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, zero_head)],
                )

            margins = compute_ioi_margin_from_logits(
                ablated_logits, pairs, name_to_id, target_pos
            )
            mean_margin = margins.mean().item()
            delta = mean_margin - base_mean  # ablated - base

            rows.append(
                {
                    "type": "attn",
                    "layer": layer,
                    "head": head,
                    "mean_margin_ablated": mean_margin,
                    "mean_margin_base": base_mean,
                    "delta_margin": delta,
                }
            )

    # --- MLP blocks ---
    print("Scanning MLP blocks...")
    for layer in range(n_layers):
        # NOTE: use 'hook_post' for MLP output in current TransformerLens
        hook_name = f"blocks.{layer}.mlp.hook_post"

        def zero_mlp(mlp_out, hook):
            return torch.zeros_like(mlp_out)

        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, zero_mlp)],
            )

        margins = compute_ioi_margin_from_logits(
            ablated_logits, pairs, name_to_id, target_pos
        )
        mean_margin = margins.mean().item()
        delta = mean_margin - base_mean

        rows.append(
            {
                "type": "mlp",
                "layer": layer,
                "head": -1,
                "mean_margin_ablated": mean_margin,
                "mean_margin_base": base_mean,
                "delta_margin": delta,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"[OUT] Saved global IOI circuit scan to {OUT_PATH}")

    print("Top 10 units by |delta_margin|:")
    print(
        df.reindex(df["delta_margin"].abs().sort_values(ascending=False).index)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

