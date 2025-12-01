#!/usr/bin/env python
"""
Compare SAE feature similarity across model families (Pythia-410M vs GPT-2-medium).

High-level idea:
  1. Load two models with TransformerLens.
  2. Run a small IOI-style prompt set through each model.
  3. Collect activations at a chosen hook point (e.g., blocks.L.hook_resid_pre).
  4. Encode activations with a sparse autoencoder (SAE) for each model.
  5. For each feature in model A, find the best-matching feature in model B by
     cosine similarity over (prompt, position) activation patterns.
  6. Summarize the distribution of these max similarities.

This script is intentionally agnostic to the SAE implementation.
You MUST fill in `load_sae_for_model_layer` and `encode_with_sae` based on your SAE library.
"""

import argparse
import os
from typing import Tuple

import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer

DEVICE = "cpu"  # keep CPU for reproducibility / laptop sanity


# -----------------------------
# TODO: hook these into your SAE implementation
# -----------------------------

def load_sae_for_model_layer(model_name: str, layer: int, hook_name: str):
    """
    USER TODO:
      Replace this stub with code that loads a trained SAE for the given
      (model_name, layer, hook_name).

    Example pseudocode for sae_lens (NOT guaranteed correct):

        from sae_lens import SAE
        ckpt_path = f"sae_checkpoints/{model_name}_L{layer}_{hook_name}.pt"
        sae = SAE.load_from_checkpoint(ckpt_path, map_location=DEVICE)
        return sae

    For now, this returns None and will cause a runtime error if not overwritten.
    """
    raise NotImplementedError(
        "You must implement `load_sae_for_model_layer` to load your SAE."
    )


def encode_with_sae(sae, acts: torch.Tensor) -> torch.Tensor:
    """
    USER TODO:
      Given a batch of activations `acts` with shape [batch, seq, d_model],
      return SAE *codes* for each position, shape [batch, seq, n_features].

      How you do this depends on your SAE library.

    Example pseudocode:

        # if sae.forward returns (reconstruction, codes):
        recon, codes = sae(acts)
        return codes

        # or if sae has encode method:
        codes = sae.encode(acts)
        return codes
    """
    raise NotImplementedError(
        "You must implement `encode_with_sae` to use your SAE."
    )


# -----------------------------
# Core logic
# -----------------------------

def get_ioi_prompts(n: int = 64):
    """
    Very small hard-coded IOI-style prompts. For serious analysis, you should
    reuse your full IOI dataset, but this keeps the script self-contained.

    We just cycle over a small template list until we reach n prompts.
    """
    templates = [
        "When John met Mary, John greeted Mary.",
        "When Alice visited Bob, Alice thanked Bob.",
        "After Sarah called David, Sarah reminded David.",
        "Before Emma emailed Noah, Emma called Noah.",
    ]
    prompts = []
    i = 0
    while len(prompts) < n:
        prompts.append(templates[i % len(templates)])
        i += 1
    return prompts


def collect_acts(model: HookedTransformer, prompts, layer: int, hook_name: str) -> torch.Tensor:
    """
    Collect activations at `blocks.{layer}.{hook_name}` for all tokens of all prompts.
    Returns tensor of shape [total_tokens, d_model].
    """
    full_hook_name = f"blocks.{layer}.{hook_name}"

    all_acts = []

    def hook_fn(acts, hook):
        # acts shape: [batch, seq, d_model]
        all_acts.append(acts.detach().cpu())

    with model.hooks(fwd_hooks=[(full_hook_name, hook_fn)]):
        toks = model.to_tokens(prompts, prepend_bos=True).to(DEVICE)
        _ = model(toks)

    acts = torch.cat(all_acts, dim=0)  # [batch, seq, d_model]
    # Flatten batch and seq into one dimension: [N, d_model]
    acts = acts.reshape(-1, acts.shape[-1])
    return acts


def get_sae_codes_for_model(model_name: str, layer: int, hook_name: str, prompts) -> np.ndarray:
    """
    Load model + SAE, run prompts, and return SAE codes as a 2D array [N, n_features].
    """
    print(f"[INFO] Loading model {model_name} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(model_name, device=DEVICE)
    model.eval()

    print(f"[INFO] Loading SAE for {model_name}, layer {layer}, hook {hook_name}...")
    sae = load_sae_for_model_layer(model_name, layer, hook_name)

    print(f"[INFO] Collecting activations for {model_name}...")
    acts = collect_acts(model, prompts, layer, hook_name)  # [N, d_model]
    acts = acts.to(DEVICE)

    print(f"[INFO] Encoding activations with SAE for {model_name}...")
    codes = encode_with_sae(sae, acts)  # expected [N, n_features]
    if codes.dim() == 3:
        # In case SAE preserves batch/seq structure, flatten it
        codes = codes.reshape(-1, codes.shape[-1])

    return codes.detach().cpu().numpy()


def compute_feature_similarity(codes_a: np.ndarray, codes_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given SAE codes from two models:
      codes_a: [N, F_a]
      codes_b: [N, F_b]

    We compute cosine similarity between every feature in A and every feature in B
    over the N positions. Then, for each feature in A, we take the max over B
    (best-match similarity), and vice versa.

    Returns:
      best_a_to_b: [F_a] array of max sims for each feature in A
      best_b_to_a: [F_b] array of max sims for each feature in B
    """
    # Center features
    A = codes_a - codes_a.mean(axis=0, keepdims=True)
    B = codes_b - codes_b.mean(axis=0, keepdims=True)

    # Normalize features
    A_norm = np.linalg.norm(A, axis=0, keepdims=True) + 1e-8
    B_norm = np.linalg.norm(B, axis=0, keepdims=True) + 1e-8

    A_unit = A / A_norm  # [N, F_a]
    B_unit = B / B_norm  # [N, F_b]

    # Cosine similarity matrix: [F_a, F_b]
    sim = A_unit.T @ B_unit  # [F_a, F_b]

    best_a_to_b = sim.max(axis=1)  # [F_a]
    best_b_to_a = sim.max(axis=0)  # [F_b]
    return best_a_to_b, best_b_to_a


def summarize_similarity(name: str, sims: np.ndarray):
    sims = np.asarray(sims)
    print(f"== {name} ==")
    print(f"n      = {len(sims)}")
    print(f"mean   = {sims.mean():.4f}")
    print(f"std    = {sims.std():.4f}")
    print(f"p05    = {np.quantile(sims, 0.05):.4f}")
    print(f"p50    = {np.quantile(sims, 0.50):.4f}")
    print(f"p95    = {np.quantile(sims, 0.95):.4f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_name", type=str, default="pythia-410m")
    parser.add_argument("--model_b_name", type=str, default="gpt2-medium")
    parser.add_argument("--layer", type=int, default=10, help="Layer index to analyze")
    parser.add_argument("--hook_name", type=str, default="hook_resid_pre",
                        help="Hook name inside blocks.L.* to use (e.g. hook_resid_pre, hook_mlp_out)")
    parser.add_argument("--n_prompts", type=int, default=64)
    parser.add_argument("--output_path", type=str,
                        default="results/sae_feature_overlap_pythia-410m_gpt2-medium_layer10.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    prompts = get_ioi_prompts(n=args.n_prompts)

    codes_a = get_sae_codes_for_model(args.model_a_name, args.layer, args.hook_name, prompts)
    codes_b = get_sae_codes_for_model(args.model_b_name, args.layer, args.hook_name, prompts)

    print(f"[INFO] codes_a shape = {codes_a.shape}, codes_b shape = {codes_b.shape}")

    best_a_to_b, best_b_to_a = compute_feature_similarity(codes_a, codes_b)

    print("\n### SAE feature overlap summary ###\n")
    summarize_similarity(f"{args.model_a_name} → {args.model_b_name}", best_a_to_b)
    summarize_similarity(f"{args.model_b_name} → {args.model_a_name}", best_b_to_a)

    # Save summary to CSV for paper
    df = pd.DataFrame({
        "direction": [f"{args.model_a_name}_to_{args.model_b_name}",
                      f"{args.model_b_name}_to_{args.model_a_name}"],
        "mean": [best_a_to_b.mean(), best_b_to_a.mean()],
        "std": [best_a_to_b.std(), best_b_to_a.std()],
        "p05": [np.quantile(best_a_to_b, 0.05), np.quantile(best_b_to_a, 0.05)],
        "p50": [np.quantile(best_a_to_b, 0.50), np.quantile(best_b_to_a, 0.50)],
        "p95": [np.quantile(best_a_to_b, 0.95), np.quantile(best_b_to_a, 0.95)],
        "n_features": [len(best_a_to_b), len(best_b_to_a)],
        "layer": [args.layer, args.layer],
        "hook_name": [args.hook_name, args.hook_name],
    })
    df.to_csv(args.output_path, index=False)
    print(f"[INFO] Saved SAE feature overlap summary to {args.output_path}")


if __name__ == "__main__":
    main()

