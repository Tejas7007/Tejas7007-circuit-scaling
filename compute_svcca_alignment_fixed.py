#!/usr/bin/env python3
"""
compute_svcca_alignment_fixed.py

CKA-based alignment between two models (e.g., pythia-410m and gpt2-medium),
using IOI-style prompts.

Outputs:
  results/cka_alignment_<modelA>_<modelB>.csv
"""

import os
import sys
import argparse
from typing import List, Dict

import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer

# ============================================================
# Path setup: add repo root and scripts/ to sys.path
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))         # .../circuit-scaling/scripts
REPO_ROOT = os.path.dirname(THIS_DIR)                         # .../circuit-scaling
SCRIPTS_DIR = THIS_DIR                                        # explicitly

for p in [REPO_ROOT, SCRIPTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ============================================================
# IOI prompt loader - with multiple fallback imports
# ============================================================
def load_ioi_prompts(n_prompts: int, seed: int = 42) -> List[str]:
    """
    Load IOI-style prompts using make_ioi_prompts from ioi_dataset.py.
    Tries multiple import locations.
    """
    make_ioi_prompts = None
    
    # Try 1: from scripts directory (same dir as this file)
    try:
        sys.path.insert(0, SCRIPTS_DIR)
        from ioi_dataset import make_ioi_prompts
        print("[INFO] Loaded make_ioi_prompts from scripts/ioi_dataset.py")
    except ImportError:
        pass
    
    # Try 2: from repo root
    if make_ioi_prompts is None:
        try:
            sys.path.insert(0, REPO_ROOT)
            from ioi_dataset import make_ioi_prompts
            print("[INFO] Loaded make_ioi_prompts from repo root ioi_dataset.py")
        except ImportError:
            pass
    
    # Try 3: as scripts.ioi_dataset
    if make_ioi_prompts is None:
        try:
            from scripts.ioi_dataset import make_ioi_prompts
            print("[INFO] Loaded make_ioi_prompts from scripts.ioi_dataset")
        except ImportError:
            pass
    
    if make_ioi_prompts is None:
        raise ImportError(
            "Could not import make_ioi_prompts from ioi_dataset.\n"
            "Make sure ioi_dataset.py exists in either:\n"
            f"  - {SCRIPTS_DIR}/ioi_dataset.py\n"
            f"  - {REPO_ROOT}/ioi_dataset.py\n"
            "And that you're running from the repo root."
        )

    prompts, _ = make_ioi_prompts(n_prompts, seed=seed)
    return prompts


# ============================================================
# Torch CKA (no TensorFlow/SVCCA dependency)
# ============================================================
def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    X, Y: [examples, features]
    Returns scalar CKA value.
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    KX = X @ X.T
    KY = Y @ Y.T

    hsic = (KX * KY).sum()
    norm_x = torch.sqrt((KX * KX).sum())
    norm_y = torch.sqrt((KY * KY).sum())

    return (hsic / (norm_x * norm_y + 1e-8)).item()


# ============================================================
# Collect per-layer representations (batched)
# ============================================================
def collect_layer_reps(
    model: HookedTransformer,
    prompts: List[str],
    device: str,
    batch_size: int = 16,
) -> Dict[int, torch.Tensor]:
    """
    Returns dict: {layer -> [N, d_model]} where N = len(prompts).
    We average over sequence positions to get one vector per prompt.
    """
    tokens = model.to_tokens(prompts, prepend_bos=True).to(device)
    N, L = tokens.shape
    print(f"[INFO] Tokens shape for {model.cfg.model_name}: {tokens.shape}")

    layer_reps = {layer: [] for layer in range(model.cfg.n_layers)}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        toks = tokens[start:end]

        logits, cache = model.run_with_cache(toks, return_type="logits")

        for layer in range(model.cfg.n_layers):
            acts = cache[f"blocks.{layer}.hook_resid_post"]  # [b, seq, d]
            pooled = acts.mean(dim=1)                        # [b, d]
            layer_reps[layer].append(pooled.detach().cpu())

        # free cache as we go
        del cache
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    for layer in layer_reps:
        layer_reps[layer] = torch.cat(layer_reps[layer], dim=0)

    return layer_reps


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--n_prompts", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = args.device
    n_prompts = args.n_prompts

    print(f"[INFO] Using {n_prompts} IOI prompts.")
    prompts = load_ioi_prompts(n_prompts)

    # Load models
    print(f"[INFO] Loading model {args.model_a} on {device}...")
    A = HookedTransformer.from_pretrained(args.model_a, device=device)
    print(
        f"Loaded {args.model_a} with {A.cfg.n_layers} layers, d_model={A.cfg.d_model}"
    )

    print(f"[INFO] Loading model {args.model_b} on {device}...")
    B = HookedTransformer.from_pretrained(args.model_b, device=device)
    print(
        f"Loaded {args.model_b} with {B.cfg.n_layers} layers, d_model={B.cfg.d_model}"
    )

    # Collect representations
    print("[INFO] Collecting layer reps (model A)...")
    reps_A = collect_layer_reps(A, prompts, device, args.batch_size)

    print("[INFO] Collecting layer reps (model B)...")
    reps_B = collect_layer_reps(B, prompts, device, args.batch_size)

    # Sanity check: same #layers and same d_model
    assert A.cfg.n_layers == B.cfg.n_layers, "Models must have same #layers"
    assert A.cfg.d_model == B.cfg.d_model, "Models must have same d_model for this script"

    # CKA alignment per layer
    rows = []
    for layer in range(A.cfg.n_layers):
        X = reps_A[layer]
        Y = reps_B[layer]
        cka_val = compute_cka(X.float(), Y.float())
        rows.append({"layer": layer, "cka": cka_val})
        print(f"[LAYER {layer:02d}] CKA = {cka_val:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out_path = f"results/cka_alignment_{args.model_a}_{args.model_b}.csv"
    df.to_csv(out_path, index=False)
    print(f"[OUT] Wrote CKA alignment to {out_path}")


if __name__ == "__main__":
    main()
