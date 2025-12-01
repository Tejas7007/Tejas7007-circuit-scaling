#!/usr/bin/env python
"""
Compute random-head OV CKA baseline between two models.
"""

import os
import numpy as np
import torch
from transformer_lens import HookedTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def linear_cka(X, Y):
    """
    X, Y: [N, d]
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xsq = (X * X).sum()
    ysq = (Y * Y).sum()
    dot = (X * Y).sum()
    return float((dot**2 / (xsq * ysq + 1e-8)).item())


def collect_ov(model, prompts, layer, head):
    """
    Collect OV outputs for a single head across all prompts.

    Returns [N, d_model] where N = sum over (batch * seq_len).
    """
    reps = []

    def hook_fn(z, hook):
        # z: [batch, seq, n_heads, d_head]
        W_O = model.W_O[layer]            # [n_heads, d_head, d_model]
        z_head = z[:, :, head, :]         # [batch, seq, d_head]
        ov = torch.einsum("bsd,dm->bsm", z_head, W_O[head])  # [b, s, d_model]
        flat = ov.reshape(-1, ov.shape[-1])                  # [b*s, d_model]
        reps.append(flat.detach().cpu())
        return z

    model.reset_hooks()
    model.add_hook(f"blocks.{layer}.attn.hook_z", hook_fn)

    with torch.no_grad():
        for p in prompts:
            toks = model.to_tokens(p).to(DEVICE)
            _ = model(toks)

    model.reset_hooks()

    if not reps:
        raise RuntimeError("No OV reps collected")

    reps = torch.cat(reps, dim=0)  # [N, d_model]
    return reps


def main(model_a_name, model_b_name, n_pairs=500):
    os.makedirs("results", exist_ok=True)

    print(f"[INFO] Loading models {model_a_name}, {model_b_name}")
    model_a = HookedTransformer.from_pretrained(model_a_name).to(DEVICE)
    model_b = HookedTransformer.from_pretrained(model_b_name).to(DEVICE)

    prompts = [
        "The Eiffel Tower is located in",
        "Marie Curie was born in",
        "Albert Einstein was born in",
        "The capital of France is",
        "The largest ocean on Earth is the",
        "Isaac Newton was born in",
    ]

    n_layers = min(model_a.cfg.n_layers, model_b.cfg.n_layers)
    n_heads = min(model_a.cfg.n_heads, model_b.cfg.n_heads)

    rng = np.random.default_rng(42)
    cka_vals = []

    for i in range(n_pairs):
        la = rng.integers(0, n_layers)
        ha = rng.integers(0, n_heads)
        lb = rng.integers(0, n_layers)
        hb = rng.integers(0, n_heads)

        print(f"[{i+1}/{n_pairs}] A(L{la}H{ha}) vs B(L{lb}H{hb})")

        X = collect_ov(model_a, prompts, la, ha)
        Y = collect_ov(model_b, prompts, lb, hb)

        n = min(len(X), len(Y))
        Xn = X[:n]
        Yn = Y[:n]

        cka = linear_cka(Xn, Yn)
        cka_vals.append(cka)

    a = model_a_name.replace("/", "_")
    b = model_b_name.replace("/", "_")
    out_path = f"results/cka_random_baseline_{a}_{b}.npz"
    np.savez(out_path, cka_vals=np.array(cka_vals))

    print(f"[INFO] Saved random CKA baseline to {out_path}")
    print(f"[INFO] Mean CKA={np.mean(cka_vals):.4f}, std={np.std(cka_vals):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_name", type=str, required=True)
    parser.add_argument("--model_b_name", type=str, required=True)
    parser.add_argument("--n_pairs", type=int, default=500)
    args = parser.parse_args()
    main(args.model_a_name, args.model_b_name, args.n_pairs)

