#!/usr/bin/env python3
"""
Compute per-layer SVCCA similarity between Pythia-410m and GPT2-Medium.
We use LN2-normalized residual stream (hook_resid_post) as the representation,
which is compatible across both families.

Output: results/svcca_pythia410m_gpt2medium.csv
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer

DEVICE = "cpu"
N_PROMPTS = 50
SEQ_LEN = 40

# Prompts: simple synthetic sentences to avoid token mismatch & ensure same seq length
def build_prompts(n=50):
    base = "Alice talks to Bob about "
    objs = [
        "music", "sports", "physics", "politics", "movies", "books", "travel",
        "food", "history", "math", "technology", "art", "biology", "chemistry",
        "economics"
    ]
    prompts = []
    for i in range(n):
        prompts.append(base + objs[i % len(objs)])
    return prompts


def collect_layer_representations(model, prompts):
    """
    Collect LN2-normalized residual stream: blocks.L.hook_resid_post
    Shape per prompt: [seq, d_model]
    Returns: dict layer -> np.array (N_prompts, seq*d_model)
    """
    outputs = {}
    for L in range(model.cfg.n_layers):
        hook_name = f"blocks.{L}.hook_resid_post"
        outputs[L] = []

        def hook_fn(tensor, hook):
            outputs[L].append(tensor[0].detach().cpu().numpy())  # (seq, d_model)

        for prompt in prompts:
            toks = model.to_tokens(prompt).to(DEVICE)
            model.run_with_hooks(
                toks.unsqueeze(0),
                fwd_hooks=[(hook_name, hook_fn)],
                return_type=None
            )

    # flatten per prompt into shape: N_prompts × (seq*d_model)
    final = {}
    for L in outputs:
        arr = np.stack([x.reshape(-1) for x in outputs[L]], axis=0)
        final[L] = arr
    return final


def compute_svd(x):
    """Utility: center and SVD."""
    x = x - x.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(x, full_matrices=False)
    return U, S, Vt


def svcca(X, Y, k=50):
    """
    X: (n, d)
    Y: (n, d)
    Perform Smith et al. 2017 CCA via truncated SVD.
    Return mean canonical correlation (SVCCA score).
    """
    Ux, Sx, Vx = compute_svd(X)
    Uy, Sy, Vy = compute_svd(Y)

    # project into top-k
    Xk = (Ux[:, :k] * Sx[:k])
    Yk = (Uy[:, :k] * Sy[:k])

    # CCA
    Cxx = Xk.T @ Xk
    Cyy = Yk.T @ Yk
    Cxy = Xk.T @ Yk

    # whiten
    eps = 1e-5
    Cxx_inv = np.linalg.inv(Cxx + eps * np.eye(k))
    Cyy_inv = np.linalg.inv(Cyy + eps * np.eye(k))

    M = np.linalg.sqrtm(Cxx_inv) @ Cxy @ np.linalg.sqrtm(Cyy_inv)
    corr = np.linalg.svd(M, compute_uv=False)
    return float(corr.mean())


def main():
    print("[INFO] Loading models...")
    A = HookedTransformer.from_pretrained("pythia-410m", device=DEVICE)
    B = HookedTransformer.from_pretrained("gpt2-medium", device=DEVICE)

    prompts = build_prompts(N_PROMPTS)
    print("[INFO] Collecting layer representations (Pythia)...")
    rep_A = collect_layer_representations(A, prompts)
    print("[INFO] Collecting layer representations (GPT2)...")
    rep_B = collect_layer_representations(B, prompts)

    rows = []

    for L in tqdm(range(A.cfg.n_layers), desc="SVCCA per layer"):
        XA = rep_A[L]
        YB = rep_B[L]

        # Ensure equal number of samples
        n = min(XA.shape[0], YB.shape[0])
        XA = XA[:n]
        YB = YB[:n]

        score = svcca(XA, YB, k=min(50, min(XA.shape[1], YB.shape[1])))
        rows.append({"layer": L, "svcca": score})

    df = pd.DataFrame(rows)
    df.to_csv("results/svcca_pythia410m_gpt2medium.csv", index=False)
    print("[OUT] Wrote SVCCA scores to results/svcca_pythia410m_gpt2medium.csv")
    print(df)


if __name__ == "__main__":
    main()

