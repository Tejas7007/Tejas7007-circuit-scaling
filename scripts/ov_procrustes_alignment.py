#!/usr/bin/env python
"""
OV Procrustes alignment for selected head pairs.
"""

import os
import torch
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from transformer_lens import HookedTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_w_ov(model, layer, head):
    W_V = model.W_V[layer][head]     # [d_model, d_head]
    W_O = model.W_O[layer][head]     # [d_head, d_model]
    return W_V @ W_O                 # [d_model, d_model]

def procrustes(A, B):
    A = A.detach().cpu().numpy().reshape(-1, 1)
    B = B.detach().cpu().numpy().reshape(-1, 1)
    R, _ = orthogonal_procrustes(A, B)
    A_aligned = A @ R
    return float(((A_aligned - B)**2).sum()**0.5)

def main(model_a_name, model_b_name, pair_str):
    os.makedirs("results", exist_ok=True)

    model_a = HookedTransformer.from_pretrained(model_a_name).to(DEVICE)
    model_b = HookedTransformer.from_pretrained(model_b_name).to(DEVICE)

    pairs = []
    for p in pair_str.split(";"):
        la, ha, lb, hb = map(int, p.split("-"))
        pairs.append((la, ha, lb, hb))

    rows = []

    for (la, ha, lb, hb) in pairs:
        W_a = get_w_ov(model_a, la, ha)
        W_b = get_w_ov(model_b, lb, hb)

        frob = torch.norm(W_a - W_b).item()
        norm_a = torch.norm(W_a).item()
        norm_b = torch.norm(W_b).item()
        proc = procrustes(W_a, W_b)

        rows.append({
            "layer_a": la,
            "head_a": ha,
            "layer_b": lb,
            "head_b": hb,
            "frobenius": frob,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "procrustes_error": proc,
        })

    df = pd.DataFrame(rows)
    out = f"results/ov_procrustes_{model_a_name.replace('/','_')}_{model_b_name.replace('/','_')}.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_name", type=str, required=True)
    parser.add_argument("--model_b_name", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True,
                        help="Format: '5-13-0-4;10-7-8-8'")
    args = parser.parse_args()
    main(args.model_a_name, args.model_b_name, args.pairs)

