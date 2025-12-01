#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

###############################
# CKA helpers
###############################
def gram_linear(x):
    return x @ x.T

def cka(X, Y):
    K = gram_linear(X)
    L = gram_linear(Y)

    H = np.eye(K.shape[0]) - np.ones((K.shape[0], K.shape[0])) / K.shape[0]
    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic = np.sum(Kc * Lc)
    norm = np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc))
    if norm == 0:
        return 0.0
    return hsic / norm

###############################
# Collect OV
###############################
def collect_OV(model, prompts):
    outputs = {(l, h): [] for l in range(model.cfg.n_layers)
                           for h in range(model.cfg.n_heads)}

    print("[INFO] Collecting OV outputs...")
    for prompt in tqdm(prompts):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

        for layer in range(model.cfg.n_layers):
            v = cache[f"blocks.{layer}.attn.hook_v"]    # (seq, n_heads, d_head)
            for head in range(model.cfg.n_heads):
                OV = v[:, head, :]                      # (seq, d_head)
                OV = OV.reshape(-1, OV.shape[-1])       # flatten seq
                outputs[(layer, head)].append(OV.cpu().numpy())

    # stack each (layer, head)
    mats = {}
    for k, v in outputs.items():
        mats[k] = np.concatenate(v, axis=0)
    return mats

###############################
# Main
###############################
def main():
    print("[INFO] Loading models...")
    modelA = HookedTransformer.from_pretrained("pythia-410m", device="cpu")
    modelB = HookedTransformer.from_pretrained("gpt2-medium", device="cpu")

    prompts = [
        "John and Mary went to the store. John gave a book to Mary because",
        "Alice and Bob walked to school. Bob gave a pencil to Alice because",
        "Tom and Sarah visited the museum. Tom handed a ticket to Sarah because",
        "James and Emily met at the park. Emily passed the ball to James because",
        "Kevin and Laura traveled together. Kevin lent the charger to Laura because"
    ]

    OV_A = collect_OV(modelA, prompts)
    OV_B = collect_OV(modelB, prompts)

    print("[INFO] Aligning and computing CKA...")
    results = []
    for (layer, head), A in OV_A.items():
        B = OV_B[(layer, head)]

        n = min(A.shape[0], B.shape[0])
        A_use, B_use = A[:n], B[:n]

        score = cka(A_use, B_use)
        results.append([layer, head, score])

    df = pd.DataFrame(results, columns=["layer", "head", "cka"])
    out_path = "results/cka_alignment_pythia410m_gpt2medium.csv"
    df.to_csv(out_path, index=False)

    print(f"[OUT] Wrote CKA alignment results to {out_path}")
    print("\n=== TOP 20 HEADS ===")
    print(df.sort_values("cka", ascending=False).head(20))

if __name__ == "__main__":
    main()

