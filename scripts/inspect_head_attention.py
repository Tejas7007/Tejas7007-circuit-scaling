#!/usr/bin/env python
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

# --------- Simple built-in IOI dataset (no external imports) ----------

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

def build_ioi_prompts(max_prompts: int = 128):
    prompts = []
    for i, A in enumerate(NAMES):
        for j, B in enumerate(NAMES):
            if i == j:
                continue
            prompts.append(TEMPLATE.format(A=A, B=B))
    return prompts[:max_prompts]

# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--k", type=int, default=None, help="(unused, kept for backward compat)")
    args = parser.parse_args()

    # Choose device: prefer mps if available, then cuda, else cpu
    if args.device is not None:
        device = args.device
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"[INFO] Loading model {args.model} on {device}")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    prompts = build_ioi_prompts(max_prompts=128)
    print(f"[INFO] Built {len(prompts)} IOI prompts.")

    # Batch all prompts at once
    tokens = model.to_tokens(prompts, prepend_bos=True).to(device)

    hook_name = f"blocks.{args.layer}.attn.hook_pattern"

    # run_with_cache returns (output, cache); we DON'T pass cache=...
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name == hook_name,
        return_type="logits",
    )

    attn = cache[hook_name]  # [batch, n_heads, q, k]

    # Take the specified head and average across prompts
    attn_head = attn[:, args.head, :, :]          # [batch, q, k]
    avg_attn = attn_head.mean(dim=0).detach().to("cpu").numpy()  # [q, k]

    plt.figure(figsize=(7, 5))
    im = plt.imshow(avg_attn, aspect="auto")
    plt.colorbar(im, label="Attention weight")
    plt.xlabel("Key / destination token position")
    plt.ylabel("Query / source token position")
    plt.title(f"{args.model} — L{args.layer}H{args.head} avg attention (IOI prompts)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=240)
    plt.close()

    print(f"[OUT] Saved attention map to {args.out}")

if __name__ == "__main__":
    main()

