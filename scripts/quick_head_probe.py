#!/usr/bin/env python
from __future__ import annotations
import argparse, torch
from circuitscaling.models import load_model

def tok_last_id(tok, s): return tok.encode(" " + s)[-1]

def logit_diff(model, prompts, correct_id, distractor_id, device="cpu"):
    toks = model.to_tokens(prompts, prepend_bos=True).to(device)
    with torch.no_grad():
        logits = model(toks, return_type="logits")  # [B,T,V]
    last = logits[:, -1, :]                        # position AFTER the prompt
    return (last[:, correct_id] - last[:, distractor_id]).mean().item()

def zero_one_head(layer, head):
    def hook(z, _):
        z = z.clone(); z[:, :, head, :] = 0.0; return z
    return hook, f"blocks.{layer}.attn.hook_result"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    model = load_model(args.model, device=args.device)
    tok = model.tokenizer

    # IOI prompt ends **right before** the name we care about (no trailing space here)
    nameA, nameB = "Alice", "Bob"
    prompts = [f"{nameA} talked to {nameB} because {nameA.lower()} thought" for _ in range(args.n)]

    idA, idB = tok_last_id(tok, nameA), tok_last_id(tok, nameB)
    base = logit_diff(model, prompts, idA, idB, device=args.device)

    hook_fn, hook_name = zero_one_head(args.layer, args.head)
    toks = model.to_tokens(prompts, prepend_bos=True).to(args.device)
    with torch.no_grad():
        logits = model.run_with_hooks(toks, return_type="logits", fwd_hooks=[(hook_name, hook_fn)])
    last = logits[:, -1, :]
    ablated = (last[:, idA] - last[:, idB]).mean().item()

    print(f"L{args.layer}H{args.head}  base={base:.4f}  ablated={ablated:.4f}  Δ={ablated-base:+.4f}")
