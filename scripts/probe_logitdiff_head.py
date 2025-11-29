#!/usr/bin/env python
from __future__ import annotations
import argparse, torch
from circuitscaling.models import load_model
from circuitscaling.datasets import ioi_name_pairs

def last_token_ids(tok, names):
    return [tok.encode(" " + n)[-1] for n in names]

def build_batch(model, n, device):
    pairs = ioi_name_pairs(n)
    prompts = [f"When {A} and {B} went to the store, {A} gave a gift to" for (A,B) in pairs]
    toks = model.to_tokens(prompts, prepend_bos=True).to(device)
    tok = model.tokenizer
    aid = torch.tensor(last_token_ids(tok, [A for (A,_) in pairs]), device=device)
    bid = torch.tensor(last_token_ids(tok, [B for (_,B) in pairs]), device=device)
    return toks, aid, bid

def logit_diff_from_logits(logits, aid, bid):
    last = logits[:, -1, :]
    return (last[torch.arange(last.size(0)), bid] - last[torch.arange(last.size(0)), aid]).mean().item()

def zero_one_head(layer, head):
    def hook(z, hook):
        z = z.clone(); z[:, :, head, :] = 0.0; return z
    return hook, f"blocks.{layer}.attn.hook_z"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-410m")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    model = load_model(args.model, device=args.device)
    toks, aid, bid = build_batch(model, args.n, args.device)

    with torch.no_grad():
        base_logits = model(toks, return_type="logits")
    base = logit_diff_from_logits(base_logits, aid, bid)

    hook_fn, hook_name = zero_one_head(args.layer, args.head)
    with torch.no_grad():
        logits = model.run_with_hooks(toks, return_type="logits", fwd_hooks=[(hook_name, hook_fn)])
    ablated = logit_diff_from_logits(logits, aid, bid)

    print(f"L{args.layer}H{args.head}  base={base:.4f}  ablated={ablated:.4f}  Δ={ablated-base:+.4f}  (negative => suppressor)")
