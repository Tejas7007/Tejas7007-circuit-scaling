#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json, math
import torch
from circuitscaling.models import load_model
from circuitscaling.datasets import ioi_prompts, ioi_name_pairs

def last_token_ids(tok, names):
    return [tok.encode(" " + n)[-1] for n in names]

def build_batch(model, n, device):
    pairs = ioi_name_pairs(n)
    prompts = [f"When {A} and {B} went to the store, {A} gave a gift to" for (A,B) in pairs]
    toks = model.to_tokens(prompts, prepend_bos=True).to(device)
    tok = model.tokenizer
    aid = torch.tensor(last_token_ids(tok, [A for (A,_) in pairs]), device=device, dtype=torch.long)
    bid = torch.tensor(last_token_ids(tok, [B for (_,B) in pairs]), device=device, dtype=torch.long)
    return toks, aid, bid

def logit_diff_from_logits(logits, aid, bid):
    last = logits[:, -1, :]
    return (last[torch.arange(last.size(0)), bid] - last[torch.arange(last.size(0)), aid]).mean().item()

def zero_one_head(layer, head):
    def hook(z, hook):
        z = z.clone()
        z[:, :, head, :] = 0.0
        return z
    return hook, f"blocks.{layer}.attn.hook_z"

def scan_heads_logitdiff(model_name:str, device:str, n_prompts:int, max_layers:int|None):
    model = load_model(model_name, device=device)
    toks, aid, bid = build_batch(model, n_prompts, device)
    with torch.no_grad():
        base_logits = model(toks, return_type="logits")
    base_ld = logit_diff_from_logits(base_logits, aid, bid)

    n_layers = model.cfg.n_layers if max_layers is None else min(max_layers, model.cfg.n_layers)
    n_heads = model.cfg.n_heads
    results = []
    for L in range(n_layers):
        for H in range(n_heads):
            hook_fn, hook_name = zero_one_head(L, H)
            with torch.no_grad():
                logits = model.run_with_hooks(toks, return_type="logits", fwd_hooks=[(hook_name, hook_fn)])
            ablated_ld = logit_diff_from_logits(logits, aid, bid)
            delta = ablated_ld - base_ld
            results.append(((L,H), float(delta)))
    return base_ld, results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['pythia-410m'])
    ap.add_argument('--n_prompts', type=int, default=200)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--max_layers', type=int, default=None)
    ap.add_argument('--topk', type=int, default=15)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    out = {}
    for m in args.models:
        print(f"==> Scanning (logit-diff) for {m} ...")
        base_ld, scores = scan_heads_logitdiff(m, args.device, args.n_prompts, args.max_layers)
        clean = [((L,H), d) for ((L,H), d) in scores if not math.isnan(d)]
        neg_sorted = sorted(clean, key=lambda x: x[1])[:args.topk]
        pos_sorted = sorted(clean, key=lambda x: x[1], reverse=True)[:args.topk]
        out[m] = {
            "n_prompts": args.n_prompts,
            "base_logit_diff_mean": base_ld,
            "top_suppressor_candidates": [{"layer":L,"head":H,"delta_logit_diff":d} for ((L,H),d) in neg_sorted],
            "top_anti_suppressors": [{"layer":L,"head":H,"delta_logit_diff":d} for ((L,H),d) in pos_sorted],
            "note": "Δ(logit-diff) = (ablated − base). Negative ⇒ head helps suppression."
        }
    with open("results/copy_suppression_logitdiff_scan.json","w") as f:
        json.dump(out, f, indent=2)
    print("Wrote results/copy_suppression_logitdiff_scan.json")

if __name__ == '__main__':
    main()
