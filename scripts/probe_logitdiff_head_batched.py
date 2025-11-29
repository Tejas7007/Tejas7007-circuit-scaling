#!/usr/bin/env python
from __future__ import annotations
import argparse, torch
from circuitscaling.models import load_model
from circuitscaling.datasets import ioi_name_pairs

def last_ids(tok, names):
    return [tok.encode(" "+n)[-1] for n in names]

def make_prompts(n):
    pairs = ioi_name_pairs(n)
    prompts = [f"When {A} and {B} went to the store, {A} gave a gift to" for (A,B) in pairs]
    A = [A for (A,_) in pairs]; B = [B for (_,B) in pairs]
    return prompts, A, B

def logitdiff_from_logits(logits, aid, bid):
    last = logits[:, -1, :]
    rng = torch.arange(last.size(0), device=last.device)
    return (last[rng, bid] - last[rng, aid]).mean().item()

def zero_one_head(layer, head):
    def hook(z, hook): z = z.clone(); z[:, :, head, :] = 0.0; return z
    return hook, f"blocks.{layer}.attn.hook_z"

def run_batched(model, prompts, A, B, batch_size, device, layer=None, head=None):
    tok = model.tokenizer
    def enc(names):
        return torch.tensor([tok.encode(" "+n)[-1] for n in names], device=device, dtype=torch.long)

    base_sum = 0.0; abl_sum = 0.0; count = 0
    hook = None; hook_name = None
    if layer is not None and head is not None:
        hook, hook_name = zero_one_head(layer, head)

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i+batch_size]
        a_ids = enc(A[i:i+batch_size]); b_ids = enc(B[i:i+batch_size])
        toks = model.to_tokens(chunk, prepend_bos=True).to(device)

        with torch.no_grad():
            base_logits = model(toks, return_type="logits")
        base_sum += logitdiff_from_logits(base_logits, a_ids, b_ids) * len(chunk)

        if hook is not None:
            with torch.no_grad():
                abl_logits = model.run_with_hooks(toks, return_type="logits", fwd_hooks=[(hook_name, hook)])
            abl_sum += logitdiff_from_logits(abl_logits, a_ids, b_ids) * len(chunk)

        count += len(chunk)

    base = base_sum / count
    ablated = (abl_sum / count) if hook is not None else None
    return base, ablated

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=200)
    args = ap.parse_args()

    model = load_model(args.model, device=args.device)
    prompts, A, B = make_prompts(args.n)
    base, ablated = run_batched(model, prompts, A, B, args.batch_size, args.device, args.layer, args.head)
    delta = ablated - base
    print(f"L{args.layer}H{args.head}  base={base:.4f}  ablated={ablated:.4f}  Δ={delta:+.4f}  (Δ<0 ⇒ suppressor)")
