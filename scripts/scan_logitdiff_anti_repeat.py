import argparse
import json
import os
import random

import torch
from transformer_lens import HookedTransformer

# -----------------------
# 0. Model name resolution
# -----------------------


def resolve_model_name(model_short: str) -> str:
    """
    Map our short model argument (e.g. 'gpt2', 'gpt2-medium', 'pythia-70m')
    to the official name that HookedTransformer.from_pretrained expects.
    """
    # Pythia: use EleutherAI and -deduped
    if model_short.startswith("pythia"):
        # If user already passed a full HF id, just return it
        if model_short.startswith("EleutherAI/"):
            return model_short
        # If it already has -deduped, just add EleutherAI/
        if "-deduped" in model_short:
            return f"EleutherAI/{model_short}"
        # Default: EleutherAI/<model>-deduped
        return f"EleutherAI/{model_short}-deduped"

    # GPT-2 family: transformer-lens already knows these names directly
    if model_short in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2"}:
        return model_short

    # Anything else: pass through unchanged
    return model_short


# -----------------------
# 1. Prompt construction
# -----------------------

NAMES = [
    " Alice",
    " Bob",
    " Carol",
    " Dave",
    " Erin",
    " Frank",
    " Grace",
    " Heidi",
    " Ivan",
    " Judy",
    " Mallory",
    " Oscar",
]


def make_name_pairs(n_prompts: int):
    pairs = []
    for _ in range(n_prompts):
        a, b = random.sample(NAMES, 2)
        pairs.append((a, b))
    return pairs


def build_prompts(model: HookedTransformer, n_prompts: int, device: str):
    """
    Build a batch of prompts of the form:
        "<A> met <B>. <A> talked to"
    where the correct next token is <B> and the distractor is <A> (anti-repeat).

    Returns:
        tokens:    LongTensor [batch, seq_len]
        corr_ids:  LongTensor [batch]
        dist_ids:  LongTensor [batch]
    """
    pairs = make_name_pairs(n_prompts)
    prompts = []
    corr_ids = []
    dist_ids = []

    for a, b in pairs:
        text = f"{a} met{b}. {a} talked to"
        prompts.append(text)

        # Get token IDs for the names (use last token of the name string)
        a_toks = model.to_tokens(a, prepend_bos=False)[0]
        b_toks = model.to_tokens(b, prepend_bos=False)[0]
        dist_ids.append(int(a_toks[-1]))
        corr_ids.append(int(b_toks[-1]))

    # Tokenize prompts and pad to max length
    tok_list = [model.to_tokens(p, prepend_bos=True)[0] for p in prompts]
    max_len = max(t.shape[0] for t in tok_list)
    batch = torch.full(
        (len(tok_list), max_len),
        fill_value=model.tokenizer.pad_token_id
        if model.tokenizer.pad_token_id is not None
        else model.tokenizer.eos_token_id,
        dtype=torch.long,
    )
    for i, t in enumerate(tok_list):
        batch[i, : t.shape[0]] = t

    return (
        batch.to(device),
        torch.tensor(corr_ids, dtype=torch.long, device=device),
        torch.tensor(dist_ids, dtype=torch.long, device=device),
    )


# -----------------------
# 2. Metrics
# -----------------------


def compute_logit_diff(logits: torch.Tensor, corr_ids: torch.Tensor, dist_ids: torch.Tensor):
    """
    logits:   [batch, seq, vocab]
    corr_ids: [batch]
    dist_ids: [batch]
    Return mean(logit_corr - logit_dist) over batch, using last position.
    """
    last_logits = logits[:, -1, :]  # [batch, vocab]
    corr = last_logits.gather(-1, corr_ids.unsqueeze(-1)).squeeze(-1)
    dist = last_logits.gather(-1, dist_ids.unsqueeze(-1)).squeeze(-1)
    return (corr - dist).mean().item()


def zero_one_head_z(z, hook, head_idx: int):
    """
    z: [batch, seq, n_heads, d_head]
    Zero out a single head.
    """
    z = z.clone()
    z[..., head_idx, :] = 0.0
    return z


def scan_heads_logitdiff(model: HookedTransformer, device: str, n_prompts: int, topk: int = 20):
    """
    For the anti-repeat task:
      1) Compute base logit-diff.
      2) For each head, ablate and recompute logit-diff.
      3) Record Δ(logit-diff) = ablated - base.
    """
    model.to(device)

    # Build prompts & IDs once
    toks, corr_ids, dist_ids = build_prompts(model, n_prompts, device=device)

    with torch.no_grad():
        base_logits = model(toks, return_type="logits")
    base_ld = compute_logit_diff(base_logits, corr_ids, dist_ids)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    deltas = []

    for layer in range(n_layers):
        for head in range(n_heads):
            hook_name = f"blocks.{layer}.attn.hook_z"

            def hook_fn(z, hook, head_idx=head):
                return zero_one_head_z(z, hook, head_idx)

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    toks,
                    return_type="logits",
                    fwd_hooks=[(hook_name, hook_fn)],
                )

            ablated_ld = compute_logit_diff(ablated_logits, corr_ids, dist_ids)
            delta = ablated_ld - base_ld
            deltas.append(
                {
                    "layer": layer,
                    "head": head,
                    "delta": float(delta),  # NOTE: unified key name "delta"
                }
            )

    # Sort by delta (most negative = strongest suppressors)
    deltas_sorted = sorted(deltas, key=lambda h: h["delta"])

    top_suppressors = deltas_sorted[:topk]
    top_anti = sorted(deltas_sorted, key=lambda h: -h["delta"])[:topk]

    return {
        "n_prompts": n_prompts,
        "base_logit_diff_mean": base_ld,
        "heads": deltas,  # all heads (for joint analysis)
        "top_suppressor_candidates": top_suppressors,
        "top_anti_suppressors": top_anti,
        "note": "Δ(logit-diff) = (ablated − base). Negative ⇒ head helps anti-repeat suppression.",
    }


# -----------------------
# 3. CLI
# -----------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="e.g. pythia-70m, gpt2, gpt2-medium")
    parser.add_argument("--n_prompts", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    full_name = resolve_model_name(args.model)
    print(f"==> Scanning anti-repeat (logit-diff) for {full_name} on {args.device} ...")
    model = HookedTransformer.from_pretrained(full_name, device=args.device)

    results = scan_heads_logitdiff(model, args.device, args.n_prompts, topk=args.topk)
    results["model_name"] = args.model
    results["hf_name"] = full_name

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "anti_repeat_logitdiff_scan.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out_path}")
    print(
        f"base_logit_diff_mean={results['base_logit_diff_mean']:.4f}  "
        f"(Δ<0 ⇒ suppressor)"
    )
    print("\nTop suppressor candidates (layer, head, Δ):")
    for h in results["top_suppressor_candidates"][:5]:
        print(f"  L{h['layer']}H{h['head']}  Δ={h['delta']:.4f}")


if __name__ == "__main__":
    main()

