#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from typing import List

import torch
from transformer_lens import HookedTransformer

# Device default
DEFAULT_DEVICE = "cpu"


@dataclass
class HeadPair:
    layer_a: int
    head_a: int
    layer_b: int
    head_b: int


def parse_pairs(arg: str) -> List[HeadPair]:
    """Parse 'la-ha-lb-hb;...' into HeadPair objects."""
    pairs = []
    for chunk in arg.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        la, ha, lb, hb = map(int, chunk.split("-"))
        pairs.append(HeadPair(la, ha, lb, hb))
    return pairs


def get_ioi_prompts() -> List[str]:
    """Small IOI prompt set."""
    subjects = ["Alice", "John", "Mary", "David"]
    objects = ["Berlin", "Paris", "Tokyo", "New York"]
    templates = [
        "{A} went to {X}. Then {B} went to {Y}. Who went to {X}?",
        "{A} and {B} visited {X} and {Y}. {A} went to {X}. Who went to {X}?",
        "{A} traveled from {X} to {Y}, while {B} stayed in {X}. Who stayed in {X}?",
    ]

    prompts = []
    for A in subjects:
        for B in subjects:
            if A == B:
                continue
            for X in objects:
                for Y in objects:
                    if X == Y:
                        continue
                    for tmpl in templates:
                        prompts.append(tmpl.format(A=A, B=B, X=X, Y=Y))

    return prompts[:64]


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Jensen–Shannon divergence."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return 0.5 * (kl_pm + kl_qm)


def collect_head_attention(model, layer, head, prompts, device):
    """Collect mean attention of (layer, head) at last token."""
    toks = model.to_tokens(prompts).to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            toks,
            names_filter=lambda name: name == f"blocks.{layer}.attn.hook_pattern"
        )

    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # [B, H, Q, K]
    pattern = pattern[:, head]                            # [B, Q, K]
    last = pattern[:, -1, :]                              # [B, K]
    mean_last = last.mean(dim=0)                          # [K]
    return mean_last, toks


def decode_top_tokens(model, toks, attn, k=10):
    """Return top-k attended tokens as strings."""
    idx = torch.topk(attn, k=min(k, attn.shape[0]))
    values, positions = idx.values, idx.indices
    out = []
    ref = toks[0]
    for w, pos in zip(values.tolist(), positions.tolist()):
        if pos >= ref.shape[0]:
            continue
        token_str = model.to_string(ref[pos].item())
        out.append(f"{w:.3f} -> pos {pos} = {repr(token_str)}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_name", required=True)
    parser.add_argument("--model_b_name", required=True)
    parser.add_argument("--pairs", required=True,
                        help="Format: '5-13-0-4;10-7-8-8'")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()

    device = args.device
    pairs = parse_pairs(args.pairs)

    print(f"[INFO] Loading models {args.model_a_name}, {args.model_b_name} on {device}...")
    model_a = HookedTransformer.from_pretrained(args.model_a_name, device=device)
    model_b = HookedTransformer.from_pretrained(args.model_b_name, device=device)

    prompts = get_ioi_prompts()
    print(f"[INFO] Using {len(prompts)} IOI prompts.")

    for hp in pairs:
        print()
        print(f"=== Case: A(L{hp.layer_a}H{hp.head_a}) vs B(L{hp.layer_b}H{hp.head_b}) ===")

        attn_a, toks_a = collect_head_attention(
            model_a, hp.layer_a, hp.head_a, prompts, device
        )
        attn_b, toks_b = collect_head_attention(
            model_b, hp.layer_b, hp.head_b, prompts, device
        )

        L = min(attn_a.shape[0], attn_b.shape[0])
        p = attn_a[:L]
        q = attn_b[:L]
        p = p / p.sum()
        q = q / q.sum()

        js = js_divergence(p, q).item()
        print(f"JS divergence: {js:.4f}")

        print("Top tokens A:")
        for x in decode_top_tokens(model_a, toks_a, p, k=10):
            print("  ", x)

        print("Top tokens B:")
        for x in decode_top_tokens(model_b, toks_b, q, k=10):
            print("  ", x)


if __name__ == "__main__":
    main()

