#!/usr/bin/env python
import os
import argparse
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------
# Simple IOI prompt set
# ------------------------

NAMES = [
    ("John", "Mary"),
    ("Alice", "Bob"),
    ("Tom", "Sarah"),
    ("David", "Emma"),
]

TEMPLATE = "{A} and {B} went to the store. {A} gave a book to {B} because "


def build_ioi_prompts() -> Tuple[List[str], List[str], List[str]]:
    """
    Build a tiny IOI-like prompt dataset:
    prompts: "<A> and <B> ... <A> gave a book to <B> because "
    correct: B
    wrong:   A
    """
    prompts = []
    correct = []
    wrong = []
    for A, B in NAMES:
        prompts.append(TEMPLATE.format(A=A, B=B))
        correct.append(B)
        wrong.append(A)
    return prompts, correct, wrong


# ------------------------
# Model helpers
# ------------------------

def load_model(model_name: str) -> HookedTransformer:
    print(f"Loading model {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    return model


def get_tokens(model: HookedTransformer, prompts: List[str]) -> torch.Tensor:
    return model.to_tokens(prompts, prepend_bos=True)


def get_last_token_id(model: HookedTransformer, text: str) -> int:
    """
    Get the id of the LAST token in `text` (handles multi-token names).
    """
    toks = model.to_tokens(text, prepend_bos=False)  # [1, seq]
    return toks[0, -1].item()


# ------------------------
# OV direction (via W_O)
# ------------------------

def analyze_ov_direction(
    model: HookedTransformer,
    layer: int,
    head: int,
    out_dir: str,
    top_k: int = 15,
):
    """
    Approximate OV direction using W_O:

    - W_O[layer, head] has shape [d_head, d_model]
    - We average over d_head to get a single residual-space vector [d_model]
    - Then project that through W_U to see which tokens get boosted/suppressed.
    """
    os.makedirs(out_dir, exist_ok=True)

    # W_O: [n_layers, n_heads, d_head, d_model]
    W_O = model.W_O[layer, head]      # [d_head, d_model]
    ov_vec = W_O.mean(dim=0)          # [d_model]

    W_U = model.W_U                   # [d_model, d_vocab]
    logits_dir = ov_vec @ W_U         # [d_vocab]

    top_pos = torch.topk(logits_dir, top_k).indices
    top_neg = torch.topk(-logits_dir, top_k).indices

    vocab = model.to_string
    top_pos_tokens = [(vocab(i), float(logits_dir[i].detach())) for i in top_pos]
    top_neg_tokens = [(vocab(i), float(logits_dir[i].detach())) for i in top_neg]

    path = os.path.join(out_dir, f"ov_analysis_L{layer}H{head}.txt")
    with open(path, "w") as f:
        f.write(f"OV analysis (via W_O) for layer {layer}, head {head}\n\n")
        f.write("Top positive tokens:\n")
        for tok, val in top_pos_tokens:
            f.write(f"{tok!r}\t{val:.4f}\n")
        f.write("\nTop negative tokens:\n")
        for tok, val in top_neg_tokens:
            f.write(f"{tok!r}\t{val:.4f}\n")

    print(f"[OV] Saved OV token analysis to {path}")


# ------------------------
# Attention patterns
# ------------------------

def analyze_attention_patterns(
    model: HookedTransformer,
    layer: int,
    head: int,
    prompts: List[str],
    out_dir: str,
):
    """
    Capture and plot the mean attention pattern for a single head
    over the IOI prompts.
    """
    os.makedirs(out_dir, exist_ok=True)
    tokens = get_tokens(model, prompts)

    attn_patterns = []

    def hook_pattern(pattern, hook):
        # pattern: [batch, n_heads, seq_Q, seq_K]
        attn_patterns.append(pattern[:, head].detach().cpu())  # [batch, seq_Q, seq_K]

    hook_name = f"blocks.{layer}.attn.hook_pattern"

    with model.hooks(fwd_hooks=[(hook_name, hook_pattern)]):
        _ = model(tokens)

    pattern_tensor = torch.cat(attn_patterns, dim=0)  # [batch_total, seq_Q, seq_K]
    mean_pattern = pattern_tensor.mean(dim=0)         # [seq_Q, seq_K]

    plt.figure(figsize=(6, 5))
    plt.imshow(mean_pattern.numpy(), aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.title(f"Mean attention pattern L{layer}H{head}")
    out_path = os.path.join(out_dir, f"attn_pattern_L{layer}H{head}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ATTN] Saved attention heatmap to {out_path}")


# ------------------------
# Δlogit contribution for that head
# ------------------------

def compute_delta_logit_for_head(
    model: HookedTransformer,
    layer: int,
    head: int,
    prompts: List[str],
    correct: List[str],
    wrong: List[str],
    out_dir: str,
):
    """
    Measure contribution of (layer, head) to IOI margin by ablating the head
    (zeroing its z output) and looking at change in (correct - wrong) logit.
    """
    os.makedirs(out_dir, exist_ok=True)
    tokens = get_tokens(model, prompts)

    target_pos = tokens.shape[1] - 1  # last position

    # Use LAST token of each name (handles multi-token names)
    correct_ids = [get_last_token_id(model, c) for c in correct]
    wrong_ids = [get_last_token_id(model, w) for w in wrong]

    # Baseline run
    with torch.no_grad():
        logits_base = model(tokens)  # [batch, seq, d_vocab]
    logits_base_last = logits_base[:, target_pos, :]

    base_correct = logits_base_last[range(len(correct)), correct_ids]
    base_wrong = logits_base_last[range(len(wrong)), wrong_ids]
    base_diff = base_correct - base_wrong

    # Ablated run: zero this head's output z at (layer, head)
    def zero_head(z, hook):
        # z: [batch, seq, n_heads, d_head]
        z = z.clone()
        z[:, :, head, :] = 0.0
        return z

    hook_name = f"blocks.{layer}.attn.hook_z"

    with torch.no_grad():
        logits_ablate = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, zero_head)],
        )

    logits_ablate_last = logits_ablate[:, target_pos, :]

    ablate_correct = logits_ablate_last[range(len(correct)), correct_ids]
    ablate_wrong = logits_ablate_last[range(len(wrong)), wrong_ids]
    ablate_diff = ablate_correct - ablate_wrong

    delta = ablate_diff - base_diff  # contribution of that head

    out_file = os.path.join(out_dir, f"delta_logit_L{layer}H{head}.txt")
    with open(out_file, "w") as f:
        for i, prompt in enumerate(prompts):
            f.write(f"Example {i}:\n")
            f.write(f"  prompt: {prompt}\n")
            f.write(f"  base_diff:   {base_diff[i].item():.4f}\n")
            f.write(f"  ablate_diff: {ablate_diff[i].item():.4f}\n")
            f.write(f"  delta (ablate - base): {delta[i].item():.4f}\n\n")

        f.write(f"\nMean delta across dataset: {delta.mean().item():.4f}\n")

    print(f"[ΔLOGIT] Saved delta-logit analysis to {out_file}")


# ------------------------
# Driver
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="e.g. 'EleutherAI/pythia-410m-deduped' or 'gpt2-medium'",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="results/case_studies")
    args = parser.parse_args()

    prompts, correct, wrong = build_ioi_prompts()
    model = load_model(args.model_name)

    tag = f"{args.model_name.replace('/', '_')}_L{args.layer}H{args.head}"
    out_dir = os.path.join(args.out_dir, tag)

    analyze_attention_patterns(model, args.layer, args.head, prompts, out_dir)
    analyze_ov_direction(model, args.layer, args.head, out_dir)
    compute_delta_logit_for_head(model, args.layer, args.head, prompts, correct, wrong, out_dir)


if __name__ == "__main__":
    main()

