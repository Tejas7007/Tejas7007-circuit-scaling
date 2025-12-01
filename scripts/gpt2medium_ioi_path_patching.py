#!/usr/bin/env python
"""
Exhaustive IOI path patching for GPT-2-Medium.

For every attention head and every MLP block, we:
  - Run a CLEAN IOI dataset (A gives book to B).
  - Run a CORRUPTED dataset (swap A and B).
  - Patch that unit's activation in the corrupted run with the clean run's activation.
  - Measure how much the IOI margin (correct-name vs wrong-name logit) recovers.

Outputs:
  - results/gpt2medium_ioi_path_patching.csv
  - paper/figs/gpt2medium_ioi_path_patching_top_units.png
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

# ------------------- Config -------------------

MODEL_NAME = "gpt2-medium"
OUT_CSV = Path("results/gpt2medium_ioi_path_patching.csv")
TOP_FIG = Path("paper/figs/gpt2medium_ioi_path_patching_top_units.png")

# Device selection (MPS -> CUDA -> CPU)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"[INFO] Using device: {device}")


# ------------------- IOI dataset -------------------

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


def build_ioi_pairs(names: List[str]) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Returns:
      clean_prompts: A gives book to B (correct IOI)
      corrupt_prompts: B and A swapped (A/B swapped)
      pairs: list of (A, B) for the CLEAN version
    """
    clean = []
    corrupt = []
    pairs: List[Tuple[str, str]] = []
    for i, A in enumerate(names):
        for j, B in enumerate(names):
            if i == j:
                continue
            clean.append(TEMPLATE.format(A=A, B=B))
            corrupt.append(TEMPLATE.format(A=B, B=A))
            pairs.append((A, B))
    return clean, corrupt, pairs


def get_tokens(model: HookedTransformer, prompts: List[str]) -> torch.Tensor:
    toks = model.to_tokens(prompts, prepend_bos=True)
    return toks.to(device)


def last_token_id(model: HookedTransformer, text: str) -> int:
    toks = model.to_tokens(text, prepend_bos=False)
    return toks[0, -1].item()


def compute_ioi_margin_from_logits(
    logits: torch.Tensor,
    pairs: List[Tuple[str, str]],
    name_to_id: Dict[str, int],
    target_pos: int,
) -> torch.Tensor:
    """
    logits: [batch, seq, vocab]
    margin_i = logit(correct_name_last_token) - logit(wrong_name_last_token)
    """
    logits_last = logits[:, target_pos, :]  # [batch, vocab]
    margins = []
    for i, (A, B) in enumerate(pairs):
        logit_correct = logits_last[i, name_to_id[B]]
        logit_wrong = logits_last[i, name_to_id[A]]
        margins.append(logit_correct - logit_wrong)
    return torch.stack(margins)


# ------------------- Main path patching logic -------------------

def path_patch_unit(
    model: HookedTransformer,
    tokens_clean: torch.Tensor,
    tokens_corrupt: torch.Tensor,
    pairs: List[Tuple[str, str]],
    name_to_id: Dict[str, int],
    target_pos: int,
    node_type: str,
    layer: int,
    head: int,
) -> Dict:
    """
    Patch a single node's activation in corrupted run with the clean run's activation.

    node_type: "attn" or "mlp"
    For attn, we patch blocks.{layer}.attn.hook_z for a single head.
    For mlp, we patch blocks.{layer}.mlp.hook_post (whole vector).
    """

    # First, run base corrupted to get baseline IOI margin
    with torch.no_grad():
        base_logits = model(tokens_corrupt)
    base_margins = compute_ioi_margin_from_logits(
        base_logits, pairs, name_to_id, target_pos
    )
    base_mean = base_margins.mean().item()

    # Run CLEAN + CORRUPTED with cache to get activations
    if node_type == "attn":
        hook_name = f"blocks.{layer}.attn.hook_z"
    elif node_type == "mlp":
        hook_name = f"blocks.{layer}.mlp.hook_post"
    else:
        raise ValueError(f"Unknown node_type: {node_type}")

    names_filter = lambda name: name == hook_name

    with torch.no_grad():
        logits_clean, cache_clean = model.run_with_cache(
            tokens_clean, names_filter=names_filter
        )
        logits_corrupt, cache_corrupt = model.run_with_cache(
            tokens_corrupt, names_filter=names_filter
        )

    # Sanity: compute clean & corrupt margins
    clean_margins = compute_ioi_margin_from_logits(
        logits_clean, pairs, name_to_id, target_pos
    )
    corrupt_margins = compute_ioi_margin_from_logits(
        logits_corrupt, pairs, name_to_id, target_pos
    )
    clean_mean = clean_margins.mean().item()
    corrupt_mean = corrupt_margins.mean().item()

    # Prepare patch function
    if node_type == "attn":
        clean_act = cache_clean[hook_name]  # [batch, seq, n_heads, d_head]

        def patch_fn(z, hook, h=head, clean_act=clean_act):
            z = z.clone()
            z[:, :, h, :] = clean_act[:, :, h, :]
            return z

    else:  # mlp
        clean_act = cache_clean[hook_name]  # [batch, seq, d_model]

        def patch_fn(mlp_out, hook, clean_act=clean_act):
            # Replace entire MLP output with clean
            return clean_act

    # Run CORRUPTED with patch
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            tokens_corrupt,
            fwd_hooks=[(hook_name, patch_fn)],
        )

    patched_margins = compute_ioi_margin_from_logits(
        patched_logits, pairs, name_to_id, target_pos
    )
    patched_mean = patched_margins.mean().item()

    # Effects
    patch_effect = patched_mean - corrupt_mean  # recovery when patching this node
    # Also record how far clean vs corrupt are apart
    total_gap = clean_mean - corrupt_mean

    return {
        "type": node_type,
        "layer": layer,
        "head": head if node_type == "attn" else -1,
        "baseline_mean_margin_corrupt": corrupt_mean,
        "clean_mean_margin": clean_mean,
        "patched_mean_margin": patched_mean,
        "total_clean_corrupt_gap": total_gap,
        "patch_effect": patch_effect,
    }


def main():
    # Ensure dirs
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    TOP_FIG.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building IOI dataset...")
    clean_prompts, corrupt_prompts, pairs = build_ioi_pairs(NAMES)
    print(f"[INFO] #IOI pairs: {len(pairs)}")

    print(f"[INFO] Loading model {MODEL_NAME} on {device}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

    tokens_clean = get_tokens(model, clean_prompts)
    tokens_corrupt = get_tokens(model, corrupt_prompts)
    target_pos = tokens_clean.shape[1] - 1

    # Precompute name → vocab id
    names = set([n for (A, B) in pairs for n in (A, B)])
    name_to_id = {name: last_token_id(model, name) for name in names}
    print("[INFO] Built name_to_id for:", name_to_id)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    rows = []

    # ---------- Attention heads ----------
    print("[STEP] Path patching all attention heads...")
    for layer in range(n_layers):
        for head in range(n_heads):
            print(f"[ATTN] Layer {layer}, head {head}")
            row = path_patch_unit(
                model=model,
                tokens_clean=tokens_clean,
                tokens_corrupt=tokens_corrupt,
                pairs=pairs,
                name_to_id=name_to_id,
                target_pos=target_pos,
                node_type="attn",
                layer=layer,
                head=head,
            )
            rows.append(row)

    # ---------- MLP blocks ----------
    print("[STEP] Path patching all MLP blocks...")
    for layer in range(n_layers):
        print(f"[MLP] Layer {layer}")
        row = path_patch_unit(
            model=model,
            tokens_clean=tokens_clean,
            tokens_corrupt=tokens_corrupt,
            pairs=pairs,
            name_to_id=name_to_id,
            target_pos=target_pos,
            node_type="mlp",
            layer=layer,
            head=-1,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OUT] Wrote exhaustive path-patching results to {OUT_CSV}")

    # ---------- Quick summary + top-units plot ----------
    df["abs_patch_effect"] = df["patch_effect"].abs()

    print("\n=== Top 20 units by |patch_effect| ===")
    print(
        df.sort_values("abs_patch_effect", ascending=False)
          .head(20)
          .to_string(index=False)
    )

    # Bar plot: top 15 units
    top = df.sort_values("abs_patch_effect", ascending=False).head(15)
    labels = [
        f"{t.upper()} L{L}H{H}" if t == "attn" else f"MLP L{L}"
        for t, L, H in zip(top["type"], top["layer"], top["head"])
    ]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(top)), top["patch_effect"])
    plt.xticks(range(len(top)), labels, rotation=45, ha="right")
    plt.ylabel("Patch effect on IOI margin\n(patched_mean - corrupt_mean)")
    plt.title("GPT-2-Medium IOI Path Patching – Top Units")
    plt.tight_layout()
    plt.savefig(TOP_FIG, dpi=240)
    plt.close()
    print(f"[OUT] Wrote top-units path-patching figure to {TOP_FIG}")


if __name__ == "__main__":
    main()

