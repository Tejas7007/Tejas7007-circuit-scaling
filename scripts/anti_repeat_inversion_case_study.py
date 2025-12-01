#!/usr/bin/env python
import os
from typing import List, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

DEVICE = "cpu"
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_inverted_heads(path: str, max_heads: int = 3) -> List[Tuple[int, int]]:
    """Load inverted IOI heads from the expanded taxonomy file."""
    df = pd.read_csv(path)
    print("[INFO] Taxonomy columns:", list(df.columns))

    df = df[df["taxonomy"] == "inverted"].copy()

    if "abs_delta_ioi" in df.columns:
        df = df.sort_values("abs_delta_ioi", ascending=False)

    heads = []
    for _, row in df.head(max_heads).iterrows():
        heads.append((int(row["layer"]), int(row["head"])))

    print("[INFO] Selected inverted heads:", heads)
    return heads


def build_anti_repeat_prompts() -> List[str]:
    names = ["John", "Mary", "Alice", "Bob"]
    prompts = []

    for name in names:
        prompts.append(f"{name} {name} {name}")
        prompts.append(f"The {name} {name} {name}")

    prompts.append("1 2 3 3")
    prompts.append("red red red")
    prompts.append("the the the")

    return prompts


def get_logits_for_prompt(model: HookedTransformer, prompt: str):
    toks = model.to_tokens(prompt, prepend_bos=False).to(DEVICE)
    with torch.no_grad():
        logits = model(toks)
    return toks, logits


def compute_repeat_margin(model: HookedTransformer, prompt: str) -> float:
    toks, logits = get_logits_for_prompt(model, prompt)
    pos = logits.shape[1] - 1

    last_id = int(toks[0, -1])
    vocab = logits.shape[-1]
    alt = (last_id + 1) % vocab

    logpos = logits[0, pos]
    return float(logpos[last_id] - logpos[alt])


def case_study_for_head(
    model_name: str,
    layer: int,
    head: int,
    prompts: List[str],
    out_dir_figs: str,
    tag: str,
):
    print(f"[INFO] Loading model {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=DEVICE)

    fig, axes = plt.subplots(1, len(prompts), figsize=(4 * len(prompts), 3))
    if len(prompts) == 1:
        axes = [axes]

    for i, prompt in enumerate(prompts):
        toks = model.to_tokens(prompt, prepend_bos=False).to(DEVICE)
        captured = {}

        def hook_pattern(pat, hook):
            captured["pattern"] = pat.detach().cpu()
            return pat

        hook_name = f"blocks.{layer}.attn.hook_pattern"

        with model.hooks(fwd_hooks=[(hook_name, hook_pattern)]):
            _ = model(toks)

        if "pattern" not in captured:
            continue

        pat = captured["pattern"][0, head]

        im = axes[i].imshow(pat, aspect="auto", origin="lower")
        axes[i].set_title(prompt, fontsize=8)
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(f"{model_name} – L{layer}H{head} anti-repeat patterns", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(out_dir_figs, exist_ok=True)
    fig_path = os.path.join(out_dir_figs, f"{tag}_L{layer}H{head}_attn.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OUT] Saved: {fig_path}")

    base_margins = []
    ablated_margins = []

    def ablate(z, hook):
        z = z.clone()
        z[:, :, head, :] = 0.0
        return z

    hook_z = f"blocks.{layer}.attn.hook_z"

    for prompt in prompts:
        base_margins.append(compute_repeat_margin(model, prompt))

        toks = model.to_tokens(prompt, prepend_bos=False).to(DEVICE)
        with model.hooks(fwd_hooks=[(hook_z, ablate)]):
            with torch.no_grad():
                logits = model(toks)

        last_id = int(toks[0, -1])
        vocab = logits.shape[-1]
        alt = (last_id + 1) % vocab
        pos = logits.shape[1] - 1
        logpos = logits[0, pos]

        ablated_margins.append(float(logpos[last_id] - logpos[alt]))

    base_mean = float(np.mean(base_margins))
    abl_mean = float(np.mean(ablated_margins))
    delta = abl_mean - base_mean

    print(
        f"[STATS] {model_name} L{layer}H{head}: "
        f"base={base_mean:.4f}, ablated={abl_mean:.4f}, delta={delta:.4f}"
    )


def main():
    taxonomy = "results/ioi_divergence_taxonomy_expanded.csv"
    if not os.path.exists(taxonomy):
        raise FileNotFoundError("Run divergence_taxonomy_expanded.py first.")

    inverted = load_inverted_heads(taxonomy, max_heads=3)
    print(f"[INFO] Inverted IOI heads from taxonomy: {inverted}")

    prompts = build_anti_repeat_prompts()

    out = "figs/anti_repeat_inversion_case_studies"
    os.makedirs(out, exist_ok=True)

    for (layer, head) in inverted:
        print(f"\n=== Inverted head L{layer}H{head} ===\n")

        case_study_for_head(
            model_name="pythia-410m",
            layer=layer,
            head=head,
            prompts=prompts,
            out_dir_figs=out,
            tag="pythia410m",
        )

        case_study_for_head(
            model_name="gpt2-medium",
            layer=layer,
            head=head,
            prompts=prompts,
            out_dir_figs=out,
            tag="gpt2medium",
        )


if __name__ == "__main__":
    main()

