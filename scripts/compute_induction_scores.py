#!/usr/bin/env python
import json
import random
from typing import List, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer


DEVICE = "cpu"
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


LETTERS = [" A", " B", " C", " D", " E", " F"]  # simple small alphabet


def build_induction_dataset(
    model: HookedTransformer,
    n_prompts: int = 64,
) -> List[dict]:
    """
    Build a very simple induction-like dataset:
    We construct prompts of the form: "A B X Y A"
    and test whether the model prefers B over C after the second A.

    For each example:
      - choose distinct letters A, B, C from LETTERS
      - optional filler letters in between
      - target position = final A
      - correct next token = B
      - wrong next token = C
    """
    examples = []
    for _ in range(n_prompts):
        A, B, C = random.sample(LETTERS, 3)
        # optionally add 0–2 filler letters
        fillers = random.sample(LETTERS, k=random.randint(0, 2))
        prompt = " ".join([A.strip(), B.strip()] + [f.strip() for f in fillers] + [A.strip()])
        # Build strings for targets. We use last token of each string.
        correct_str = B
        wrong_str = C

        # Tokenize prompt and targets
        toks = model.to_tokens(prompt, prepend_bos=False)
        correct_toks = model.to_tokens(correct_str, prepend_bos=False)
        wrong_toks = model.to_tokens(wrong_str, prepend_bos=False)

        correct_id = int(correct_toks[0, -1])
        wrong_id = int(wrong_toks[0, -1])
        target_pos = toks.shape[1] - 1  # final token position (the second A)

        examples.append(
            dict(
                prompt=prompt,
                toks=toks,
                target_pos=target_pos,
                correct_id=correct_id,
                wrong_id=wrong_id,
            )
        )
    return examples


def compute_margin(
    model: HookedTransformer,
    ex: dict,
) -> float:
    """
    Compute induction margin: logit(correct) - logit(wrong)
    at ex['target_pos'].
    """
    with torch.no_grad():
        logits = model(ex["toks"].to(DEVICE))  # [1, seq, vocab]
    pos = ex["target_pos"]
    logits_pos = logits[0, pos]
    return float(logits_pos[ex["correct_id"]] - logits_pos[ex["wrong_id"]])


def compute_head_score_for_model(
    model_name: str,
    family: str,
    outfile: str,
    n_prompts: int = 64,
):
    print(f"[INFO] Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=DEVICE)

    print(f"[INFO] Building induction dataset (n={n_prompts})...")
    dataset = build_induction_dataset(model, n_prompts=n_prompts)

    print("[INFO] Computing base margins...")
    base_margins = []
    for ex in dataset:
        m = compute_margin(model, ex)
        base_margins.append(m)
    base_mean = float(np.mean(base_margins))
    print(f"[INFO] Base mean induction margin: {base_mean:.4f}")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    rows = []

    print("[INFO] Scoring heads by ablation...")
    for layer in tqdm(range(n_layers), desc=f"{model_name} layers"):
        for head in range(n_heads):

            def hook_fn(z, hook, head_idx=head):
                # z: [batch, pos, head_index, d_head]
                z = z.clone()
                z[:, :, head_idx, :] = 0.0
                return z

            hook_name = f"blocks.{layer}.attn.hook_z"

            ablated_margins = []
            for ex in dataset:
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    with torch.no_grad():
                        logits = model(ex["toks"].to(DEVICE))
                pos = ex["target_pos"]
                logits_pos = logits[0, pos]
                margin = float(
                    logits_pos[ex["correct_id"]] - logits_pos[ex["wrong_id"]]
                )
                ablated_margins.append(margin)

            mean_abl = float(np.mean(ablated_margins))
            delta = mean_abl - base_mean  # like IOI: negative means ablation hurts
            rows.append(
                dict(
                    family=family,
                    model=model_name,
                    layer=layer,
                    head=head,
                    base_mean_margin=base_mean,
                    mean_margin_ablated=mean_abl,
                    delta_induction=delta,
                    abs_delta_induction=abs(delta),
                )
            )

    df = pd.DataFrame(rows)
    df.to_csv(outfile, index=False)
    print(f"[OUT] Wrote induction scores for {model_name} to {outfile}")


def main():
    # You can change models here if needed
    jobs = [
        ("pythia", "pythia-160m", "results/induction_head_scores_pythia-160m.csv"),
        ("gpt2", "gpt2-medium", "results/induction_head_scores_gpt2-medium.csv"),
    ]
    for family, model_name, outfile in jobs:
        compute_head_score_for_model(
            model_name=model_name,
            family=family,
            outfile=outfile,
            n_prompts=64,  # tweak if too slow or too noisy
        )


if __name__ == "__main__":
    main()

