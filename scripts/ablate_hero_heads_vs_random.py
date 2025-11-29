import os
import random
from collections import defaultdict

import torch
import pandas as pd
from transformer_lens import HookedTransformer

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

FAMILY = "gpt2"
MODEL_NAME = "gpt2-large"
HF_NAME = "gpt2-large"

# Where we read hero heads from
HERO_HEADS_CSV = "paper/tables/hero_heads_for_paper.csv"  # falls back to results if needed

# Where to save results
ABLATION_DIR = "results/ablation"
os.makedirs(ABLATION_DIR, exist_ok=True)
ABLATION_CSV = os.path.join(ABLATION_DIR, "hero_vs_random_ablation.csv")

# Dataset sizes and ablation settings
N_IOI = 100          # smaller to be fast + memory safe
N_ANTI = 100
KS = [0, 2, 3, 8]    # number of heads to ablate (3 = all hero heads)
RAND_TRIALS = 3      # random ablation sets per K

# Shared names across tasks
NAMES = ["John", "Mary", "Alice", "Bob", "Tom", "Sarah"]

# Chunk size for forward passes to avoid OOM
BATCH_SIZE = 32

# ---------------------------------------------------------
# Templates and datasets
# ---------------------------------------------------------

IOI_TEMPLATES = [
    "{A} and {B} went to the store. Later, {A} gave the bag to {B}.",
    "{A} and {B} were debating yesterday. Later, {A} thanked {B} for the discussion.",
    "Yesterday, {A} met {B} for coffee. In the evening, {A} messaged {B} to say thanks.",
]

ANTI_TEMPLATES = [
    "{A} kept interrupting {B} in the meeting. Everyone only heard {A}, {A}, and {A} again. The person who kept repeating was",
    "During the talk, the audience heard {A} again and again. {A} spoke over {B} multiple times. The name that was repeated was",
    "In the transcript, the same speaker appears over and over: {A}, {A}, {A}. The repeated name is",
]


def build_ioi_dataset(n=N_IOI):
    """IOI-style task: predict B given context involving A and B."""
    prompts = []
    answers = []

    for _ in range(n):
        a, b = random.sample(NAMES, 2)
        template = random.choice(IOI_TEMPLATES)
        prompt = template.format(A=a, B=b)
        prompts.append(prompt)
        answers.append(b)
    return prompts, answers


def build_anti_dataset(n=N_ANTI):
    """
    Anti-repeat-style: model should identify the *repeated* name (A).
    """
    prompts = []
    answers = []

    for _ in range(n):
        a, b = random.sample(NAMES, 2)
        template = random.choice(ANTI_TEMPLATES)
        prompt = template.format(A=a, B=b)
        prompts.append(prompt)
        answers.append(a)
    return prompts, answers


# ---------------------------------------------------------
# Device + metrics
# ---------------------------------------------------------


def get_device():
    """
    Force CPU to avoid MPS / GPU OOM for gpt2-large.
    """
    print("Using device: cpu (forced to avoid MPS/GPU OOM)")
    return torch.device("cpu")


def get_name_token_ids(model):
    """
    Map each name in NAMES -> its single token id.
    We assume each name is a single token in the tokenizer.
    """
    token_ids = {}
    for name in NAMES:
        toks = model.to_tokens(name, prepend_bos=False)
        if toks.shape[-1] != 1:
            raise ValueError(f"Name '{name}' is not a single token for this tokenizer.")
        token_ids[name] = toks[0, 0].item()
    return token_ids


def compute_name_metrics(model, prompts, answers, name_token_ids, ablate_heads=None):
    """
    Evaluate accuracy and mean logit diff on a name-prediction task with chunking.
    - prompts: list[str]
    - answers: list[str] (must be in NAMES)
    - name_token_ids: dict name -> token id
    - ablate_heads: None or list[(layer, head)]
    """
    device = model.cfg.device

    # Tokenize full batch once
    tokens = model.to_tokens(prompts, prepend_bos=True).to(device)

    # Candidate token ids (for all names)
    cand_ids_list = [name_token_ids[name] for name in NAMES]
    cand_ids = torch.tensor(cand_ids_list, device=device)

    # Correct ids per example
    correct_ids = torch.tensor([name_token_ids[a] for a in answers], device=device)

    # Build hooks if needed
    fwd_hooks = None
    if ablate_heads is not None and len(ablate_heads) > 0:
        layer_to_heads = defaultdict(list)
        for layer, head in ablate_heads:
            layer_to_heads[int(layer)].append(int(head))

        fwd_hooks = []

        for layer, heads in layer_to_heads.items():
            hook_name = f"blocks.{layer}.attn.hook_z"

            def make_hook(heads_for_layer):
                def hook_fn(z, hook):
                    # z: [batch, pos, n_heads, d_head]
                    z[:, :, heads_for_layer, :] = 0.0
                    return z

                return hook_fn

            fwd_hooks.append((hook_name, make_hook(heads)))

    # Run in chunks to keep memory small
    logits_chunks = []
    B = tokens.shape[0]
    for start in range(0, B, BATCH_SIZE):
        end = min(B, start + BATCH_SIZE)
        tok_chunk = tokens[start:end]

        if fwd_hooks is None:
            logits_chunk = model(tok_chunk)
        else:
            logits_chunk = model.run_with_hooks(tok_chunk, fwd_hooks=fwd_hooks, return_type="logits")

        logits_chunks.append(logits_chunk[:, -1, :].detach())  # [chunk, vocab]

    final_logits = torch.cat(logits_chunks, dim=0)  # [batch, vocab]

    # Candidate logits: [batch, num_names]
    cand_logits = final_logits[:, cand_ids]

    # Accuracy: pick highest logit among candidate names
    pred_idx = cand_logits.argmax(dim=-1)  # [batch]
    preds = cand_ids[pred_idx]             # [batch]
    acc = (preds == correct_ids).float().mean().item()

    # Mean logit diff: correct vs other candidates
    correct_mask = cand_ids.unsqueeze(0) == correct_ids.unsqueeze(1)  # [batch, num_names]
    correct_logits = (cand_logits * correct_mask).sum(dim=1)

    other_mask = ~correct_mask
    other_counts = other_mask.sum(dim=1).clamp(min=1)
    other_logits_mean = (cand_logits * other_mask).sum(dim=1) / other_counts

    mean_diff = (correct_logits - other_logits_mean).mean().item()

    return {
        "acc": acc,
        "mean_diff": mean_diff,
    }


def evaluate_with_ablation(model, name_token_ids, ioi_prompts, ioi_answers, anti_prompts, anti_answers, ablate_heads=None):
    """Evaluate IOI and anti datasets under a given ablation."""
    ioi_metrics = compute_name_metrics(model, ioi_prompts, ioi_answers, name_token_ids, ablate_heads)
    anti_metrics = compute_name_metrics(model, anti_prompts, anti_answers, name_token_ids, ablate_heads)

    return {
        "acc_ioi": ioi_metrics["acc"],
        "mean_diff_ioi": ioi_metrics["mean_diff"],
        "acc_anti": anti_metrics["acc"],
        "mean_diff_anti": anti_metrics["mean_diff"],
    }


# ---------------------------------------------------------
# Hero heads + random heads
# ---------------------------------------------------------


def load_hero_heads():
    """Load hero heads from CSV, filter to gpt2/gpt2-large shared heads."""
    if os.path.exists(HERO_HEADS_CSV):
        path = HERO_HEADS_CSV
    else:
        path = "results/hero_heads_for_paper.csv"

    df = pd.read_csv(path)

    df = df[(df["family"] == FAMILY) & (df["model"] == MODEL_NAME)]
    hero_df = df[df["category"] == "shared"]

    hero_heads = set()
    for _, row in hero_df.iterrows():
        hero_heads.add((int(row["layer"]), int(row["head"])))
    return hero_heads


def main():
    random.seed(0)
    torch.manual_seed(0)

    device = get_device()

    # --------------------
    # Build datasets
    # --------------------
    print("Building IOI and anti-repeat datasets...")
    ioi_prompts, ioi_answers = build_ioi_dataset(N_IOI)
    anti_prompts, anti_answers = build_anti_dataset(N_ANTI)

    # --------------------
    # Load model
    # --------------------
    print(f"\n=== Loading model {HF_NAME} for {FAMILY}/{MODEL_NAME} ===")
    model = HookedTransformer.from_pretrained(
        HF_NAME,
        device=device,
    )

    name_token_ids = get_name_token_ids(model)

    # --------------------
    # Hero heads
    # --------------------
    hero_heads = load_hero_heads()
    print(f"\nTotal hero heads (shared) for {MODEL_NAME}: {len(hero_heads)}")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_heads = {(layer, head) for layer in range(n_layers) for head in range(n_heads)}
    non_hero_heads = list(all_heads - hero_heads)

    print(f"Total heads in model: {len(all_heads)}")
    print(f"Non-hero heads available for random ablations: {len(non_hero_heads)}")

    rows = []

    # --------------------
    # Baseline (no ablation)
    # --------------------
    print("\nEvaluating baseline (no ablation)...")
    baseline_metrics = evaluate_with_ablation(
        model,
        name_token_ids,
        ioi_prompts,
        ioi_answers,
        anti_prompts,
        anti_answers,
        ablate_heads=None,
    )
    print(
        f"  IOI baseline:  acc={baseline_metrics['acc_ioi']:.3f}, "
        f"mean_diff={baseline_metrics['mean_diff_ioi']:.3f}"
    )
    print(
        f"  Anti baseline: acc={baseline_metrics['acc_anti']:.3f}, "
        f"mean_diff={baseline_metrics['mean_diff_anti']:.3f}"
    )

    rows.append(
        {
            "family": FAMILY,
            "model": MODEL_NAME,
            "k": 0,
            "set_type": "baseline",
            "trial": 0,
            "n_hero_heads_total": len(hero_heads),
            "n_heads_total": len(all_heads),
            **baseline_metrics,
        }
    )

    # --------------------
    # Ablations
    # --------------------
    for k in KS:
        print(f"\n=== K = {k} heads ===")

        # Hero ablation
        if k > 0 and len(hero_heads) >= k:
            hero_subset = sorted(list(hero_heads))[:k]
            print(f"  Hero ablation on first {k} hero heads: {hero_subset}")

            metrics = evaluate_with_ablation(
                model,
                name_token_ids,
                ioi_prompts,
                ioi_answers,
                anti_prompts,
                anti_answers,
                ablate_heads=hero_subset,
            )
            print(
                f"    HERO IOI:  acc={metrics['acc_ioi']:.3f}, "
                f"mean_diff={metrics['mean_diff_ioi']:.3f}"
            )
            print(
                f"    HERO anti: acc={metrics['acc_anti']:.3f}, "
                f"mean_diff={metrics['mean_diff_anti']:.3f}"
            )

            rows.append(
                {
                    "family": FAMILY,
                    "model": MODEL_NAME,
                    "k": k,
                    "set_type": "hero",
                    "trial": 0,
                    "n_hero_heads_total": len(hero_heads),
                    "n_heads_total": len(all_heads),
                    **metrics,
                }
            )
        elif k > 0:
            print(f"  [Skipped hero ablation for k={k}: only {len(hero_heads)} hero heads available]")

        if k == 0:
            continue

        # Random ablations
        print(f"  Random ablations (non-hero heads), {RAND_TRIALS} trials...")
        for trial in range(1, RAND_TRIALS + 1):
            if len(non_hero_heads) < k:
                print(f"    [Skipped random trial {trial}: not enough non-hero heads]")
                continue

            rand_subset = random.sample(non_hero_heads, k)
            metrics = evaluate_with_ablation(
                model,
                name_token_ids,
                ioi_prompts,
                ioi_answers,
                anti_prompts,
                anti_answers,
                ablate_heads=rand_subset,
            )

            print(
                f"    Trial {trial}: IOI acc={metrics['acc_ioi']:.3f}, "
                f"anti acc={metrics['acc_anti']:.3f}"
            )

            rows.append(
                {
                    "family": FAMILY,
                    "model": MODEL_NAME,
                    "k": k,
                    "set_type": "random",
                    "trial": trial,
                    "n_hero_heads_total": len(hero_heads),
                    "n_heads_total": len(all_heads),
                    **metrics,
                }
            )

    # --------------------
    # Save CSV
    # --------------------
    df_out = pd.DataFrame(rows)
    df_out.to_csv(ABLATION_CSV, index=False)
    print(f"\nWrote ablation summary to {ABLATION_CSV}")


if __name__ == "__main__":
    main()

