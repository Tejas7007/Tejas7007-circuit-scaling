import os
import json
import argparse
from typing import List, Dict

import torch
import pandas as pd
from transformer_lens import HookedTransformer


def get_device(preferred: str | None = None) -> torch.device:
    if preferred is not None:
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def map_to_hf_name(family: str, model_name: str) -> str:
    """Map our family/model identifiers to HuggingFace IDs used by HookedTransformer."""
    if family == "gpt2":
        # "gpt2", "gpt2-medium", "gpt2-large"
        return model_name
    if family == "pythia":
        # e.g. "pythia-160m" -> "EleutherAI/pythia-160m-deduped"
        return f"EleutherAI/{model_name}-deduped"
    if family == "gpt-neo":
        # Our csv uses "gpt-neo-125M"
        return f"EleutherAI/{model_name}"
    if family == "opt":
        # Only 125m in this project
        if model_name == "opt-125m":
            return "facebook/opt-125m"
    # Fallback: just return what we have
    return model_name


def get_token_id(model: HookedTransformer, text: str) -> int:
    toks = model.to_tokens(text, prepend_bos=False)
    # assume exactly one token
    if toks.shape[-1] != 1:
        raise ValueError(f"text {text!r} does not map to a single token (got {toks.shape[-1]})")
    return toks[0, 0].item()


def compute_logit_diff(logits: torch.Tensor, correct_id: int, wrong_id: int) -> float:
    """Return scalar logit(correct) - logit(wrong) at final position."""
    last_logits = logits[0, -1, :]
    return (last_logits[correct_id] - last_logits[wrong_id]).item()


def patch_head_value(layer: int, head: int, clean_cache):
    """
    Return a list of (hook_name, hook_fn) pairs that patch the value stream
    of a single head from the clean cache into a corrupted run.
    """
    hook_name = f"blocks.{layer}.attn.hook_v"

    def hook_fn(v, hook):
        # v: [batch, seq, n_heads, d_head]
        v[:, :, head, :] = clean_cache[hook.name][:, :, head, :]
        return v

    # IMPORTANT: run_with_hooks expects a list of (name, fn) pairs
    return [(hook_name, hook_fn)]


def run_causal_trace_for_pair(
    model: HookedTransformer,
    device: torch.device,
    prompt_clean: str,
    prompt_corrupt: str,
    correct_tok: int,
    wrong_tok: int,
    layer: int,
    head: int,
) -> Dict[str, float]:
    """Run one causal trace for a single (clean, corrupt) prompt pair."""
    tokens_clean = model.to_tokens(prompt_clean, prepend_bos=True).to(device)
    tokens_corrupt = model.to_tokens(prompt_corrupt, prepend_bos=True).to(device)

    # Clean run with cache
    clean_logits, clean_cache = model.run_with_cache(tokens_clean, return_type="logits")
    # Corrupted baseline
    corrupted_logits = model(tokens_corrupt, return_type="logits")

    base_ld = compute_logit_diff(corrupted_logits, correct_tok, wrong_tok)

    # Patched run: patch the v stream of (layer, head) from clean cache
    hooks = patch_head_value(layer, head, clean_cache)
    patched_logits = model.run_with_hooks(tokens_corrupt, fwd_hooks=hooks, return_type="logits")
    patched_ld = compute_logit_diff(patched_logits, correct_tok, wrong_tok)

    return {
        "base_logit_diff": base_ld,
        "patched_logit_diff": patched_ld,
        "influence": patched_ld - base_ld,
    }


def build_ioi_prompts() -> List[Dict[str, str]]:
    # Classic IOI-style name swap prompts
    pairs = [
        (
            "When John and Mary went to the store, John gave a book to",
            "When John and Mary went to the store, Mary gave a book to",
            " John",
            " Mary",
        ),
        (
            "When Mary and John went to the store, Mary gave a book to",
            "When Mary and John went to the store, John gave a book to",
            " Mary",
            " John",
        ),
    ]
    out = []
    for clean, corrupt, correct, wrong in pairs:
        out.append(
            {
                "prompt_clean": clean,
                "prompt_corrupt": corrupt,
                "correct": correct,
                "wrong": wrong,
            }
        )
    return out


def build_anti_prompts() -> List[Dict[str, str]]:
    # Simple repetition vs non-repetition prompts
    pairs = [
        (
            "The cat sat on the mat and the cat",
            "The cat sat on the mat and and and",
            " cat",
            " and",
        ),
        (
            "I like pizza and I like pasta",
            "I like pizza and I like like",
            " pasta",
            " like",
        ),
    ]
    out = []
    for clean, corrupt, correct, wrong in pairs:
        out.append(
            {
                "prompt_clean": clean,
                "prompt_corrupt": corrupt,
                "correct": correct,
                "wrong": wrong,
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heads_csv",
        type=str,
        default="results/hero_heads_for_paper.csv",
        help="CSV of hero heads (family, model, category, layer, head, ...).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/causal_tracing/hero_head_causal_results.json",
        help="Where to write JSON results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Preferred device: cuda, mps, or cpu (auto if omitted).",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    df = pd.read_csv(args.heads_csv)
    # Keep only heads we care about: shared / ioi_only / anti_only
    df = df[df["category"].isin(["shared", "ioi_only", "anti_only"])].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ioi_specs = build_ioi_prompts()
    anti_specs = build_anti_prompts()

    results: List[Dict] = []

    # Cache models by HF name to avoid re-loading
    model_cache: Dict[str, HookedTransformer] = {}

    for idx, row in df.iterrows():
        family = row["family"]
        model_name = row["model"]
        category = row["category"]
        layer = int(row["layer"])
        head = int(row["head"])

        hf_name = map_to_hf_name(family, model_name)

        if hf_name not in model_cache:
            print(f"\n=== Loading model {hf_name} for {family}/{model_name} ===")
            model = HookedTransformer.from_pretrained(hf_name, device=device)
            model_cache[hf_name] = model
        else:
            model = model_cache[hf_name]

        print(f"\n=== Tracing {family}/{model_name} — {category} — L{layer}H{head} ===")

        ioi_influences: List[float] = []
        anti_influences: List[float] = []

        # For each IOI spec
        for spec in ioi_specs:
            correct_id = get_token_id(model, spec["correct"])
            wrong_id = get_token_id(model, spec["wrong"])
            metrics = run_causal_trace_for_pair(
                model=model,
                device=device,
                prompt_clean=spec["prompt_clean"],
                prompt_corrupt=spec["prompt_corrupt"],
                correct_tok=correct_id,
                wrong_tok=wrong_id,
                layer=layer,
                head=head,
            )
            ioi_influences.append(metrics["influence"])

        # For each anti-repeat spec
        for spec in anti_specs:
            correct_id = get_token_id(model, spec["correct"])
            wrong_id = get_token_id(model, spec["wrong"])
            metrics = run_causal_trace_for_pair(
                model=model,
                device=device,
                prompt_clean=spec["prompt_clean"],
                prompt_corrupt=spec["prompt_corrupt"],
                correct_tok=correct_id,
                wrong_tok=wrong_id,
                layer=layer,
                head=head,
            )
            anti_influences.append(metrics["influence"])

        result_row = {
            "family": family,
            "model": model_name,
            "hf_name": hf_name,
            "category": category,
            "layer": layer,
            "head": head,
            "mean_ioi_influence": float(torch.tensor(ioi_influences).mean().item())
            if ioi_influences
            else 0.0,
            "mean_anti_influence": float(torch.tensor(anti_influences).mean().item())
            if anti_influences
            else 0.0,
            "ioi_influences": ioi_influences,
            "anti_influences": anti_influences,
        }
        results.append(result_row)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote causal tracing results to {args.out}")


if __name__ == "__main__":
    main()

