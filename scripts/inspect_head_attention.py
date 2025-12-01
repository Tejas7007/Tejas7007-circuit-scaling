import argparse
import torch

from transformer_lens import HookedTransformer


def build_prompts():
    """
    Simple, hand-crafted prompts for the two behaviors.

    IOI-style: ambiguous pronoun / object choice where copy suppression matters.
    Anti-repeat: contexts where repeating the previous token is bad.
    """
    ioi_prompts = [
        "When John and Mary went to the store, John gave a book to",
        "Alice and Bob walked into the office. Alice thanked",
        "Sarah met David at the park, and Sarah spoke to",
        "Tom and Emma were arguing because Tom insulted",
    ]

    anti_repeat_prompts = [
        "The word is cat, not",
        "The answer is blue, not",
        "He said hello, not",
        "The password is secret, not",
    ]

    return ioi_prompts, anti_repeat_prompts


def print_head_pattern(model, toks, layer, head, label, k=7):
    """
    For a batch of prompts, print the top-k attention targets
    of (layer, head) for the *last* token in each sequence.
    """
    _, cache = model.run_with_cache(
        toks,
        names_filter=[f"blocks.{layer}.attn.hook_pattern"],
        return_type="logits",
    )
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # [batch, heads, q, k]
    # take the last query position (q = last token)
    head_pattern = pattern[:, head, -1, :]  # [batch, seq]

    print(f"\n=== {label}: layer {layer}, head {head} ===")
    for i in range(head_pattern.shape[0]):
        attn = head_pattern[i]  # [seq]
        # Get top-k attention positions
        k_eff = min(k, attn.shape[0])
        vals, idxs = torch.topk(attn, k=k_eff)
        vals = vals.tolist()
        idxs = idxs.tolist()

        # Get string tokens for this prompt
        # (re-tokenize to keep it simple)
        # Note: prepend_bos=True to match toks
        prompt_str = model.to_str_tokens(
            model.to_string(toks[i]),
            prepend_bos=True,
        )

        print(f"\nPrompt {i+1}:")
        print("  Text:", model.to_string(toks[i]))
        print("  Top attention targets for last token:")
        for pos, weight in zip(idxs, vals):
            tok_str = prompt_str[pos] if pos < len(prompt_str) else "<out-of-range>"
            print(f"    pos {pos:2d}  token={tok_str!r}  attn={weight:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="e.g. pythia-70m")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--k", type=int, default=7, help="top-k attention positions to print")
    args = parser.parse_args()

    model_name = f"EleutherAI/{args.model}-deduped"

    print(f"Loading {model_name} on {args.device} ...")
    model = HookedTransformer.from_pretrained(model_name, device=args.device)

    ioi_prompts, anti_prompts = build_prompts()

    # IOI batch
    ioi_toks = model.to_tokens(ioi_prompts, prepend_bos=True)
    print_head_pattern(model, ioi_toks, args.layer, args.head, label="IOI", k=args.k)

    # Anti-repeat batch
    anti_toks = model.to_tokens(anti_prompts, prepend_bos=True)
    print_head_pattern(model, anti_toks, args.layer, args.head, label="Anti-repeat", k=args.k)


if __name__ == "__main__":
    main()


