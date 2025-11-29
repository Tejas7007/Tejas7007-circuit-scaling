from __future__ import annotations
from typing import Callable, List, Tuple, Dict, Any
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

def _zero_head_hook(layer_idx:int, head_idx:int):
    def hook(z: torch.Tensor, _):
        z = z.clone()
        z[:, :, head_idx, :] = 0.0
        return z
    return hook

def last_token_loss(model: HookedTransformer, tokens: torch.Tensor) -> float:
    """
    Cross-entropy only on the LAST position (per prompt), averaged across batch.
    """
    with torch.no_grad():
        logits = model(tokens, return_type="logits")        # [B, T, V]
    targets = tokens[:, 1:]                                 # [B, T-1]
    logits_shifted = logits[:, :-1, :]                      # [B, T-1, V]
    last_logits = logits_shifted[:, -1, :]                  # [B, V]
    last_targets = targets[:, -1]                           # [B]
    loss = F.cross_entropy(last_logits, last_targets, reduction="mean")
    return float(loss.item())

def head_delta_loss(
    model: HookedTransformer,
    prompts: list[str],
    layer_idx:int,
    head_idx:int,
    device:str="cpu",
) -> float:
    """
    Δloss on the FINAL token only: loss(ablated) - loss(baseline).
    Positive Δ => head helps; Negative Δ => head hurts/suppresses.
    """
    tokens = model.to_tokens(prompts, prepend_bos=True).to(device)

    base = last_token_loss(model, tokens)

    hook_name = f"blocks.{layer_idx}.attn.hook_result"
    hook_fn = _zero_head_hook(layer_idx, head_idx)
    with torch.no_grad():
        ablated_logits = model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=[(hook_name, hook_fn)],
        )
    targets = tokens[:, 1:]
    logits_shifted = ablated_logits[:, :-1, :]
    last_logits = logits_shifted[:, -1, :]
    last_targets = targets[:, -1]
    ablated_loss = F.cross_entropy(last_logits, last_targets, reduction="mean").item()

    return float(ablated_loss - base)
