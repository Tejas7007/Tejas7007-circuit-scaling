from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from transformer_lens import HookedTransformer
from .patching import head_delta_loss

def scan_heads_delta_loss(
    model: HookedTransformer,
    prompts: List[str],
    device:str="cpu",
    max_layers:int|None=None,
) -> List[Tuple[Tuple[int,int], float]]:
    """
    For each head (layer, head), compute Δloss = loss(ablated) - loss(baseline).
    Returns a list sorted by Δloss ascending (most negative first).
    """
    n_layers = model.cfg.n_layers if max_layers is None else min(max_layers, model.cfg.n_layers)
    n_heads = model.cfg.n_heads
    results: List[Tuple[Tuple[int,int], float]] = []

    for layer in range(n_layers):
        for head in range(n_heads):
            try:
                d = head_delta_loss(model, prompts, layer, head, device=device)
            except RuntimeError as e:
                # In case of OOM or other transient errors, skip
                d = float('nan')
            results.append(((layer, head), d))

    # sort: negative (potential suppressors) at top
    results.sort(key=lambda x: (float('inf') if (x[1] != x[1]) else x[1]))  # NaNs sink to end
    return results
