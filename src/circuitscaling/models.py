from __future__ import annotations
from typing import Optional, List
from transformer_lens import HookedTransformer

MODEL_ALIASES = {
    "pythia-70m": "EleutherAI/pythia-70m-deduped",
    "pythia-160m": "EleutherAI/pythia-160m-deduped",
    "pythia-410m": "EleutherAI/pythia-410m-deduped",
    "pythia-1b": "EleutherAI/pythia-1b-deduped",
    "pythia-2.8b": "EleutherAI/pythia-2.8b-deduped",
    "pythia-6.9b": "EleutherAI/pythia-6.9b-deduped",
    "pythia-12b": "EleutherAI/pythia-12b-deduped",
    "gpt2-small": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
}

def load_model(name:str, device:str="cuda" if False else "cpu") -> HookedTransformer:
    hf_name = MODEL_ALIASES.get(name, name)
    model = HookedTransformer.from_pretrained(hf_name, device=device)
    model.eval()
    return model
