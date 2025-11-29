
from __future__ import annotations
from typing import List, Tuple
import random

_SINGLE_TOKENISH_NAMES = [
    "John","Mary","Tom","Kate","Raj","Liam","Mia","Omar","Lina","Chen","Sara","Ivy","Alex","Noah","Zoe"
]

def ioi_prompts(n:int=1000) -> List[str]:
    """
    IOI-style prompts that end *right before a name should appear next*.
    Example: "When John and Mary went to the store, John gave a gift to"
    """
    tpls = [
        "When {A} and {B} went to the store, {A} gave a gift to",
        "{A} wrote an email to {B}, and later {A} forwarded it to",
        "{A} met {B} at the cafe. After a while, {A} waved at",
        "Yesterday, {A} thanked {B}. Today, {A} sent a message to",
    ]
    prompts = []
    for _ in range(n):
        A, B = random.sample(_SINGLE_TOKENISH_NAMES, 2)
        tpl = random.choice(tpls)
        prompts.append(tpl.format(A=A, B=B))
    return prompts

def ioi_name_pairs(n:int=1000) -> List[Tuple[str,str]]:
    """Return (A,B) pairs used to build prompts; useful for computing logit-diff (correct=B, distractor=A)."""
    pairs = []
    for _ in range(n):
        A, B = random.sample(_SINGLE_TOKENISH_NAMES, 2)
        pairs.append((A,B))
    return pairs
