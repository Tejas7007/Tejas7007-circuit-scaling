from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AttributionResult:
    head: tuple
    delta_loss: float
    method: str
