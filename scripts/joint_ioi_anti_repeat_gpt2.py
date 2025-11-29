#!/usr/bin/env python
"""
Build a joint IOI vs anti-repeat head-level table for GPT-2 family.

Expected input files (in results/):
  - copy_suppression_logitdiff_scan_gpt2.json
  - copy_suppression_logitdiff_scan_gpt2-medium.json
  - anti_repeat_logitdiff_scan_gpt2.json
  - anti_repeat_logitdiff_scan_gpt2-medium.json

Output:
  - results/joint_ioi_anti_repeat_gpt2.csv

Each row = one (model, layer, head) with:
  - delta_ioi
  - delta_anti
"""

import json
import os
from typing import List, Dict, Any

import pandas as pd

RESULTS_DIR = "results"

IOI_FILES = {
    "gpt2": os.path.join(RESULTS_DIR, "copy_suppression_logitdiff_scan_gpt2.json"),
    "gpt2-medium": os.path.join(RESULTS_DIR, "copy_suppression_logitdiff_scan_gpt2-medium.json"),
}

ANTI_FILES = {
    "gpt2": os.path.join(RESULTS_DIR, "anti_repeat_logitdiff_scan_gpt2.json"),
    "gpt2-medium": os.path.join(RESULTS_DIR, "anti_repeat_logitdiff_scan_gpt2-medium.json"),
}


def _extract_heads(data: Any, path: str) -> List[Dict[str, Any]]:
    """
    Robustly pull out a list of head dicts from a scan JSON.

    We handle a few possibilities:
      - {"heads": [...]}                      # flat
      - {"model": {... "heads": [...]}}       # nested
      - {"top_suppressor_candidates": [...]}  # flat
      - {"model": {"top_suppressor_candidates": [...]}}  # nested

    For IOI JSONs you generated earlier, it's likely:
      { "gpt2": { "top_suppressor_candidates": [...] , ... } }
    """

    # Case 1: direct "heads"
    if isinstance(data, dict) and "heads" in data:
        heads = data["heads"]

    # Case 2: dict with a single top-level key, inner has "heads" or "top_suppressor_candidates"
    elif isinstance(data, dict) and len(data) == 1:
        inner = list(data.values())[0]
        if isinstance(inner, dict):
            if "heads" in inner:
                heads = inner["heads"]
            elif "top_suppressor_candidates" in inner:
                heads = inner["top_suppressor_candidates"]
            else:
                raise ValueError(f"Unrecognized nested JSON structure in {path}: keys={list(inner.keys())}")
        else:
            raise ValueError(f"Unrecognized nested JSON structure in {path}: inner is not a dict")

    # Case 3: flat dict with "top_suppressor_candidates"
    elif isinstance(data, dict) and "top_suppressor_candidates" in data:
        heads = data["top_suppressor_candidates"]

    else:
        raise ValueError(f"Unrecognized JSON structure in {path}: top-level keys={list(data.keys())}")

    if not isinstance(heads, list):
        raise ValueError(f"'heads' should be a list in {path}")

    return heads


def load_heads(path: str) -> List[Dict[str, Any]]:
    """
    Load per-head deltas from a scan JSON, returning list of
    {layer, head, delta}.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    raw_heads = _extract_heads(data, path)

    processed = []
    for h in raw_heads:
        # Try several key names for layer/head/delta
        layer = h.get("layer", h.get("layer_index", h.get("layer_idx")))
        head = h.get("head", h.get("head_index", h.get("head_idx")))
        delta = h.get("delta", h.get("delta_logit_diff", h.get("logit_diff_delta")))

        if layer is None or head is None or delta is None:
            print(f"[WARN] Skipping bad head entry in {path}: {h}")
            continue

        processed.append(
            {
                "layer": int(layer),
                "head": int(head),
                "delta": float(delta),
            }
        )

    return processed


def build_joint_table() -> pd.DataFrame:
    rows = []

    for model_name in sorted(IOI_FILES.keys()):
        ioi_path = IOI_FILES[model_name]
        anti_path = ANTI_FILES[model_name]

        print(f"Loading IOI heads for {model_name} from {ioi_path}")
        print(f"Loading anti-repeat heads for {model_name} from {anti_path}")

        ioi_heads = load_heads(ioi_path)
        anti_heads = load_heads(anti_path)

        # Map (layer, head) -> delta
        ioi_map = {(h["layer"], h["head"]): h["delta"] for h in ioi_heads}
        anti_map = {(h["layer"], h["head"]): h["delta"] for h in anti_heads}

        all_keys = set(ioi_map.keys()) | set(anti_map.keys())

        for (layer, head) in sorted(all_keys):
            rows.append(
                {
                    "model": model_name,
                    "layer": layer,
                    "head": head,
                    "delta_ioi": ioi_map.get((layer, head), 0.0),
                    "delta_anti": anti_map.get((layer, head), 0.0),
                }
            )

    df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "joint_ioi_anti_repeat_gpt2.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote joint GPT-2 IOI vs anti-repeat table to {out_path}")
    print(df.head())
    return df


if __name__ == "__main__":
    build_joint_table()

