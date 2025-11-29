import json
import os
from typing import Dict, Tuple, List

import pandas as pd


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

MODEL_CONFIGS = [
    # Pythia family
    {
        "family": "pythia",
        "model": "pythia-70m",
        "ioi_file": "results/copy_suppression_logitdiff_scan_pythia-70m.json",
        "anti_file": "results/anti_repeat_logitdiff_scan_pythia-70m.json",
    },
    {
        "family": "pythia",
        "model": "pythia-160m",
        "ioi_file": "results/copy_suppression_logitdiff_scan_pythia-160m.json",
        "anti_file": "results/anti_repeat_logitdiff_scan_pythia-160m.json",
    },
    {
        "family": "pythia",
        "model": "pythia-410m",
        "ioi_file": "results/copy_suppression_logitdiff_scan_pythia-410m.json",
        "anti_file": "results/anti_repeat_logitdiff_scan_pythia-410m.json",
    },
    {
        "family": "pythia",
        "model": "pythia-1b",
        "ioi_file": "results/copy_suppression_logitdiff_scan_pythia-1b.json",
        "anti_file": "results/anti_repeat_logitdiff_scan_pythia-1b.json",
    },

    # GPT-Neo
    {
        "family": "gpt-neo",
        "model": "gpt-neo-125M",
        "ioi_file": "results/copy_suppression_logitdiff_scan_gpt-neo-125M.json",
        "anti_file": "results/anti_repeat_logitdiff_scan_gpt-neo-125M.json",
    },

    # GPT-2 family
    {
        "family": "gpt2",
        "model": "gpt2",
        "ioi_file": "results/copy_suppression_logitdiff_scan_gpt2.json",
        "anti_file": "results/anti_repeat_logitdiff_scan_gpt2.json",
    },
    {
        "family": "gpt2",
        "model": "gpt2-medium",
        "ioi_file": "results/copy_suppression_logitdiff_scan_gpt2-medium.json",
        "anti_file": "results/anti_repeat_logitdiff_scan_gpt2-medium.json",
    },
    {
        "family": "gpt2",
        "model": "gpt2-large",
        "ioi_file": "results/copy_suppression_logitdiff_scan_gpt2-large.json",
        "anti_file": "results/anti_repeat_logitdiff_scan_gpt2-large.json",
    },

    # OPT
    {
        "family": "opt",
        "model": "opt-125m",
        "ioi_file": "results/copy_suppression_logitdiff_scan_opt-125m.json",
        "anti_file": "results/anti_repeat_logitdiff_scan_opt-125m.json",
    },
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _load_payload(path: str) -> Dict:
    """Load a JSON file and return the inner dict with the actual results.

    Expected patterns:
      { "<model_name>": { ... results ... } }
    or:
      { "top_suppressor_candidates": [ ... ] }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Case 1: already looks like a payload
    if isinstance(data, dict) and "top_suppressor_candidates" in data:
        return data

    # Case 2: top-level keyed by model name
    if isinstance(data, dict):
        if len(data) != 1:
            print(
                f"[WARN] Multiple top-level keys in {path}, "
                f"using the first one: {list(data.keys())[0]}"
            )
        key = next(iter(data.keys()))
        payload = data[key]
        if isinstance(payload, dict):
            return payload

    raise ValueError(f"Unrecognized JSON structure in {path}")


def _extract_head_deltas(payload: Dict) -> Dict[Tuple[int, int], float]:
    """Return mapping (layer, head) -> delta from the payload.

    We support multiple possible key names for the delta value:
      - 'delta_logit_diff' (our preferred)
      - 'delta' (fallback, in case some scripts wrote this)
    """
    heads_list = payload.get("top_suppressor_candidates", [])
    if not isinstance(heads_list, list):
        raise ValueError("Payload 'top_suppressor_candidates' is not a list")

    out: Dict[Tuple[int, int], float] = {}
    for h in heads_list:
        layer = int(h["layer"])
        head = int(h["head"])

        if "delta_logit_diff" in h:
            delta = float(h["delta_logit_diff"])
        elif "delta" in h:
            delta = float(h["delta"])
        else:
            # If neither key exists, just skip this head with a warning
            print(
                f"[WARN] Head entry missing delta field: {h}. "
                f"Expected 'delta_logit_diff' or 'delta'. Skipping."
            )
            continue

        out[(layer, head)] = delta

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    rows: List[Dict] = []

    for cfg in MODEL_CONFIGS:
        family = cfg["family"]
        model = cfg["model"]
        ioi_file = cfg["ioi_file"]
        anti_file = cfg["anti_file"]

        print(f"Processing {family}/{model}")

        ioi_payload = _load_payload(ioi_file)
        anti_payload = _load_payload(anti_file)

        ioi_map = _extract_head_deltas(ioi_payload)
        anti_map = _extract_head_deltas(anti_payload)

        # union of heads that appear in either IOI or anti-repeat
        all_heads = sorted(set(ioi_map.keys()) | set(anti_map.keys()))

        for (layer, head) in all_heads:
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "layer": layer,
                    "head": head,
                    "delta_ioi": ioi_map.get((layer, head), 0.0),
                    "delta_anti": anti_map.get((layer, head), 0.0),
                }
            )

    df = pd.DataFrame(rows, columns=["family", "model", "layer", "head", "delta_ioi", "delta_anti"])
    out_path = os.path.join("results", "joint_ioi_anti_repeat_all.csv")
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote unified joint table to {out_path}")


if __name__ == "__main__":
    main()

