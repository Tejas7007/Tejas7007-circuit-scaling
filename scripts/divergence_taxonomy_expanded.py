#!/usr/bin/env python3
"""
divergence_taxonomy_expanded.py

Expanded taxonomy for IOI head transfer Pythia-410M -> GPT-2-Medium.

Taxonomy labels (per base IOI head in Pythia-410M):
- missing_in_target   : aligned head exists but target_abs_delta_ioi < tau
- lost                : aligned head IOI-like but |delta_ioi_tgt| << |delta_ioi_src|
- inverted            : sign(delta_src) != sign(delta_tgt)
- weak                : IOI-like but |delta_ioi_tgt| < weak_tau
- relocated           : IOI-like head appears in nearby layers in GPT-2-Medium
- diffused            : many weak IOI-ish heads nearby, none strong
- other               : fallback

Outputs:
    results/ioi_divergence_taxonomy_expanded.csv
"""

import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "paper" / "tables" / "ioi_transfer_pythia410m_to_gpt2medium.csv"
JOINT = ROOT / "paper" / "tables" / "joint_ioi_anti_repeat_heads.csv"

OUT = ROOT / "results" / "ioi_divergence_taxonomy_expanded.csv"
OUT.parent.mkdir(exist_ok=True)

# Thresholds (match rest of paper)
TAU_IOI = 0.015      # IOI-like threshold
WEAK_TAU = 0.015     # "weak but IOI-like" boundary
RELOC_RADIUS = 2     # +/- layers for relocated
DIFFUSED_RADIUS = 1  # +/- layers for diffused


def main():
    base_df = pd.read_csv(BASE)
    joint = pd.read_csv(JOINT)

    print("[INFO] Columns in joint_ioi_anti_repeat_heads.csv:", list(joint.columns))

    # We assume these columns exist in joint:
    # ['family', 'model', 'layer', 'head', 'delta_ioi', 'abs_delta_ioi', ...]
    required_cols = ["family", "model", "layer", "head", "delta_ioi", "abs_delta_ioi"]
    for c in required_cols:
        if c not in joint.columns:
            raise ValueError(f"Expected column '{c}' missing from {JOINT}")

    rows = []

    # Base IOI-like heads: Pythia-410M, abs_delta_ioi >= TAU_IOI
    src = joint[
        (joint["family"] == "pythia")
        & (joint["model"] == "pythia-410m")
        & (joint["abs_delta_ioi"] >= TAU_IOI)
    ].copy()

    print(f"[INFO] Found {len(src)} IOI-like heads in Pythia-410m using abs_delta_ioi >= {TAU_IOI}")

    for _, row in src.iterrows():
        L = int(row["layer"])
        H = int(row["head"])
        delta_src = float(row["delta_ioi"])
        abs_src = abs(delta_src)

        # Find aligned entry in ioi_transfer_pythia410m_to_gpt2medium.csv
        tgt_match = base_df[(base_df["layer"] == L) & (base_df["head"] == H)]
        if len(tgt_match) == 0:
            taxonomy = "missing_in_target"
            rows.append({"layer": L, "head": H, "taxonomy": taxonomy})
            continue

        tgt_row = tgt_match.iloc[0]
        delta_tgt = float(tgt_row["target_delta_ioi"])
        abs_tgt = abs(delta_tgt)

        # 1) Missing in target by IOI-threshold
        if tgt_row["target_abs_delta_ioi"] < TAU_IOI:
            taxonomy = "missing_in_target"
            rows.append({"layer": L, "head": H, "taxonomy": taxonomy})
            continue

        # From here on, target_abs_delta_ioi >= TAU_IOI (IOI-like in target)

        # 2) Lost: magnitude collapses
        if abs_tgt < 0.3 * abs_src:
            taxonomy = "lost"
            rows.append({"layer": L, "head": H, "taxonomy": taxonomy})
            continue

        # 3) Inverted: sign flip
        if np.sign(delta_src) != np.sign(delta_tgt):
            taxonomy = "inverted"
            rows.append({"layer": L, "head": H, "taxonomy": taxonomy})
            continue

        # 4) Weak: IOI-like but small magnitude
        if abs_tgt < WEAK_TAU:
            taxonomy = "weak"
            rows.append({"layer": L, "head": H, "taxonomy": taxonomy})
            continue

        # 5) Relocated: IOI-like head nearby in GPT-2-Medium within +/- RELOC_RADIUS layers
        tgt_ioi = joint[
            (joint["family"] == "gpt2")
            & (joint["model"] == "gpt2-medium")
            & (joint["abs_delta_ioi"] >= TAU_IOI)
            & (joint["layer"].between(L - RELOC_RADIUS, L + RELOC_RADIUS))
        ]

        if len(tgt_ioi) > 0:
            taxonomy = "relocated"
            rows.append({"layer": L, "head": H, "taxonomy": taxonomy})
            continue

        # 6) Diffused: many weak IOI-ish heads nearby
        tgt_weak = joint[
            (joint["family"] == "gpt2")
            & (joint["model"] == "gpt2-medium")
            & (joint["abs_delta_ioi"].between(0.005, TAU_IOI))
            & (joint["layer"].between(L - DIFFUSED_RADIUS, L + DIFFUSED_RADIUS))
        ]

        if len(tgt_weak) >= 4:
            taxonomy = "diffused"
            rows.append({"layer": L, "head": H, "taxonomy": taxonomy})
            continue

        # 7) Fallback
        taxonomy = "other"
        rows.append({"layer": L, "head": H, "taxonomy": taxonomy})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"[OUT] Wrote expanded taxonomy to {OUT}")
    print("[COUNTS]")
    print(out["taxonomy"].value_counts())


if __name__ == "__main__":
    main()

