#!/usr/bin/env python

import os
import json
import csv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def ensure_dirs():
    os.makedirs("results/case_study_comparisons", exist_ok=True)
    os.makedirs("figs/case_study_comparisons", exist_ok=True)


def load_text_if_exists(path: str) -> str:
    if not os.path.exists(path):
        return f"[MISSING FILE: {path}]"
    with open(path, "r") as f:
        return f.read()


def make_attn_side_by_side_figure(
    base_png: str,
    tgt_png: str,
    base_label: str,
    tgt_label: str,
    out_path: str,
):
    """
    Load two attention heatmaps and save a side-by-side comparison figure.
    """
    if not os.path.exists(base_png):
        print(f"[WARN] Base attention PNG not found: {base_png}")
        return
    if not os.path.exists(tgt_png):
        print(f"[WARN] Target attention PNG not found: {tgt_png}")
        return

    base_img = mpimg.imread(base_png)
    tgt_img = mpimg.imread(tgt_png)

    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(base_img)
    ax1.set_title(base_label)
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(tgt_img)
    ax2.set_title(tgt_label)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OUT] Saved side-by-side attention plot to {out_path}")


def main():
    ensure_dirs()

    # ------------------------------------------------------------------
    # Define the head pairs we want to compare
    # ------------------------------------------------------------------
    pairs = [
        {
            "pair_id": "pythia410m_L5H13_vs_gpt2m_L0H4",
            "base_model": "pythia-410m",
            "base_layer": 5,
            "base_head": 13,
            "tgt_model": "gpt2-medium",
            "tgt_layer": 0,
            "tgt_head": 4,
        },
        {
            "pair_id": "pythia410m_L10H7_vs_gpt2m_L8H8",
            "base_model": "pythia-410m",
            "base_layer": 10,
            "base_head": 7,
            "tgt_model": "gpt2-medium",
            "tgt_layer": 8,
            "tgt_head": 8,
        },
        {
            "pair_id": "pythia410m_L15H11_vs_gpt2m_L12H6",
            "base_model": "pythia-410m",
            "base_layer": 15,
            "base_head": 11,
            "tgt_model": "gpt2-medium",
            "tgt_layer": 12,
            "tgt_head": 6,
        },
    ]

    summary_rows = []

    for p in pairs:
        pair_id = p["pair_id"]
        base_model = p["base_model"]
        base_layer = p["base_layer"]
        base_head = p["base_head"]
        tgt_model = p["tgt_model"]
        tgt_layer = p["tgt_layer"]
        tgt_head = p["tgt_head"]

        print(f"\n[PAIR] {pair_id}")

        # ------------------------------------------------------------------
        # Construct paths for base (Pythia)
        # ------------------------------------------------------------------
        base_dir = os.path.join(
            "results",
            "case_studies",
            f"{base_model}_L{base_layer}H{base_head}",
        )
        base_attn = os.path.join(
            base_dir,
            f"attn_pattern_L{base_layer}H{base_head}.png",
        )
        base_ov = os.path.join(
            base_dir,
            f"ov_analysis_L{base_layer}H{base_head}.txt",
        )
        base_dlogit = os.path.join(
            base_dir,
            f"delta_logit_L{base_layer}H{base_head}.txt",
        )

        # ------------------------------------------------------------------
        # Construct paths for target (GPT-2)
        # ------------------------------------------------------------------
        tgt_dir = os.path.join(
            "results",
            "case_studies",
            f"{tgt_model}_L{tgt_layer}H{tgt_head}",
        )
        tgt_attn = os.path.join(
            tgt_dir,
            f"attn_pattern_L{tgt_layer}H{tgt_head}.png",
        )
        tgt_ov = os.path.join(
            tgt_dir,
            f"ov_analysis_L{tgt_layer}H{tgt_head}.txt",
        )
        tgt_dlogit = os.path.join(
            tgt_dir,
            f"delta_logit_L{tgt_layer}H{tgt_head}.txt",
        )

        # ------------------------------------------------------------------
        # Make side-by-side attention figure
        # ------------------------------------------------------------------
        fig_out = os.path.join(
            "figs",
            "case_study_comparisons",
            f"{pair_id}_attn_side_by_side.png",
        )
        make_attn_side_by_side_figure(
            base_png=base_attn,
            tgt_png=tgt_attn,
            base_label=f"{base_model} L{base_layer}H{base_head}",
            tgt_label=f"{tgt_model} L{tgt_layer}H{tgt_head}",
            out_path=fig_out,
        )

        # ------------------------------------------------------------------
        # Load OV + delta-logit text for JSON summary
        # ------------------------------------------------------------------
        base_ov_text = load_text_if_exists(base_ov)
        base_dlogit_text = load_text_if_exists(base_dlogit)
        tgt_ov_text = load_text_if_exists(tgt_ov)
        tgt_dlogit_text = load_text_if_exists(tgt_dlogit)

        json_out = os.path.join(
            "results",
            "case_study_comparisons",
            f"{pair_id}_summary.json",
        )
        summary_obj = {
            "pair_id": pair_id,
            "base_model": base_model,
            "base_layer": base_layer,
            "base_head": base_head,
            "tgt_model": tgt_model,
            "tgt_layer": tgt_layer,
            "tgt_head": tgt_head,
            "paths": {
                "base_attn_png": base_attn,
                "base_ov_txt": base_ov,
                "base_delta_logit_txt": base_dlogit,
                "tgt_attn_png": tgt_attn,
                "tgt_ov_txt": tgt_ov,
                "tgt_delta_logit_txt": tgt_dlogit,
                "attn_side_by_side_png": fig_out,
            },
            "ov_analysis": {
                "base": base_ov_text,
                "target": tgt_ov_text,
            },
            "delta_logit_analysis": {
                "base": base_dlogit_text,
                "target": tgt_dlogit_text,
            },
        }

        with open(json_out, "w") as f:
            json.dump(summary_obj, f, indent=2)
        print(f"[OUT] Wrote JSON summary to {json_out}")

        # Row for LaTeX/CSV table
        summary_rows.append(
            [
                pair_id,
                base_model,
                base_layer,
                base_head,
                tgt_model,
                tgt_layer,
                tgt_head,
                fig_out,
                json_out,
            ]
        )

    # ----------------------------------------------------------------------
    # Write combined mechanistic comparison table
    # ----------------------------------------------------------------------
    table_out = os.path.join(
        "results",
        "case_study_comparisons",
        "combined_mechanistic_table.csv",
    )
    with open(table_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair_id",
                "base_model",
                "base_layer",
                "base_head",
                "tgt_model",
                "tgt_layer",
                "tgt_head",
                "attn_comparison_fig",
                "json_summary_path",
            ]
        )
        writer.writerows(summary_rows)

    print(f"\n[OUT] Wrote combined mechanistic table to {table_out}")


if __name__ == "__main__":
    main()

