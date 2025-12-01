#!/usr/bin/env python3
"""
plot_gpt2m_ioi_circuit_diagram.py

Create a schematic IOI circuit diagram for GPT-2-Medium using
the top path-patching units from:

  paper/tables/gpt2medium_ioi_circuit_components.csv

Saves:
  paper/figs/gpt2m_ioi_circuit_diagram.png
  paper/figs/gpt2m_ioi_circuit_diagram.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scripts.fig_style import set_icml_style, savefig_icml


def assign_stage(layer: int) -> str:
    """
    Map layer index (0-23) to coarse stage: early / mid / late.
    Adjust thresholds if you like.
    """
    if layer <= 7:
        return "Early"
    elif layer <= 15:
        return "Mid"
    else:
        return "Late"


def main():
    set_icml_style()

    csv_path = "paper/tables/gpt2medium_ioi_circuit_components.csv"
    df = pd.read_csv(csv_path)

    # Ensure required columns
    required = ["type", "layer", "head", "unit_label", "patch_effect", "abs_patch_effect"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Compute stage
    df["stage"] = df["layer"].apply(assign_stage)

    # Normalize sizes for plotting
    # Add a small epsilon to avoid zeros
    eps = 1e-6
    max_abs = df["abs_patch_effect"].max() + eps
    df["size_norm"] = df["abs_patch_effect"] / max_abs

    # We’ll map size_norm in [0,1] to marker size in points^2
    min_size = 40
    max_size = 300
    df["marker_size"] = min_size + df["size_norm"] * (max_size - min_size)

    # Color by sign of patch_effect
    # > 0: positive (patching moves logits toward clean)
    # < 0: negative
    cmap_pos = "#1f77b4"  # blue
    cmap_neg = "#ff7f0e"  # orange
    df["color"] = df["patch_effect"].apply(lambda v: cmap_pos if v >= 0 else cmap_neg)

    # Layout:
    # x-axis: stage (Early, Mid, Late)
    # y-axis: layer index (0 at top, 23 at bottom)
    stage_order = ["Early", "Mid", "Late"]
    stage_x = {stage: ix for ix, stage in enumerate(stage_order)}

    # Jitter X slightly so attention and mlp units in same layer don't overlap perfectly
    def x_pos(row):
        base = stage_x[row["stage"]]
        if row["type"].strip() == "attn":
            return base - 0.15  # left
        else:
            return base + 0.15  # right

    df["x"] = df.apply(x_pos, axis=1)
    # Invert y so layer 0 is at top
    df["y"] = -df["layer"]

    # Prepare figure: this one can be a bit wider than a single column
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Draw "background bands" for early / mid / late for visual grouping
    ymin = -(df["layer"].max() + 1)
    ymax = 1
    for stage in stage_order:
        x_center = stage_x[stage]
        ax.axvspan(x_center - 0.5, x_center + 0.5,
                   facecolor="#f0f0f0", alpha=0.4, zorder=0)
        ax.text(x_center, ymax + 0.5, stage,
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Plot attention and MLP separately so we can use different markers
    attn = df[df["type"].str.strip() == "attn"]
    mlp = df[df["type"].str.strip() == "mlp"]

    # Attention heads: circles
    ax.scatter(
        attn["x"], attn["y"],
        s=attn["marker_size"],
        c=attn["color"],
        edgecolor="black",
        linewidth=0.4,
        marker="o",
        label="Attention heads",
        zorder=3,
        alpha=0.9,
    )

    # MLP blocks: squares
    ax.scatter(
        mlp["x"], mlp["y"],
        s=mlp["marker_size"],
        c=mlp["color"],
        edgecolor="black",
        linewidth=0.4,
        marker="s",
        label="MLP blocks",
        zorder=3,
        alpha=0.9,
    )

    # Optional: lightly connect stages with "flow arrows" (schematic only)
    # We won't encode extra data here; it's just to suggest early→mid→late IOI flow.
    for i in range(len(stage_order) - 1):
        x0 = stage_x[stage_order[i]]
        x1 = stage_x[stage_order[i + 1]]
        ax.annotate(
            "",
            xy=(x1 - 0.6, ymin + 0.5),
            xytext=(x0 + 0.6, ymin + 0.5),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.0,
                color="black",
                alpha=0.6,
            ),
            zorder=1,
        )

    # Axes formatting
    ax.set_xlim(-0.7, len(stage_order) - 0.3)
    ax.set_ylim(ymin - 0.5, ymax + 1.0)

    ax.set_xticks([stage_x[s] for s in stage_order])
    ax.set_xticklabels(stage_order)

    # y-axis labeled by layer index
    layers_to_show = list(range(0, 24, 4))
    ax.set_yticks([-l for l in layers_to_show])
    ax.set_yticklabels(layers_to_show)
    ax.set_ylabel("Layer index (0 = earliest)")

    ax.set_title("GPT-2 Medium IOI Circuit (top path-patching units)")

    # Legend for marker type and color meaning
    # We'll make a custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Attention head",
               markerfacecolor="white", markeredgecolor="black", markersize=6),
        Line2D([0], [0], marker="s", color="w", label="MLP block",
               markerfacecolor="white", markeredgecolor="black", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Patch improves IOI",
               markerfacecolor=cmap_pos, markeredgecolor="black", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="Patch hurts IOI",
               markerfacecolor=cmap_neg, markeredgecolor="black", markersize=6),
    ]

    ax.legend(handles=legend_elements, loc="lower right", frameon=False)

    fig.tight_layout()
    savefig_icml(fig, "paper/figs/gpt2m_ioi_circuit_diagram")
    plt.close(fig)


if __name__ == "__main__":
    main()

