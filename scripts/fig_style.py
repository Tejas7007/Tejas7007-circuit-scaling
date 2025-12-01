#!/usr/bin/env python3
"""
Common matplotlib style helpers for ICML-like figures.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_icml_style():
    """Set a reasonably ICML-like, single-column figure style."""
    mpl.rcdefaults()

    # Single-column ICML ~3.25in width -> ~3.3, height ~2.3–2.5
    mpl.rcParams.update({
        "figure.figsize": (3.3, 2.4),
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "pdf.fonttype": 42,  # editable text in Illustrator
        "ps.fonttype": 42,
    })

    # Nice but simple color cycle
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
        ]
    )


def savefig_icml(fig, basename: str):
    """
    Save both PDF and PNG for a figure.

    Args:
        fig: matplotlib.figure.Figure
        basename: path without extension, e.g. 'paper/figs/my_figure'
    """
    os.makedirs(os.path.dirname(basename), exist_ok=True)

    pdf_path = basename + ".pdf"
    png_path = basename + ".png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"[OUT] Saved figure to {pdf_path} and {png_path}")

