"""Shared matplotlib styling for CRISP-ML(Q) phase-01 artifacts."""

import matplotlib.pyplot as plt

PALETTE = {
    "bg": "#ffffff",
    "panel": "#f8f9fa",
    "grid": "#e5e7eb",
    "text": "#111827",
    "muted": "#6b7280",
    "accent1": "#4f46e5",
    "accent2": "#e11d48",
    "accent3": "#0891b2",
    "accent4": "#7c3aed",
    "accent5": "#ea580c",
    "green": "#059669",
    "gradient": ["#4f46e5", "#6d28d9", "#7c3aed", "#8b5cf6", "#a78bfa"],
}


def apply_pro_style():
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlepad": 14,
            "text.color": PALETTE["text"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
            "legend.facecolor": PALETTE["panel"],
            "legend.edgecolor": PALETTE["grid"],
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": PALETTE["bg"],
            "savefig.pad_inches": 0.3,
        }
    )
