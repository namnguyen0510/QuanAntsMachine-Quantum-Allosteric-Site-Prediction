"""
Visualization utilities — 3D structure overlays, comparison bars, recall curves.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from quanallo.data.schemas import ProteinGraph

# Color palette by method kind
KIND_COLORS = {
    "quantum":          "#c1272d",   # red
    "quantum_inspired": "#f15a29",   # orange-red
    "classical":        "#5e6c75",   # gray
    "hybrid":           "#1f4e79",   # blue
    "quanant":          "#7b3294",   # purple
    "ensemble":         "#2a9d8f",   # teal
}


def plot_scores_3d(
    graph: ProteinGraph,
    scores: np.ndarray,
    top_indices: Optional[Sequence[int]] = None,
    title: str = "Allosteric scores",
    ax=None,
    save_path: Optional[str | Path] = None,
    show_active: bool = True,
    show_gt: bool = True,
):
    """
    Visualize per-residue scores in 3D, overlaid on the Cα structure.

    - Background residues: colored by score (viridis colormap).
    - Active site: red circles.
    - Ground truth (if available): gold stars.
    - Top-k predictions: magenta crosses.
    """
    xyz = graph.coords
    N = graph.N
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

    s = np.asarray(scores, dtype=float)
    s_norm = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)

    other = np.ones(N, dtype=bool)
    if show_active:
        other[graph.active_idx] = False
    if show_gt and graph.ground_truth_idx is not None:
        other[graph.ground_truth_idx] = False

    sc = ax.scatter(
        xyz[other, 0], xyz[other, 1], xyz[other, 2],
        c=s_norm[other], cmap="viridis", s=42, alpha=0.85,
        edgecolors="k", linewidths=0.3,
    )
    if show_active:
        ax.scatter(
            xyz[graph.active_idx, 0], xyz[graph.active_idx, 1], xyz[graph.active_idx, 2],
            c="red", s=120, marker="o", edgecolors="black", linewidths=0.8,
            label="Active site",
        )
    if show_gt and graph.ground_truth_idx is not None:
        ax.scatter(
            xyz[graph.ground_truth_idx, 0],
            xyz[graph.ground_truth_idx, 1],
            xyz[graph.ground_truth_idx, 2],
            c="gold", s=140, marker="*", edgecolors="black", linewidths=0.6,
            label="Ground-truth pocket",
        )
    if top_indices:
        top = np.asarray(top_indices, dtype=int)
        ax.scatter(
            xyz[top, 0], xyz[top, 1], xyz[top, 2],
            c="none", s=300, marker="o", edgecolors="magenta", linewidths=2.5,
            label="Top-k predictions",
        )

    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
    return ax


def plot_comparison_bars(
    results: dict,
    metric: str = "weighted_top5",
    title: str = "Method comparison",
    method_kinds: Optional[dict] = None,
    baseline_lines: Optional[dict] = None,
    save_path: Optional[str | Path] = None,
    ax=None,
):
    """
    Bar chart of a single metric across methods, color-coded by kind.

    Parameters
    ----------
    results : dict of {method_name: PredictionResult or {metric: value}}
        Predictions to compare.
    metric : str
        Attribute / key of each result to plot (default ``weighted_top5``).
    method_kinds : dict, optional
        Map method_name -> kind ('quantum', 'classical', ...). Used for coloring.
    baseline_lines : dict, optional
        Map label -> y-value. Draw horizontal reference lines.
    """
    names = list(results.keys())
    vals = []
    for n, r in results.items():
        if hasattr(r, metric):
            vals.append(getattr(r, metric))
        elif isinstance(r, dict) and metric in r:
            vals.append(r[metric])
        else:
            vals.append(0.0)
    colors = []
    for n in names:
        kind = (method_kinds or {}).get(n, "quantum")
        colors.append(KIND_COLORS.get(kind, "#888"))

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
    x = np.arange(len(names))
    ax.bar(x, vals, color=colors, alpha=0.92, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if baseline_lines:
        for label, y in baseline_lines.items():
            ax.axhline(y, ls=":", color="gray", alpha=0.7)
            ax.text(len(names) - 0.5, y + 0.02, label,
                    ha="right", va="bottom", fontsize=8, color="gray")
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=c, label=k) for k, c in KIND_COLORS.items()
                    if k in (method_kinds or {}).values()]
    if legend_items:
        ax.legend(handles=legend_items, fontsize=8, loc="best")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
    return ax


def plot_precision_curves(
    results: dict,
    title: str = "Precision@k vs k-hop tolerance",
    method_kinds: Optional[dict] = None,
    save_path: Optional[str | Path] = None,
    ax=None,
):
    """Plot precision@5 curves vs k-hop tolerance for each method.

    Each result must expose a ``precision_at_k`` dict (or attribute).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        p = getattr(r, "precision_at_k", None)
        if p is None and isinstance(r, dict):
            p = r.get("precision_at_k")
        if p is None:
            continue
        ks = sorted(p.keys())
        ys = [p[k] for k in ks]
        kind = (method_kinds or {}).get(name, "quantum")
        ax.plot(ks, ys, marker="o", color=KIND_COLORS.get(kind, "#888"),
                linewidth=2, label=name)
    ax.set_xlabel("k-hop tolerance to ground truth")
    ax.set_ylabel("Precision@5")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
    return ax


def plot_species_contribution(
    contributions: dict[str, np.ndarray],
    graph: ProteinGraph,
    top_n: int = 30,
    title: str = "Per-species contribution to pheromone",
    save_path: Optional[str | Path] = None,
    ax=None,
):
    """Stacked bar of cumulative pheromone deposits per species per residue.

    The top-N residues by total deposit are shown. GT residues are bold-green
    in the x-axis tick labels.
    """
    species = list(contributions.keys())
    total = sum(contributions.values())
    top_idx = np.argsort(total)[::-1][:top_n]
    resnums = [graph.residue_at(i)["resnum"] for i in top_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(10, top_n * 0.45), 5))
    bottom = np.zeros(len(top_idx))
    x = np.arange(len(top_idx))
    palette = ["#c1272d", "#d62728", "#8b1a1a", "#e07b39", "#a83232",
               "#1f4e79", "#7b3294", "#2a9d8f"]
    for i, sp in enumerate(species):
        vals = contributions[sp][top_idx]
        ax.bar(x, vals, bottom=bottom, color=palette[i % len(palette)],
               label=sp, edgecolor="black", linewidth=0.2)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(resnums, fontsize=8)
    ax.set_xlabel(f"Residue resnum (top-{top_n} by deposit)")
    ax.set_ylabel("Cumulative deposit")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper right")

    # Highlight GT residues
    if graph.ground_truth_idx is not None:
        gt_resnums = {graph.residue_at(i)["resnum"]
                       for i in graph.ground_truth_idx}
        for i, rn in enumerate(resnums):
            if rn in gt_resnums:
                ax.get_xticklabels()[i].set_color("darkgreen")
                ax.get_xticklabels()[i].set_fontweight("bold")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
    return ax
