"""
Core utilities: graph hop distances, fuzzy ground-truth credit, evaluation.
"""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg


# ----------------------------------------------------------------------
# Graph utilities
# ----------------------------------------------------------------------
def hop_distances(adjacency: np.ndarray, sources: Sequence[int], N: int) -> np.ndarray:
    """Shortest unweighted hop distance from each node to its nearest source.

    Parameters
    ----------
    adjacency : (N, N) array
        Any adjacency (binary or weighted) — only non-zero entries count.
    sources : list of int
        Source-node indices.
    N : int
        Total number of nodes.

    Returns
    -------
    (N,) array of int. Nodes unreachable from any source get ``N+1``.
    """
    G = sp.csr_matrix((adjacency > 0).astype(int))
    sources = np.asarray(sources, dtype=int)
    if len(sources) == 0:
        return np.full(N, N + 1, dtype=int)
    d = csg.dijkstra(G, indices=sources, unweighted=True).min(axis=0)
    return np.where(np.isfinite(d), d, N + 1).astype(int)


def sym_norm_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Symmetric normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)."""
    d = adjacency.sum(axis=1)
    s = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    return np.eye(adjacency.shape[0]) - (s[:, None] * adjacency * s[None, :])


# ----------------------------------------------------------------------
# Fuzzy ground-truth scoring
# ----------------------------------------------------------------------
def fuzzy_gt_credit(adjacency: np.ndarray, gt_idx: Sequence[int], N: int,
                    alpha: float = np.log(2)) -> np.ndarray:
    """Fuzzy GT credit per residue: exp(-α · hop_to_gt).

    With α = ln 2, the hop-decay table is:

        hop=0 : 1.0
        hop=1 : 0.5
        hop=2 : 0.25
        hop=3 : 0.125
        hop=4 : 0.0625

    The "weighted top-5" metric is the sum of credits of the 5 predicted residues
    — so an ideal predictor with all 5 in the GT scores 5.0; a predictor that
    misses by one hop on each scores 2.5; etc.
    """
    h = hop_distances(adjacency, gt_idx, N)
    return np.exp(-alpha * h)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(
    scores: np.ndarray,
    adjacency: np.ndarray,
    gt_idx: Sequence[int],
    active_idx: Sequence[int],
    N: int,
    *,
    top_k: int = 5,
    alpha: float = np.log(2),
    mask_radius: int = 0,
    surface_mask: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
    selection: str = "argmax",
    lambda_div: float = 0.4,
) -> dict:
    """
    Score a per-residue prediction vector against a ground-truth pocket.

    Parameters
    ----------
    scores : (N,) array
        Per-residue prediction (higher = better).
    adjacency : (N, N) array
        Contact graph (for hop-distance computation).
    gt_idx : list of int
        Ground-truth pocket residue indices.
    active_idx : list of int
        Active-site residue indices (used for masking).
    N : int
        Number of nodes.
    top_k : int
        Cardinality of the prediction set.
    alpha : float
        Decay rate for fuzzy GT credit (default ln 2 → halving per hop).
    mask_radius : int
        Mask residues within this many hops of any active-site residue.
        ``0`` masks only the active-site itself; ``-1`` disables masking.
    surface_mask : (N,) bool array, optional
        If given, restrict candidates to surface residues.
    coords : (N, 3) array, optional
        Required if ``selection='mmr'``.
    selection : {"argmax", "mmr"}
        Top-k selection strategy.
    lambda_div : float
        Diversity weight for MMR (0 = pure score, 1 = pure diversity).

    Returns
    -------
    dict with keys:
        top_pred : list of int (chosen residue indices)
        hops_to_gt : list of int
        credit_of_top : list of float (per-residue credits)
        weighted_top5 : float (sum of credits; max = top_k)
        precision_at_5 : dict[k_hop] -> precision (fraction of hits within k hops)
    """
    s = np.asarray(scores, dtype=float).copy()
    active_idx = np.asarray(active_idx, dtype=int)
    if mask_radius >= 0 and len(active_idx) > 0:
        ah = hop_distances(adjacency, active_idx, N)
        s[ah <= mask_radius] = -np.inf
    if surface_mask is not None:
        s[~np.asarray(surface_mask, dtype=bool)] = -np.inf

    if selection == "argmax":
        from quanallo.core.selection import argmax_top_k
        top_pred = argmax_top_k(s, k=top_k)
    elif selection == "mmr":
        if coords is None:
            raise ValueError("coords required for MMR selection")
        from quanallo.core.selection import mmr_top_k
        top_pred = mmr_top_k(s, coords, k=top_k, lambda_div=lambda_div)
    else:
        raise ValueError(f"unknown selection={selection!r}")

    credit = fuzzy_gt_credit(adjacency, gt_idx, N, alpha=alpha)
    h2g = hop_distances(adjacency, gt_idx, N)
    return {
        "top_pred": top_pred,
        "hops_to_gt": [int(h2g[p]) for p in top_pred],
        "credit_of_top": [float(credit[p]) for p in top_pred],
        "weighted_top5": float(credit[top_pred].sum()),
        "precision_at_5": {
            k: float(np.mean(h2g[top_pred] <= k)) for k in range(6)
        },
    }
