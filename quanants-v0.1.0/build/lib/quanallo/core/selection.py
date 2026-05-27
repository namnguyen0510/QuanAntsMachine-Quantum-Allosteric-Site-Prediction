"""
Top-k selection strategies.

- :func:`argmax_top_k` — pure greedy by score (allows clustering).
- :func:`mmr_top_k` — Maximum Marginal Relevance with 3D-distance diversity.
"""
from __future__ import annotations
from typing import Optional, Set

import numpy as np


def argmax_top_k(scores: np.ndarray, k: int = 5) -> list[int]:
    """Return the indices of the top-k scores (descending), skipping -inf."""
    s = np.asarray(scores, dtype=float)
    order = np.argsort(s)[::-1]
    out = []
    for i in order:
        if not np.isfinite(s[i]):
            break
        out.append(int(i))
        if len(out) == k:
            break
    return out


def mmr_top_k(
    scores: np.ndarray,
    coords: np.ndarray,
    k: int = 5,
    lambda_div: float = 0.4,
    distance_norm: float = 20.0,
    banned: Optional[Set[int]] = None,
) -> list[int]:
    """
    Maximum Marginal Relevance: greedy selection that balances relevance with
    spatial diversity in 3D.

    For each candidate ``c`` and each step:
        mmr(c) = (1 - λ) · relevance(c) + λ · diversity(c)

    where ``relevance(c)`` is the (min-max normalized) score and
    ``diversity(c)`` is ``min(d_min_to_selected / distance_norm, 1)``.

    Parameters
    ----------
    scores : (N,) array
        Per-candidate score. -inf entries are excluded.
    coords : (N, 3) array
        3D coordinates for distance computation.
    k : int
        Number of items to select.
    lambda_div : float in [0, 1]
        Diversity weight. 0 = pure greedy by score, 1 = pure max-diversity.
    distance_norm : float
        Distance (Å) that maps to diversity = 1.
    banned : set of int, optional
        Indices to exclude.

    Returns
    -------
    list of int — selected indices.
    """
    N = len(scores)
    s = np.asarray(scores, dtype=float).copy()
    finite = np.isfinite(s)
    if not finite.any():
        return []
    smin, smax = s[finite].min(), s[finite].max()
    s_norm = np.where(finite, (s - smin) / (smax - smin + 1e-12), 0.0)

    candidates = [i for i in range(N) if finite[i]]
    if banned is not None:
        candidates = [c for c in candidates if c not in banned]
    if len(candidates) < k:
        k = len(candidates)

    selected: list[int] = []
    remaining = candidates.copy()
    for _ in range(k):
        if not remaining:
            break
        best_idx, best_val = None, -np.inf
        for c in remaining:
            relevance = s_norm[c]
            if selected:
                d_min = min(
                    float(np.linalg.norm(coords[c] - coords[ss])) for ss in selected
                )
                diversity = min(d_min / distance_norm, 1.0)
            else:
                diversity = 1.0
            mmr = (1 - lambda_div) * relevance + lambda_div * diversity
            if mmr > best_val:
                best_val, best_idx = mmr, c
        if best_idx is None:
            break
        selected.append(int(best_idx))
        remaining.remove(best_idx)
    return selected
