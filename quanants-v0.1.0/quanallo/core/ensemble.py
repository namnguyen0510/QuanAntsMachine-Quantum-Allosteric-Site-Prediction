"""
Strategies for combining multiple per-residue score vectors.
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np


def rrf_combine(
    score_dict: Dict[str, np.ndarray],
    k_rrf: float = 30.0,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Reciprocal Rank Fusion.

    For each method, rank residues by score (best=0). Fused score for residue j:

        fused(j) = Σ_methods  w_method / (k_rrf + rank_method(j) + 1)

    Robust to differences in score scale between methods.

    Parameters
    ----------
    score_dict : dict of {method_name: score_vector}
    k_rrf : float
        Smoothing constant (larger → ranks matter less).
    weights : dict of {method_name: weight}, optional
        If None, all methods weighted equally.
    """
    N = next(iter(score_dict.values())).shape[0]
    fused = np.zeros(N, dtype=float)
    for name, s in score_dict.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        ranks = (-np.asarray(s)).argsort().argsort()
        fused += w / (k_rrf + ranks + 1)
    return fused


def weighted_combine(
    score_dict: Dict[str, np.ndarray],
    weights: Dict[str, float],
    normalize_first: bool = True,
) -> np.ndarray:
    """Linear weighted combination of per-residue scores.

    Parameters
    ----------
    score_dict : dict of {method_name: score_vector}
    weights : dict of {method_name: weight}
        Weights need not sum to 1.
    normalize_first : bool
        If True, min-max normalize each method's scores to [0, 1] before combining.
    """
    N = next(iter(score_dict.values())).shape[0]
    out = np.zeros(N, dtype=float)
    for name, s in score_dict.items():
        s = np.asarray(s, dtype=float)
        if normalize_first:
            lo, hi = s.min(), s.max()
            s = (s - lo) / (hi - lo + 1e-12)
        w = float(weights.get(name, 0.0))
        out += w * s
    return out


def trimmed_mean_combine(
    score_dict: Dict[str, np.ndarray],
    trim_lo: int = 1,
    trim_hi: int = 0,
) -> np.ndarray:
    """
    For each residue, drop the ``trim_lo`` lowest and ``trim_hi`` highest method
    scores, then average the remaining. Robust to single-method outliers.

    Each method's scores are min-max normalized to [0, 1] before combining.
    """
    sps = list(score_dict.keys())
    n_methods = len(sps)
    if trim_lo + trim_hi >= n_methods:
        raise ValueError("trim_lo + trim_hi must be < number of methods")
    stacked = []
    for name in sps:
        s = np.asarray(score_dict[name], dtype=float)
        lo, hi = s.min(), s.max()
        stacked.append((s - lo) / (hi - lo + 1e-12))
    arr = np.stack(stacked, axis=0)        # (n_methods, N)
    sorted_ = np.sort(arr, axis=0)
    kept = sorted_[trim_lo:n_methods - trim_hi if trim_hi > 0 else n_methods]
    return kept.mean(axis=0)


def softmax_with_floor(
    score_dict: Dict[str, float],
    temperature: float = 1.0,
    floor: float = 0.05,
) -> Dict[str, float]:
    """
    Convert a dict of raw scores into a normalized weight distribution.

    Applies a stability-shifted softmax with a minimum-weight floor so no method
    can get exactly zero (preserves exploration):

        w_i = max(floor, softmax((s_i - max s) / T))
        w  /= sum(w)
    """
    sps = list(score_dict.keys())
    raw = np.array([score_dict[sp] for sp in sps], dtype=float)
    if raw.sum() < 1e-9:
        w = np.ones_like(raw) / len(raw)
    else:
        shifted = (raw - raw.max()) / max(temperature, 1e-6)
        w = np.exp(shifted)
        w /= w.sum()
    w = np.maximum(w, floor)
    w /= w.sum()
    return {sp: float(w[i]) for i, sp in enumerate(sps)}
