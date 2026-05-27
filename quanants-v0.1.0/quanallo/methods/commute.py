"""
Classical baselines: CommuteTime and Gaussian Network Model (GNM).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from quanallo.methods.base import AllosteryMethod
from quanallo.data.schemas import ProteinGraph
from quanallo.core.utils import sym_norm_laplacian


@dataclass
class CommuteTime(AllosteryMethod):
    """
    Commute-time / effective-resistance scoring.

    For each residue j and each active-site residue i, compute the effective
    resistance R(i,j) on the graph (using the pseudo-inverse of the Laplacian).
    Score per j = Σ_i 1 / R(i,j) (low resistance to many active residues).
    """

    name: str = "commute_time"
    kind: str = "classical"
    requires_active_site: bool = True

    def compute(self, graph: ProteinGraph) -> np.ndarray:
        A = graph.adjacency_weighted
        active = graph.active_idx
        N = graph.N
        L = sym_norm_laplacian(A)
        Lp = np.linalg.pinv(L)
        diag = np.diag(Lp)
        out = np.zeros(N)
        for i in active:
            R = diag + diag[i] - 2 * Lp[:, i]
            out += 1.0 / (R + 1e-6)
        return out


@dataclass
class GNM(AllosteryMethod):
    """
    Gaussian Network Model — covariance of slow normal modes.

    Equilibrium covariance C ~ L⁺ projected onto the slowest n_modes eigenmodes.
    |C(active, j)| ranks residues whose collective fluctuations are most
    dynamically coupled to the active site — a complementary signal to
    coherent quantum dynamics.

    Parameters
    ----------
    n_slow_modes : int
        Number of lowest non-zero Laplacian eigenmodes to include.
    """

    n_slow_modes: int = 10
    name: str = "gnm"
    kind: str = "classical"
    requires_active_site: bool = True

    def compute(self, graph: ProteinGraph) -> np.ndarray:
        A = graph.adjacency_weighted
        active = graph.active_idx
        N = graph.N
        L = sym_norm_laplacian(A)
        evals, evecs = eigh(L)
        eps = 1e-8
        inv_evals = np.where(evals > eps, 1.0 / np.maximum(evals, eps), 0.0)
        sel = np.argsort(evals)
        use_idx = sel[1:self.n_slow_modes + 1]  # skip zero mode
        C = np.zeros((N, N))
        for k in use_idx:
            C += inv_evals[k] * np.outer(evecs[:, k], evecs[:, k])
        return np.abs(C[:, active]).mean(axis=1)
