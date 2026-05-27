"""
Personalized Quantum PageRank.

Directional walker with the active site as the teleport target. The damping
factor controls the trade-off between local walks (low damping) and global
teleport (high damping). The stationary distribution gives per-residue importance,
weighted by a Gaussian hop-window.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from quanallo.methods.base import AllosteryMethod
from quanallo.data.schemas import ProteinGraph
from quanallo.core.utils import hop_distances


@dataclass
class QPageRank(AllosteryMethod):
    """Personalized PageRank with active-site teleport bias.

    Parameters
    ----------
    damping : float in (0, 1)
        Random-walk damping factor (higher = more local).
    n_iter : int
        Number of power iterations.
    hop_mu, hop_sigma : float
        Gaussian hop-window applied to the stationary distribution.
    """

    damping: float = 0.85
    n_iter: int = 50
    hop_mu: float = 2.5
    hop_sigma: float = 1.8
    name: str = "qpagerank"
    kind: str = "quantum_inspired"
    requires_active_site: bool = True

    def compute(self, graph: ProteinGraph) -> np.ndarray:
        return self._compute(graph, lambda_p=0.0, pheromone=None)

    def compute_with_pheromone(
        self,
        graph: ProteinGraph,
        pheromone: np.ndarray,
        *,
        pher_strength: float = 0.5,
    ) -> np.ndarray:
        return self._compute(graph, lambda_p=pher_strength, pheromone=pheromone)

    def _compute(self, graph: ProteinGraph,
                 lambda_p: float, pheromone: np.ndarray | None) -> np.ndarray:
        A = graph.adjacency_weighted
        active = graph.active_idx
        N = graph.N
        h = hop_distances(A, active, N)
        p_tele = np.zeros(N)
        p_tele[active] = 1.0
        if pheromone is not None and lambda_p > 0:
            p_tele = p_tele + lambda_p * np.asarray(pheromone, dtype=float)
        p_tele = p_tele / (p_tele.sum() + 1e-12)

        d = A.sum(axis=1)
        P = A / np.where(d > 0, d, 1)[:, None]
        # Dangling-node fix
        P = P + (d == 0).astype(float)[:, None] * p_tele

        x = np.ones(N) / N
        for _ in range(self.n_iter):
            x = self.damping * (P.T @ x) + (1 - self.damping) * p_tele
            x = x / (x.sum() + 1e-12)
        return x * np.exp(-((h - self.hop_mu) ** 2) / (2 * self.hop_sigma ** 2))
