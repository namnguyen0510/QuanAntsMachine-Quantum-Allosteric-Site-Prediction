"""
QSVD — Quantum Singular Value Decomposition.

Quantum-inspired method: takes the bottom-``n_components`` left singular vectors
of the adjacency matrix and scores each residue by its participation in this
subspace. Bottom modes capture slow / collective structural features that
correlate with allosteric communication channels.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from quanallo.methods.base import AllosteryMethod
from quanallo.data.schemas import ProteinGraph


@dataclass
class QSVD(AllosteryMethod):
    """Quantum-inspired SVD-subspace scoring.

    Parameters
    ----------
    n_components : int
        Number of bottom singular vectors to include in the subspace.
    use_weighted : bool
        If False (default), uses the binary adjacency — gives cleaner spectrum.
        If True, uses the distance-weighted adjacency.
    """

    n_components: int = 15
    use_weighted: bool = False
    name: str = "qsvd"
    kind: str = "quantum_inspired"
    requires_active_site: bool = False

    def compute(self, graph: ProteinGraph) -> np.ndarray:
        A = graph.adjacency_weighted if self.use_weighted else graph.adjacency_binary
        U, _, _ = np.linalg.svd(A.astype(float))
        # Sum squared participation in the bottom n_components left vectors
        return np.sum(U[:, -self.n_components:] ** 2, axis=1)

    def compute_with_pheromone(
        self,
        graph: ProteinGraph,
        pheromone: np.ndarray,
        *,
        pher_strength: float = 0.5,
    ) -> np.ndarray:
        """Pheromone multiplicatively boosts edges between high-pheromone residues."""
        A = graph.adjacency_weighted if self.use_weighted else graph.adjacency_binary
        boost = 1.0 + pher_strength * (pheromone[:, None] + pheromone[None, :])
        A_mod = A * boost
        U, _, _ = np.linalg.svd(A_mod.astype(float))
        return np.sum(U[:, -self.n_components:] ** 2, axis=1)
