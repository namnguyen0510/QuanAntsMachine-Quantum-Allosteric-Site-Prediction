"""
Heat-kernel and continuous-time quantum walk (CTQW) methods.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, eigh

from quanallo.methods.base import AllosteryMethod
from quanallo.data.schemas import ProteinGraph
from quanallo.core.utils import sym_norm_laplacian, hop_distances


@dataclass
class HeatKernel(AllosteryMethod):
    """
    Heat-kernel propagation on the residue graph.

    Computes (exp(-L·t) ψ_active)² where ψ_active is the indicator of the
    active site. Captures the equilibrium spread of signal from the source.

    Parameters
    ----------
    t : float
        Diffusion time (higher = signal spreads further).
    """

    t: float = 2.0
    name: str = "heatkernel"
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
        L = sym_norm_laplacian(A)
        H_op = expm(-L * self.t)
        psi = np.zeros(N)
        if len(active) > 0:
            psi[active] = 1.0 / np.sqrt(len(active))
        if pheromone is not None and lambda_p > 0:
            psi = psi + lambda_p * np.asarray(pheromone, dtype=float)
        psi = psi / (np.linalg.norm(psi) + 1e-12)
        return (H_op @ psi) ** 2


@dataclass
class CTQW(AllosteryMethod):
    """
    Standard continuous-time quantum walk (Farhi-Gutmann form).

    Hamiltonian = symmetrically normalized Laplacian. Score = time-averaged
    propagation amplitude squared from any active-site residue to each target.

    Parameters
    ----------
    T_max : float
        End time of the walk.
    n_times : int
        Number of time points to average over.
    """

    T_max: float = 20.0
    n_times: int = 40
    name: str = "ctqw"
    kind: str = "quantum"
    requires_active_site: bool = True

    def compute(self, graph: ProteinGraph) -> np.ndarray:
        A = graph.adjacency_weighted
        active = graph.active_idx
        N = graph.N
        H = sym_norm_laplacian(A)
        evals, evecs = eigh(H)
        times = np.linspace(0.5, self.T_max, self.n_times)
        C = np.zeros((N, N))
        for t in times:
            phases = np.exp(-1j * evals * t)
            U = (evecs * phases) @ evecs.conj().T
            C += np.abs(U) ** 2
        C /= len(times)
        return C[active, :].sum(axis=0)
