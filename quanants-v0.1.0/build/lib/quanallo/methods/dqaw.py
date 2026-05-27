"""
DQAW — Directed Quantum Attractor Walk.

The flagship pair of quantum methods in QuanAllo. Both build a non-Hermitian
Hamiltonian H_eff = H_dir - i·Γ·P_active where:

  * H_dir is a Hermitian "directed" coupling matrix with edges modulated by
    the k-hop distance to the active site (edges deep in the attractor well
    get amplified).
  * Γ is an absorbing rate at the active site, making the dynamics effectively
    irreversible (signal flows IN to the active site).
  * A Gaussian hop-window then favours residues at a "sweet spot" hop distance
    from the active site (μ ≈ 2-3 hops for typical allosteric pockets).

Two scoring strategies are implemented:

  - :class:`DQAWTimeAvg`  — time-averaged occupation under non-Hermitian H_eff.
  - :class:`DQAWLifetime` — participation in the SLOWEST-decaying eigenmodes of
    H_eff. Long-lived eigenmodes correspond to quantum coherences that resist
    absorption = allosteric communication channels. This is typically the
    best single-method choice for distal allosteric pockets.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig as eig_general

from quanallo.methods.base import AllosteryMethod
from quanallo.data.schemas import ProteinGraph
from quanallo.core.utils import hop_distances


def _build_directed_h(A: np.ndarray, active_idx: np.ndarray, alpha: float):
    """Edge re-weighting with k-hop attractor field.

    M(i,j) = A(i,j) · exp(-α · min(h_i, h_j))

    where h is the hop distance to the active-site set. Edges deep in the
    attractor (low min-hop) get amplified.
    """
    N = A.shape[0]
    h = hop_distances(A, active_idx, N)
    min_h = np.minimum.outer(h, h).astype(float)
    M = A * np.exp(-alpha * min_h)
    M = 0.5 * (M + M.T)
    return M, h


@dataclass
class DQAWTimeAvg(AllosteryMethod):
    """
    Directed Quantum Attractor Walk — time-averaged occupation.

    Parameters
    ----------
    alpha : float
        Strength of edge re-weighting by hop-distance.
    beta_well : float
        Depth of the attractor potential well at the active site.
    gamma_absorb : float
        Absorption rate (imaginary part of H) at the active site.
    hop_mu, hop_sigma : float
        Gaussian hop-window: scores multiplied by exp(-(h - μ)² / 2σ²).
    T_max : float
        End time of the walk (arbitrary units).
    n_times : int
        Number of time points to sample for time averaging.
    """

    alpha: float = 0.4
    beta_well: float = 2.0
    gamma_absorb: float = 0.6
    hop_mu: float = 2.0
    hop_sigma: float = 1.3
    T_max: float = 12.0
    n_times: int = 30
    name: str = "dqaw_timeavg"
    kind: str = "quantum"
    requires_active_site: bool = True

    def compute(self, graph: ProteinGraph) -> np.ndarray:
        return self._compute(graph, extra_potential=None)

    def compute_with_pheromone(
        self,
        graph: ProteinGraph,
        pheromone: np.ndarray,
        *,
        pher_strength: float = 2.0,
    ) -> np.ndarray:
        extra_pot = -pher_strength * np.asarray(pheromone, dtype=float)
        return self._compute(graph, extra_potential=extra_pot)

    def _compute(self, graph: ProteinGraph,
                 extra_potential: np.ndarray | None) -> np.ndarray:
        A = graph.adjacency_weighted
        active = graph.active_idx
        N = graph.N
        M, h = _build_directed_h(A, active, self.alpha)
        H_herm = np.diag(M.sum(axis=1)) - M
        V = np.zeros(N)
        V[active] = -self.beta_well
        if extra_potential is not None:
            V = V + extra_potential
        H_herm = H_herm + np.diag(V)
        P_a = np.zeros(N); P_a[active] = 1.0
        H_eff = H_herm - 1j * self.gamma_absorb * np.diag(P_a)

        # Source = uniform superposition over non-active residues
        active_set = set(active.tolist())
        sources = np.array([i for i in range(N) if i not in active_set])
        psi0 = np.zeros(N, dtype=complex)
        psi0[sources] = 1.0
        psi0 /= np.linalg.norm(psi0)

        evals, evecs = eig_general(H_eff)
        evecs_inv = np.linalg.pinv(evecs)
        c0 = evecs_inv @ psi0
        times = np.linspace(0.5, self.T_max, self.n_times)
        occ = np.zeros(N)
        for t in times:
            psi_t = evecs @ (c0 * np.exp(-1j * evals * t))
            occ += np.abs(psi_t) ** 2
        occ /= len(times)
        return occ * np.exp(-((h - self.hop_mu) ** 2) / (2 * self.hop_sigma ** 2))


@dataclass
class DQAWLifetime(AllosteryMethod):
    """
    Directed Quantum Attractor Walk — eigenmode lifetime scoring.

    Eigenvalues of the non-Hermitian H_eff are complex; the imaginary part is
    a decay rate, so τ_k = 1 / |Im λ_k| is a "lifetime" of mode k. Modes with
    long τ are quantum coherences that resist absorption by the active site —
    i.e., they live on residues that the active site cannot quickly drain.
    These are the allosteric communication channels.

    Score: Σ_{k ∈ top-n_modes by τ}  |v_k(j)|²  ·  τ_k

    Then weighted by the Gaussian hop-window.

    Parameters
    ----------
    n_modes : int
        Number of longest-lived eigenmodes to sum over.
    (others as in :class:`DQAWTimeAvg`)
    """

    alpha: float = 0.4
    gamma_absorb: float = 0.2
    hop_mu: float = 2.0
    hop_sigma: float = 2.3
    n_modes: int = 8
    skip_first_mode: bool = True
    name: str = "dqaw_lifetime"
    kind: str = "quantum"
    requires_active_site: bool = True

    def compute(self, graph: ProteinGraph) -> np.ndarray:
        return self._compute(graph, extra_potential=None)

    def compute_with_pheromone(
        self,
        graph: ProteinGraph,
        pheromone: np.ndarray,
        *,
        pher_strength: float = 2.0,
    ) -> np.ndarray:
        extra_pot = -pher_strength * np.asarray(pheromone, dtype=float)
        return self._compute(graph, extra_potential=extra_pot)

    def _compute(self, graph: ProteinGraph,
                 extra_potential: np.ndarray | None) -> np.ndarray:
        A = graph.adjacency_weighted
        active = graph.active_idx
        N = graph.N
        M, h = _build_directed_h(A, active, self.alpha)
        H_herm = np.diag(M.sum(axis=1)) - M
        if extra_potential is not None:
            H_herm = H_herm + np.diag(extra_potential)
        P_a = np.zeros(N); P_a[active] = 1.0
        H_eff = H_herm - 1j * self.gamma_absorb * np.diag(P_a)
        evals, evecs = eig_general(H_eff)
        decay = -np.imag(evals)
        decay = np.where(decay > 1e-6, decay, 1e-6)
        lifetimes = 1.0 / decay
        order = np.argsort(-lifetimes)
        # Optionally skip the first mode (often a near-zero-decay artifact)
        start = 1 if self.skip_first_mode else 0
        selected = order[start:start + self.n_modes]
        scores = np.zeros(N)
        for k in selected:
            scores += np.abs(evecs[:, k]) ** 2 * lifetimes[k]
        return scores * np.exp(-((h - self.hop_mu) ** 2) / (2 * self.hop_sigma ** 2))
