"""
QuanAnt Machines — ant colonies of quantum walkers that communicate via a
shared (or per-species) pheromone trail.

Public classes
--------------
- :class:`QuanAntColony` — fixed-weights colony. Supports any subset of methods.
- :class:`AdaptiveQuanAnt` — APO-trained species weights + online updates.

Each "ant" is a single quantum-method instance with perturbed hyperparameters.
Per iteration, each ant runs once with the current pheromone field;
the top-k of each ant deposits 1 unit of pheromone, then the field evaporates
by (1 - evap_rate) and the new deposits are added.
"""
from __future__ import annotations
import os
import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from quanallo.data.schemas import ProteinGraph
from quanallo.methods import (
    AllosteryMethod, METHOD_REGISTRY,
    QSVD, DQAWTimeAvg, DQAWLifetime, QPageRank, HeatKernel,
)
from quanallo.core.ensemble import softmax_with_floor, weighted_combine, rrf_combine
from quanallo.core.utils import evaluate


# ----------------------------------------------------------------------
# Default per-species pheromone-aware ant constructor with random perturbations
# ----------------------------------------------------------------------
_BASE_DQAW_T = dict(alpha=0.4, beta_well=2.0, gamma_absorb=0.6,
                    hop_mu=2.0, hop_sigma=1.3)
_BASE_DQAW_L = dict(alpha=0.4, beta_well=2.0, gamma_absorb=0.2,
                    hop_mu=2.0, hop_sigma=2.3)
_BASE_QPR    = dict(damping=0.85, hop_mu=2.5, hop_sigma=1.8, n_iter=40)


def _instantiate_ant(species: str, seed: int) -> AllosteryMethod:
    """Build an ant of the given species with seed-dependent hyperparameter
    perturbations. ±30% on multiplicative params, ±0.3 on additive."""
    rng = np.random.default_rng(seed)
    if species == "qsvd":
        n_comp = int(rng.integers(10, 23))
        return QSVD(n_components=n_comp)
    if species == "dqaw_timeavg":
        p = {k: _BASE_DQAW_T[k] * rng.uniform(0.8, 1.2)
             for k in ["alpha", "beta_well", "gamma_absorb", "hop_sigma"]}
        p["hop_mu"] = _BASE_DQAW_T["hop_mu"] + rng.uniform(-0.3, 0.3)
        return DQAWTimeAvg(**p)
    if species == "dqaw_lifetime":
        p = {k: _BASE_DQAW_L[k] * rng.uniform(0.8, 1.2)
             for k in ["alpha", "gamma_absorb", "hop_sigma"]}
        p["hop_mu"] = _BASE_DQAW_L["hop_mu"] + rng.uniform(-0.3, 0.3)
        return DQAWLifetime(**p)
    if species == "qpagerank":
        return QPageRank(
            damping=rng.uniform(0.75, 0.95),
            hop_mu=_BASE_QPR["hop_mu"] + rng.uniform(-0.4, 0.4),
            hop_sigma=_BASE_QPR["hop_sigma"] * rng.uniform(0.8, 1.2),
            n_iter=_BASE_QPR["n_iter"],
        )
    if species == "heatkernel":
        return HeatKernel(t=2.0 * rng.uniform(0.5, 2.0))
    # Generic fallback for any other registered method (no perturbation)
    if species in METHOD_REGISTRY:
        return METHOD_REGISTRY[species]()
    raise ValueError(f"Unknown species: {species!r}")


# ----------------------------------------------------------------------
# Worker function (top-level for picklability under multiprocessing)
# ----------------------------------------------------------------------
def _run_one_ant(args):
    """Execute one ant. args = (species, seed, graph, pheromone, pher_strength)."""
    species, seed, graph, pheromone, pher_strength = args
    ant = _instantiate_ant(species, seed)
    if pheromone is not None:
        score = ant.compute_with_pheromone(graph, pheromone, pher_strength=pher_strength)
    else:
        score = ant.compute(graph)
    return (species, score)


# ----------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------
@dataclass
class QuanAntResult:
    """Output of a QuanAnt colony run."""
    final_score: np.ndarray
    """Per-residue aggregate score (length N)."""

    per_species_score: dict[str, np.ndarray] = field(default_factory=dict)
    """For multi-species colonies: per-species aggregated scores."""

    species_contribution: dict[str, np.ndarray] = field(default_factory=dict)
    """Cumulative deposit count per species per residue (only for shared-pheromone modes)."""

    history: list[dict] = field(default_factory=list)
    """Per-iteration diagnostics."""

    species_weights_history: Optional[list[dict[str, float]]] = None
    """For AdaptiveQuanAnt: per-iteration species weight vectors."""


# ----------------------------------------------------------------------
# Main colony class
# ----------------------------------------------------------------------
@dataclass
class QuanAntColony:
    """
    Ant colony of pheromone-communicating quantum walkers.

    Parameters
    ----------
    species : list of str
        Ant species to include. Each name must be a key of ``METHOD_REGISTRY``
        (the default ant-construction recipes know how to perturb 'qsvd',
        'dqaw_timeavg', 'dqaw_lifetime', 'qpagerank', 'heatkernel').
    ants_per_species : int
        Number of ants per species (each with perturbed hyperparameters).
    n_iter : int
        Number of pheromone update iterations.
    evap_rate : float in [0, 1]
        Pheromone evaporation per iteration.
    deposit_topk : int
        Each ant deposits 1 unit on its top-``deposit_topk`` residues.
    pher_strength : float
        Strength of pheromone influence on ants.
    aggregation : {"shared_pheromone", "multi_pheromone", "weighted_consensus"}
        - ``"shared_pheromone"``: single field, deposits all go in (uniform weight).
        - ``"multi_pheromone"``: each species has its own field; final = weighted sum.
        - ``"weighted_consensus"``: single field; deposits scaled by ``species_weights``.
    species_weights : dict[str, float], optional
        Weights for non-uniform aggregation. Required for ``"weighted_consensus"``
        unless using :class:`AdaptiveQuanAnt`.
    parallel : bool
        Use thread pool for parallel ant execution.
    max_workers : int, optional
        Number of worker threads. Defaults to ``2 × os.cpu_count()``.
    """

    species: Sequence[str] = ("dqaw_lifetime",)
    ants_per_species: int = 30
    n_iter: int = 8
    evap_rate: float = 0.30
    deposit_topk: int = 5
    pher_strength: float = 2.0
    aggregation: str = "shared_pheromone"
    species_weights: Optional[dict] = None
    parallel: bool = True
    max_workers: Optional[int] = None
    verbose: bool = False

    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = max(2, min(8, (os.cpu_count() or 1) * 2))
        # Validate species
        for sp in self.species:
            if sp not in METHOD_REGISTRY:
                raise ValueError(f"Unknown species: {sp!r}")
        # Validate aggregation
        valid_agg = {"shared_pheromone", "multi_pheromone", "weighted_consensus"}
        if self.aggregation not in valid_agg:
            raise ValueError(f"aggregation must be one of {valid_agg}")

    def _deploy_ants(self, arg_list: list) -> list:
        """Run ants in parallel (or serial)."""
        if not self.parallel or len(arg_list) <= 1:
            return [_run_one_ant(a) for a in arg_list]
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(_run_one_ant, arg_list))

    def _compute_deposit(self, results: list, graph: ProteinGraph) -> dict[str, np.ndarray]:
        """For each species, compute the deposit field from its ants' top-k."""
        N = graph.N
        active = graph.active_idx
        surface_mask = graph.surface_mask
        per_species = {sp: np.zeros(N) for sp in self.species}
        for sp, score in results:
            s = score.copy()
            s[active] = -np.inf
            s[~surface_mask] = -np.inf
            top = np.argsort(s)[::-1][:self.deposit_topk]
            for t in top:
                if np.isfinite(s[t]):
                    per_species[sp][t] += 1.0
        # Normalize per species
        for sp in per_species:
            mx = per_species[sp].max()
            if mx > 0:
                per_species[sp] /= mx
        return per_species

    def run(self, graph: ProteinGraph) -> QuanAntResult:
        """
        Run the colony on ``graph``.

        Returns
        -------
        :class:`QuanAntResult`
        """
        N = graph.N
        active = graph.active_idx

        # Initialize pheromone(s)
        if self.aggregation == "multi_pheromone":
            pheromones: dict[str, np.ndarray] = {
                sp: np.full(N, 0.05) for sp in self.species
            }
            for sp in self.species:
                pheromones[sp][active] = 0.0
        else:
            pheromone = np.full(N, 0.05)
            pheromone[active] = 0.0

        species_contribution = {sp: np.zeros(N) for sp in self.species}
        history: list[dict] = []
        weights = self.species_weights or {sp: 1.0 / len(self.species) for sp in self.species}

        for it in range(self.n_iter):
            # ---- build ant work list ----
            arg_list = []
            for sp_id, sp in enumerate(self.species):
                for a in range(self.ants_per_species):
                    seed = 7919 * (it + 1) + 101 * sp_id + a
                    if self.aggregation == "multi_pheromone":
                        pher = pheromones[sp]
                    else:
                        pher = pheromone
                    arg_list.append((sp, seed, graph, pher, self.pher_strength))

            # ---- run ants ----
            results = self._deploy_ants(arg_list)
            per_species_deposit = self._compute_deposit(results, graph)

            # ---- track contributions ----
            for sp in self.species:
                species_contribution[sp] += per_species_deposit[sp]

            # ---- update pheromone(s) ----
            if self.aggregation == "multi_pheromone":
                for sp in self.species:
                    pheromones[sp] = (1 - self.evap_rate) * pheromones[sp] + per_species_deposit[sp]
                    pheromones[sp][active] = 0.0
            else:
                # Shared pheromone: deposit = (weighted) sum of species deposits
                deposit = np.zeros(N)
                for sp in self.species:
                    w = weights[sp] if self.aggregation == "weighted_consensus" else 1.0
                    deposit += w * per_species_deposit[sp]
                mx = deposit.max()
                if mx > 0:
                    deposit /= mx
                pheromone = (1 - self.evap_rate) * pheromone + deposit
                pheromone[active] = 0.0

            # ---- diagnostics ----
            entry = {"iter": it}
            if self.aggregation == "multi_pheromone":
                entry["pher_max_per_species"] = {sp: float(pheromones[sp].max())
                                                  for sp in self.species}
            else:
                entry["pher_max"] = float(pheromone.max())
            history.append(entry)

            if self.verbose:
                msg = f"  iter {it+1}/{self.n_iter}"
                if self.aggregation == "multi_pheromone":
                    pmax = max(pheromones[sp].max() for sp in self.species)
                    msg += f"  pher_max={pmax:.3f}"
                else:
                    msg += f"  pher_max={pheromone.max():.3f}"
                print(msg)

        # ---- final aggregation ----
        per_species_score = {}
        if self.aggregation == "multi_pheromone":
            for sp in self.species:
                per_species_score[sp] = pheromones[sp]
            # Final = weighted sum across species
            final = np.zeros(N)
            for sp in self.species:
                final += weights[sp] * pheromones[sp]
        else:
            final = pheromone
            for sp in self.species:
                per_species_score[sp] = species_contribution[sp]

        return QuanAntResult(
            final_score=final,
            per_species_score=per_species_score,
            species_contribution=species_contribution,
            history=history,
        )


# ----------------------------------------------------------------------
# Adaptive QuanAnt: APO-trained weights + online updates
# ----------------------------------------------------------------------
def _cross_species_agreement(species_deposits: dict, top_n: int = 10) -> dict:
    """Jaccard agreement of each species' top-N with the others."""
    topk_sets = {sp: set(np.argsort(d)[::-1][:top_n].tolist())
                 for sp, d in species_deposits.items()}
    sps = list(species_deposits.keys())
    out = {}
    for sp in sps:
        agreements = []
        for sp2 in sps:
            if sp != sp2:
                a, b = topk_sets[sp], topk_sets[sp2]
                j = len(a & b) / max(1, len(a | b))
                agreements.append(j)
        out[sp] = float(np.mean(agreements)) if agreements else 0.0
    return out


@dataclass
class AdaptiveQuanAnt:
    """
    Two-stage adaptive QuanAnt with APO-to-HOLO transfer learning.

    Stage 1 — ``fit(apo_graph)``: each species runs **independently** (its own
    pheromone field) on the APO graph for ``n_iter`` iterations. Its per-species
    weighted_top5 (vs APO GT) is converted to a softmax weight with floor.

    Stage 2 — ``predict(graph)``: runs a weighted-consensus colony on ``graph``
    starting from the APO-learned weights. Each iteration, weights are updated
    by a Jaccard-agreement signal between species' top-k:

        w_t+1 = momentum · w_apo + (1 - momentum) · w_jaccard_t

    Parameters
    ----------
    species : list of str
    ants_per_species : int
    n_iter : int
    evap_rate, deposit_topk, pher_strength : as in :class:`QuanAntColony`
    softmax_temperature : float
        Softmax temperature for converting APO scores → weights. Higher = flatter.
    weight_floor : float
        Minimum weight per species (preserves exploration).
    adaptive_momentum : float in [0, 1]
        Weight of the APO prior in online updates. 1 = static, 0 = pure online.
    parallel : bool
    max_workers : int, optional
    verbose : bool
    """

    species: Sequence[str] = ("qsvd", "dqaw_timeavg", "dqaw_lifetime",
                              "qpagerank", "heatkernel")
    ants_per_species: int = 5
    n_iter: int = 7
    evap_rate: float = 0.30
    deposit_topk: int = 5
    pher_strength: float = 2.0
    softmax_temperature: float = 1.0
    weight_floor: float = 0.10
    adaptive_momentum: float = 0.40
    parallel: bool = True
    max_workers: Optional[int] = None
    verbose: bool = False

    # Learned state
    _apo_weights: Optional[dict] = field(default=None, repr=False)
    _apo_species_scores: Optional[dict] = field(default=None, repr=False)
    _apo_species_pher: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = max(2, min(8, (os.cpu_count() or 1) * 2))

    @property
    def apo_weights(self) -> dict:
        if self._apo_weights is None:
            raise RuntimeError(
                "AdaptiveQuanAnt not fitted yet. Call .fit(apo_graph) first."
            )
        return self._apo_weights

    def fit(self, apo_graph: ProteinGraph) -> "AdaptiveQuanAnt":
        """Stage 1 — learn per-species reliability on the APO graph.
        Requires ``apo_graph.ground_truth_idx`` to be populated."""
        if apo_graph.ground_truth_idx is None:
            raise ValueError(
                "AdaptiveQuanAnt.fit needs apo_graph.ground_truth_idx. "
                "(Run build_graph_from_pdb with ground_truth_ligand= ...)"
            )
        # Run each species independently
        colony = QuanAntColony(
            species=list(self.species),
            ants_per_species=self.ants_per_species,
            n_iter=self.n_iter,
            evap_rate=self.evap_rate,
            deposit_topk=self.deposit_topk,
            pher_strength=self.pher_strength,
            aggregation="multi_pheromone",
            parallel=self.parallel,
            max_workers=self.max_workers,
            verbose=self.verbose,
        )
        result = colony.run(apo_graph)
        # Per-species evaluation vs APO GT
        scores = {}
        for sp in self.species:
            ev = evaluate(
                result.per_species_score[sp],
                apo_graph.adjacency_weighted,
                apo_graph.ground_truth_idx,
                apo_graph.active_idx,
                apo_graph.N,
                surface_mask=apo_graph.surface_mask,
            )
            scores[sp] = ev["weighted_top5"]
        self._apo_species_scores = scores
        self._apo_species_pher = result.per_species_score
        self._apo_weights = softmax_with_floor(
            scores,
            temperature=self.softmax_temperature,
            floor=self.weight_floor,
        )
        if self.verbose:
            print("\n[AdaptiveQuanAnt] APO-learned species weights:")
            for sp, w in self._apo_weights.items():
                print(f"   {sp:<16} {w:.3f}  (raw={scores[sp]:.3f})")
        return self

    def predict(self, graph: ProteinGraph) -> QuanAntResult:
        """Stage 2 — deploy on ``graph`` with online adaptive weighting."""
        if self._apo_weights is None:
            raise RuntimeError("Call .fit(apo_graph) before .predict(graph)")

        N = graph.N
        active = graph.active_idx
        pheromone = np.full(N, 0.05)
        pheromone[active] = 0.0

        weights = dict(self._apo_weights)
        weight_history: list[dict[str, float]] = []
        contrib = {sp: np.zeros(N) for sp in self.species}

        for it in range(self.n_iter):
            # Build ant work list — all species share pheromone
            arg_list = []
            for sp_id, sp in enumerate(self.species):
                for a in range(self.ants_per_species):
                    seed = 7919 * (it + 1) + 101 * sp_id + a
                    arg_list.append((sp, seed, graph, pheromone, self.pher_strength))

            # Execute
            if self.parallel and len(arg_list) > 1:
                with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    results = list(pool.map(_run_one_ant, arg_list))
            else:
                results = [_run_one_ant(a) for a in arg_list]

            # Per-species deposit
            surface_mask = graph.surface_mask
            per_species_deposit = {sp: np.zeros(N) for sp in self.species}
            for sp, score in results:
                s = score.copy()
                s[active] = -np.inf
                s[~surface_mask] = -np.inf
                top = np.argsort(s)[::-1][:self.deposit_topk]
                for t in top:
                    if np.isfinite(s[t]):
                        per_species_deposit[sp][t] += 1.0
            for sp in self.species:
                mx = per_species_deposit[sp].max()
                if mx > 0:
                    per_species_deposit[sp] /= mx
                contrib[sp] += per_species_deposit[sp]

            # Online weight update via cross-species Jaccard agreement
            agreement = _cross_species_agreement(per_species_deposit, top_n=10)
            new_w = {sp: self.adaptive_momentum * weights[sp]
                       + (1 - self.adaptive_momentum) * agreement[sp]
                     for sp in self.species}
            weights = softmax_with_floor(
                new_w,
                temperature=self.softmax_temperature,
                floor=self.weight_floor,
            )
            weight_history.append(dict(weights))

            # Weighted deposit
            deposit = np.zeros(N)
            for sp in self.species:
                deposit += weights[sp] * per_species_deposit[sp]
            mx = deposit.max()
            if mx > 0:
                deposit /= mx
            pheromone = (1 - self.evap_rate) * pheromone + deposit
            pheromone[active] = 0.0

            if self.verbose:
                w_str = " ".join(f"{sp[:3]}={weights[sp]:.2f}" for sp in self.species)
                print(f"  iter {it+1}/{self.n_iter}  weights={w_str}")

        return QuanAntResult(
            final_score=pheromone,
            per_species_score={sp: contrib[sp] for sp in self.species},
            species_contribution=contrib,
            history=[{"iter": i} for i in range(self.n_iter)],
            species_weights_history=weight_history,
        )
