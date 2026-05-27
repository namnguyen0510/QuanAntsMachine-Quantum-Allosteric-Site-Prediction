"""
High-level :class:`AllostericPredictor` — the one-line entry point for users.

>>> from quanallo import AllostericPredictor
>>> predictor = AllostericPredictor(method="dqaw_lifetime")
>>> result = predictor.predict_from_pdb("4OBE.pdb", auto_active_site_ligand="GDP")
>>> print(result.top_residues)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quanallo.data.schemas import ProteinGraph
from quanallo.data.extraction import build_graph_from_pdb
from quanallo.methods import METHOD_REGISTRY, AllosteryMethod
from quanallo.methods.meta_learner import MetaLearner
from quanallo.quanant.colony import QuanAntColony, AdaptiveQuanAnt
from quanallo.core.utils import evaluate


# ======================================================================
# Output dataclass
# ======================================================================
@dataclass
class PredictionResult:
    """
    Output of an :class:`AllostericPredictor` run.

    Attributes
    ----------
    scores : np.ndarray
        Per-residue prediction scores (length N).
    top_indices : list of int
        Selected top-k residue indices (in graph.nodes order).
    top_residues : list of dict
        Detailed metadata for each top-k pick: chain, resnum, resname.
    method_used : str
        Identifier of the method that produced the scores.
    graph : ProteinGraph
        The graph that was scored.
    weighted_top5 : float, optional
        Fuzzy GT credit (0–5.0) if ground truth was available.
    precision_at_k : dict, optional
        Precision-at-5 by k-hop tolerance.
    hits : list of dict, optional
        Per-rank hit details (hop_to_gt, credit, etc.) if GT was available.
    """

    scores: np.ndarray
    top_indices: list
    top_residues: list
    method_used: str
    graph: ProteinGraph
    weighted_top5: Optional[float] = None
    precision_at_k: Optional[dict] = None
    hits: Optional[list] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return the top-k hits as a pandas DataFrame."""
        if self.hits is not None:
            return pd.DataFrame(self.hits)
        return pd.DataFrame(self.top_residues)

    def __repr__(self) -> str:
        s = f"PredictionResult(method={self.method_used!r}, top-{len(self.top_indices)} = "
        s += ",".join(f"{r['chain']}{r['resnum']}" for r in self.top_residues)
        if self.weighted_top5 is not None:
            s += f", weighted_top5={self.weighted_top5:.3f}"
        return s + ")"


# ======================================================================
# Main class
# ======================================================================
class AllostericPredictor:
    """
    High-level allosteric-site predictor.

    Supports three operating modes:

    1. **Single method** (default):

       >>> predictor = AllostericPredictor(method="dqaw_lifetime")
       >>> result = predictor.predict(graph)

    2. **Multi-method ensemble** (rank fusion or weighted average):

       >>> predictor = AllostericPredictor(
       ...     methods=["dqaw_lifetime", "qsvd", "qpagerank"],
       ...     ensemble="rrf",  # or "mean", "weighted"
       ... )

    3. **QuanAnt colony** (ant-based with pheromone communication):

       >>> predictor = AllostericPredictor(
       ...     method="quanant",
       ...     quanant_species=["qsvd", "dqaw_lifetime"],
       ...     ants_per_species=10,
       ...     n_iter=8,
       ... )

    4. **Adaptive QuanAnt** (APO→HOLO transfer learning):

       >>> predictor = AllostericPredictor(method="adaptive_quanant")
       >>> predictor.fit(apo_graph)              # use APO GT
       >>> result = predictor.predict(holo_graph)
    """

    def __init__(
        self,
        method: str = "dqaw_lifetime",
        *,
        methods: Optional[Sequence[str]] = None,
        ensemble: str = "rrf",
        ensemble_weights: Optional[dict] = None,
        top_k: int = 5,
        selection: str = "argmax",
        only_surface: bool = True,
        mask_active: bool = True,
        method_kwargs: Optional[dict] = None,
        # QuanAnt-specific
        quanant_species: Optional[Sequence[str]] = None,
        ants_per_species: int = 10,
        n_iter: int = 8,
        evap_rate: float = 0.30,
        deposit_topk: int = 5,
        aggregation: str = "shared_pheromone",
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = False,
    ):
        self.top_k = top_k
        self.selection = selection
        self.only_surface = only_surface
        self.mask_active = mask_active
        self.verbose = verbose
        self._fitted_meta: Optional[MetaLearner] = None
        self._fitted_adaptive: Optional[AdaptiveQuanAnt] = None

        if methods is not None:
            # Multi-method ensemble
            self.mode = "ensemble"
            self.method = None
            self.methods_list = list(methods)
            self.ensemble = ensemble
            self.ensemble_weights = ensemble_weights
            self.method_used_str = f"ensemble[{ensemble}]({','.join(methods)})"
        elif method == "quanant":
            self.mode = "quanant"
            self.method = None
            self.quanant_species = list(quanant_species or ["dqaw_lifetime"])
            self.ants_per_species = ants_per_species
            self.n_iter = n_iter
            self.evap_rate = evap_rate
            self.deposit_topk = deposit_topk
            self.aggregation = aggregation
            self.parallel = parallel
            self.max_workers = max_workers
            self.method_used_str = f"quanant({','.join(self.quanant_species)})"
        elif method == "adaptive_quanant":
            self.mode = "adaptive_quanant"
            self.method = None
            self.quanant_species = list(quanant_species or ["qsvd", "dqaw_timeavg",
                                                              "dqaw_lifetime",
                                                              "qpagerank", "heatkernel"])
            self.ants_per_species = ants_per_species
            self.n_iter = n_iter
            self.evap_rate = evap_rate
            self.deposit_topk = deposit_topk
            self.parallel = parallel
            self.max_workers = max_workers
            self.method_used_str = f"adaptive_quanant({','.join(self.quanant_species)})"
        else:
            if method not in METHOD_REGISTRY:
                raise ValueError(
                    f"Unknown method {method!r}. Choose from "
                    f"{sorted(METHOD_REGISTRY)} or 'quanant' / 'adaptive_quanant'."
                )
            self.mode = "single"
            method_kwargs = method_kwargs or {}
            self.method = METHOD_REGISTRY[method](**method_kwargs)
            self.methods_list = [method]
            self.method_used_str = method

    # ----------------------------------------------------------------
    # Fitting (for adaptive_quanant / meta_learner only)
    # ----------------------------------------------------------------
    def fit(self, apo_graph: ProteinGraph) -> "AllostericPredictor":
        """For modes that require APO supervision: train on the APO graph.

        Required for ``method='adaptive_quanant'`` or any ensemble that
        includes ``'meta_learner'``.
        """
        if self.mode == "adaptive_quanant":
            self._fitted_adaptive = AdaptiveQuanAnt(
                species=self.quanant_species,
                ants_per_species=self.ants_per_species,
                n_iter=self.n_iter,
                evap_rate=self.evap_rate,
                deposit_topk=self.deposit_topk,
                parallel=self.parallel,
                max_workers=self.max_workers,
                verbose=self.verbose,
            )
            self._fitted_adaptive.fit(apo_graph)
        if self.mode == "single" and isinstance(self.method, MetaLearner):
            self.method.fit(apo_graph)
        if self.mode == "ensemble" and "meta_learner" in self.methods_list:
            self._fitted_meta = MetaLearner().fit(apo_graph)
        return self

    # ----------------------------------------------------------------
    # Core: predict on a graph
    # ----------------------------------------------------------------
    def predict(self, graph: ProteinGraph) -> PredictionResult:
        """Run the predictor on ``graph``. Returns a :class:`PredictionResult`."""
        scores = self._compute_scores(graph)
        top_idx = graph.select_top_k(
            scores, k=self.top_k, mode=self.selection,
            mask_active=self.mask_active, only_surface=self.only_surface,
        )
        top_residues = [graph.residue_at(i) for i in top_idx]
        result = PredictionResult(
            scores=scores,
            top_indices=top_idx,
            top_residues=top_residues,
            method_used=self.method_used_str,
            graph=graph,
        )
        # If GT is available, add evaluation
        if graph.ground_truth_idx is not None:
            ev = evaluate(
                scores, graph.adjacency_weighted, graph.ground_truth_idx,
                graph.active_idx, graph.N,
                top_k=self.top_k,
                mask_radius=0 if self.mask_active else -1,
                surface_mask=graph.surface_mask if self.only_surface else None,
                coords=graph.coords if self.selection == "mmr" else None,
                selection=self.selection,
            )
            result.weighted_top5 = ev["weighted_top5"]
            result.precision_at_k = ev["precision_at_5"]
            result.hits = [
                {**graph.residue_at(p),
                 "hop_to_gt": ev["hops_to_gt"][rank],
                 "credit": ev["credit_of_top"][rank]}
                for rank, p in enumerate(ev["top_pred"])
            ]
        return result

    def _compute_scores(self, graph: ProteinGraph) -> np.ndarray:
        if self.mode == "single":
            return self.method.compute(graph)

        if self.mode == "ensemble":
            from quanallo.core.ensemble import rrf_combine, weighted_combine
            score_dict = {}
            for name in self.methods_list:
                if name == "meta_learner":
                    if self._fitted_meta is None:
                        raise RuntimeError(
                            "Ensemble contains 'meta_learner' — call .fit(apo_graph) first."
                        )
                    score_dict[name] = self._fitted_meta.compute(graph)
                else:
                    score_dict[name] = METHOD_REGISTRY[name]().compute(graph)
            if self.ensemble == "rrf":
                return rrf_combine(score_dict, weights=self.ensemble_weights)
            if self.ensemble in ("mean", "weighted"):
                w = self.ensemble_weights or {n: 1.0 for n in score_dict}
                return weighted_combine(score_dict, weights=w)
            raise ValueError(f"unknown ensemble strategy: {self.ensemble!r}")

        if self.mode == "quanant":
            colony = QuanAntColony(
                species=self.quanant_species,
                ants_per_species=self.ants_per_species,
                n_iter=self.n_iter,
                evap_rate=self.evap_rate,
                deposit_topk=self.deposit_topk,
                aggregation=self.aggregation,
                parallel=self.parallel,
                max_workers=self.max_workers,
                verbose=self.verbose,
            )
            return colony.run(graph).final_score

        if self.mode == "adaptive_quanant":
            if self._fitted_adaptive is None:
                raise RuntimeError(
                    "adaptive_quanant mode requires .fit(apo_graph) before .predict()."
                )
            return self._fitted_adaptive.predict(graph).final_score

        raise RuntimeError(f"unknown mode: {self.mode}")

    # ----------------------------------------------------------------
    # One-line convenience: PDB → prediction
    # ----------------------------------------------------------------
    def predict_from_pdb(
        self,
        apo_pdb: str | Path,
        *,
        holo_pdb: Optional[str | Path] = None,
        auto_active_site_ligand: Optional[str] = None,
        explicit_active_site: Optional[Sequence[tuple]] = None,
        holo_drug_name: Optional[str] = None,
        chains: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        **graph_kwargs,
    ) -> PredictionResult:
        """
        End-to-end: parse PDB → build graph → predict.

        Parameters
        ----------
        apo_pdb : str or Path
            Apo (unbound) PDB file.
        holo_pdb : str or Path, optional
            Holo (drug-bound) PDB. If supplied, used as the deployment target
            (predictions made on HOLO graph) and ``holo_drug_name`` defines GT.
        auto_active_site_ligand : str, optional
            HETATM ligand name on the APO structure (e.g. ``"GDP"`` for KRAS).
            Residues within 4.5 Å become the active site.
        explicit_active_site : list of (chain, resnum), optional
            Use this list instead of auto-detection.
        holo_drug_name : str, optional
            HETATM ligand on HOLO; residues within 4.5 Å become ground truth.
        chains : list of str, optional
            Subset of chains to include. Defaults to all protein chains.
        top_k : int, optional
            Override the predictor's default top_k for this call.
        **graph_kwargs :
            Extra arguments forwarded to :func:`build_graph_from_pdb`.

        Returns
        -------
        :class:`PredictionResult`
        """
        if top_k is not None:
            saved_k = self.top_k
            self.top_k = top_k

        # Build APO graph
        apo_graph = build_graph_from_pdb(
            apo_pdb,
            chains=chains,
            auto_active_site_ligand=auto_active_site_ligand,
            explicit_active_site=explicit_active_site,
            **graph_kwargs,
        )

        # If APO mode requires fitting and HOLO is present, fit on APO
        # (need APO GT for adaptive_quanant / meta_learner)
        if self.mode in ("adaptive_quanant",) or (
            self.mode == "single" and isinstance(self.method, MetaLearner)
        ) or (self.mode == "ensemble" and "meta_learner" in self.methods_list):
            # Need APO GT — use the same ligand-detection logic
            if holo_drug_name is None:
                # APO might already have GT if user passed ground_truth_ligand earlier
                if apo_graph.ground_truth_idx is None:
                    raise ValueError(
                        f"Mode {self.mode!r} requires APO GT for training. "
                        "Either supply holo_pdb + holo_drug_name (we'll re-use the "
                        "APO graph with HOLO-derived GT) or pass a graph_kwargs "
                        "argument ground_truth_ligand=."
                    )
            else:
                # Rebuild APO with GT from holo_drug_name (so methods can train)
                apo_graph = build_graph_from_pdb(
                    apo_pdb,
                    chains=chains,
                    auto_active_site_ligand=auto_active_site_ligand,
                    explicit_active_site=explicit_active_site,
                    ground_truth_ligand=holo_drug_name,
                    **graph_kwargs,
                )
            self.fit(apo_graph)

        # Determine deployment graph
        if holo_pdb is not None:
            target_graph = build_graph_from_pdb(
                holo_pdb,
                chains=chains,
                auto_active_site_ligand=auto_active_site_ligand,
                explicit_active_site=explicit_active_site,
                ground_truth_ligand=holo_drug_name,
                **graph_kwargs,
            )
        else:
            target_graph = apo_graph

        result = self.predict(target_graph)
        if top_k is not None:
            self.top_k = saved_k
        return result
