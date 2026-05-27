"""
Abstract base class for all allosteric-prediction methods.

Every method exposes the same interface:

    method = SomeMethod(**params)
    scores = method.compute(graph)        # -> (N,) per-residue scores

For methods that support pheromone-aware operation (used inside QuanAnt
colonies), they additionally implement:

    scores = method.compute_with_pheromone(graph, pheromone)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from quanallo.data.schemas import ProteinGraph


@dataclass
class AllosteryMethod(ABC):
    """
    Base class for allosteric-prediction methods.

    Subclasses must implement :meth:`compute`. They may optionally override
    :meth:`compute_with_pheromone` to participate in pheromone-aware colonies;
    the default implementation ignores the pheromone field.

    Attributes
    ----------
    name : str
        Short identifier (e.g. ``"dqaw_lifetime"``).
    kind : {"quantum", "quantum_inspired", "classical", "hybrid"}
        Categorisation for plotting / colour-coding.
    requires_active_site : bool
        Whether the method needs ``graph.active_idx`` to be populated.
    """

    name: str = "abstract"
    kind: str = "quantum"
    requires_active_site: bool = True

    @abstractmethod
    def compute(self, graph: ProteinGraph) -> np.ndarray:
        """Return a per-residue score vector of shape (graph.N,).

        Higher score = more likely to be an allosteric residue.
        """
        ...

    def compute_with_pheromone(
        self,
        graph: ProteinGraph,
        pheromone: np.ndarray,
        *,
        pher_strength: float = 1.0,
    ) -> np.ndarray:
        """Pheromone-aware variant. Default: ignore pheromone (fallback).

        Subclasses with pheromone support override this method.
        """
        return self.compute(graph)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, kind={self.kind!r})"
