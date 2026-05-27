"""
Core data structures for QuanAllo.

The `ProteinGraph` class is the central object passed between methods.
It holds the residue contact graph, per-residue features, and the
active-site / optional ground-truth labels.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ProteinGraph:
    """
    A residue-level contact graph for one protein structure.

    Attributes
    ----------
    nodes : pd.DataFrame
        One row per residue. Required columns:
        ``idx, chain, resnum, resname, x, y, z``
        Optional columns (auto-populated when available):
        ``is_marked, degree, weighted_degree, sasa, is_surface, dist_to_marked``
    adjacency_binary : np.ndarray, shape (N, N)
        Binary contact matrix (1 if residues are in contact, 0 otherwise).
    adjacency_weighted : np.ndarray, shape (N, N)
        Distance-weighted contact matrix (e.g. Gaussian of pairwise distance).
    active_idx : np.ndarray, shape (n_active,)
        Indices (rows in ``nodes``) of active-site residues. Used as the
        "source" / "attractor" for prediction methods.
    ground_truth_idx : np.ndarray | None
        Optional indices of true allosteric pocket residues. Used only for
        evaluation; methods never see this during prediction.
    name : str
        Human-readable identifier (e.g. ``"KRAS_G12C_APO"``).
    coords : np.ndarray, shape (N, 3)  (computed)
    surface_mask : np.ndarray, shape (N,), dtype bool   (computed)
    """

    nodes: pd.DataFrame
    adjacency_binary: np.ndarray
    adjacency_weighted: np.ndarray
    active_idx: np.ndarray
    ground_truth_idx: Optional[np.ndarray] = None
    name: str = "protein"

    # Cached
    _coords: Optional[np.ndarray] = field(default=None, repr=False)
    _surface_mask: Optional[np.ndarray] = field(default=None, repr=False)

    # -- core invariants --
    def __post_init__(self) -> None:
        N = len(self.nodes)
        if self.adjacency_binary.shape != (N, N):
            raise ValueError(
                f"adjacency_binary shape {self.adjacency_binary.shape} != ({N},{N})"
            )
        if self.adjacency_weighted.shape != (N, N):
            raise ValueError(
                f"adjacency_weighted shape {self.adjacency_weighted.shape} != ({N},{N})"
            )
        self.active_idx = np.asarray(self.active_idx, dtype=int)
        if self.ground_truth_idx is not None:
            self.ground_truth_idx = np.asarray(self.ground_truth_idx, dtype=int)

    # -- convenience properties --
    @property
    def N(self) -> int:
        """Number of residues."""
        return len(self.nodes)

    @property
    def coords(self) -> np.ndarray:
        """N×3 Cα coordinates."""
        if self._coords is None:
            self._coords = self.nodes[["x", "y", "z"]].values
        return self._coords

    @property
    def surface_mask(self) -> np.ndarray:
        """Boolean mask of surface residues (from ``nodes['is_surface']``).
        If absent, returns all-True (no filter)."""
        if self._surface_mask is None:
            if "is_surface" in self.nodes.columns:
                self._surface_mask = self.nodes["is_surface"].values.astype(bool)
            else:
                self._surface_mask = np.ones(self.N, dtype=bool)
        return self._surface_mask

    # -- helpers --
    def residue_at(self, idx: int) -> dict:
        """Get residue metadata as a plain dict for index ``idx``."""
        row = self.nodes.iloc[int(idx)]
        return {
            "idx": int(row.idx),
            "chain": str(row.chain),
            "resnum": int(row.resnum),
            "resname": str(row.resname),
        }

    def indices_for(self, residue_keys: Sequence[tuple]) -> np.ndarray:
        """Look up ``idx`` values from a list of ``(chain, resnum)`` tuples."""
        key = self.nodes.set_index(["chain", "resnum"])["idx"]
        out = []
        for c, r in residue_keys:
            if (c, int(r)) in key.index:
                out.append(int(key.loc[(c, int(r))]))
        return np.asarray(out, dtype=int)

    def select_top_k(
        self,
        scores: np.ndarray,
        k: int = 5,
        mode: str = "argmax",
        mask_active: bool = True,
        only_surface: bool = True,
        lambda_div: float = 0.4,
    ) -> list[int]:
        """
        Select top-k residue indices from a score vector.

        Parameters
        ----------
        scores : array, shape (N,)
            Per-residue score (higher = better).
        k : int
            Number of residues to return.
        mode : {"argmax", "mmr"}
            Selection strategy. ``"mmr"`` enforces 3D diversity.
        mask_active : bool
            If True, exclude active-site residues from the pool.
        only_surface : bool
            If True, restrict to surface residues.
        lambda_div : float
            (mmr only) Diversity weight in [0, 1].

        Returns
        -------
        list[int] : the chosen residue indices.
        """
        from quanallo.core.selection import argmax_top_k, mmr_top_k

        s = np.asarray(scores, dtype=float).copy()
        if mask_active:
            s[self.active_idx] = -np.inf
        if only_surface:
            s[~self.surface_mask] = -np.inf

        if mode == "argmax":
            return argmax_top_k(s, k=k)
        if mode == "mmr":
            return mmr_top_k(s, self.coords, k=k, lambda_div=lambda_div)
        raise ValueError(f"unknown mode={mode!r}; use 'argmax' or 'mmr'")

    # -- IO --
    def save(self, directory: str | Path) -> None:
        """Persist the graph to ``directory`` as nodes.csv + npy files."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.nodes.to_csv(d / "nodes.csv", index=False)
        np.save(d / "adjacency_binary.npy", self.adjacency_binary)
        np.save(d / "adjacency_weighted.npy", self.adjacency_weighted)
        np.save(d / "active_idx.npy", self.active_idx)
        if self.ground_truth_idx is not None:
            np.save(d / "ground_truth_idx.npy", self.ground_truth_idx)
        with open(d / "name.txt", "w") as fh:
            fh.write(self.name)

    @classmethod
    def load(cls, directory: str | Path) -> "ProteinGraph":
        """Load a graph saved by :meth:`save`."""
        d = Path(directory)
        nodes = pd.read_csv(d / "nodes.csv")
        adj_b = np.load(d / "adjacency_binary.npy")
        adj_w = np.load(d / "adjacency_weighted.npy")
        active = np.load(d / "active_idx.npy")
        gt = None
        if (d / "ground_truth_idx.npy").exists():
            gt = np.load(d / "ground_truth_idx.npy")
        name = "protein"
        if (d / "name.txt").exists():
            name = (d / "name.txt").read_text().strip() or "protein"
        return cls(
            nodes=nodes,
            adjacency_binary=adj_b,
            adjacency_weighted=adj_w,
            active_idx=active,
            ground_truth_idx=gt,
            name=name,
        )

    # -- factory: from PDB --
    @classmethod
    def from_pdb(cls, pdb_path: str | Path, **kwargs) -> "ProteinGraph":
        """
        Convenience: parse a PDB and build a graph in one call.

        See :func:`quanallo.data.extraction.build_graph_from_pdb` for keyword
        arguments (active-site detection, residue range, contact radius, ...).
        """
        from quanallo.data.extraction import build_graph_from_pdb

        return build_graph_from_pdb(pdb_path, **kwargs)

    def __repr__(self) -> str:
        gt = "no GT" if self.ground_truth_idx is None else f"|GT|={len(self.ground_truth_idx)}"
        return (
            f"ProteinGraph(name={self.name!r}, N={self.N}, "
            f"|active|={len(self.active_idx)}, {gt})"
        )
