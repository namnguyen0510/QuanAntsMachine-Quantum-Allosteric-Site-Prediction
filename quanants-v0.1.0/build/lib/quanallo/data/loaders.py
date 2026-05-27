"""
Loaders for pre-computed protein graph data.

Useful when you've already run a preprocessing pipeline (e.g.
``stage1_consolidated.py``) and have CSV/NPY files on disk.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from quanallo.data.schemas import ProteinGraph


def load_from_csv(
    nodes_csv: str | Path,
    adjacency_npy: Optional[str | Path] = None,
    weights_npy: Optional[str | Path] = None,
    active_site_csv: Optional[str | Path] = None,
    ground_truth_csv: Optional[str | Path] = None,
    name: str = "protein",
) -> ProteinGraph:
    """
    Load a :class:`ProteinGraph` from on-disk CSV/NPY artefacts.

    Mirrors the layout produced by ``stage1_consolidated.py``:

    - ``nodes.csv``: required, with columns idx, chain, resnum, resname, x, y, z, ...
    - ``adjacency.npy``: binary contact matrix
    - ``weights.npy``: distance-weighted contact matrix
    - ``active_site.csv``: chain,resnum,resname  → marks active site
    - ``ground_truth.csv``: chain,resnum,resname → marks GT pocket (optional)
    """
    nodes_csv = Path(nodes_csv)
    nodes = pd.read_csv(nodes_csv)
    N = len(nodes)

    if adjacency_npy is None:
        adjacency_npy = nodes_csv.parent / "adjacency.npy"
    if weights_npy is None:
        weights_npy = nodes_csv.parent / "weights.npy"

    A_bin = np.load(adjacency_npy).astype(float)
    if Path(weights_npy).exists():
        A_w = np.load(weights_npy)
    else:
        A_w = A_bin.copy()

    # Active site
    if active_site_csv is None:
        guess = nodes_csv.parent / "active_site.csv"
        if guess.exists():
            active_site_csv = guess
    active_idx = np.array([], dtype=int)
    if active_site_csv is not None and Path(active_site_csv).exists():
        df = pd.read_csv(active_site_csv)
        lookup = nodes.set_index(["chain", "resnum"])["idx"]
        ids = []
        for _, r in df.iterrows():
            key = (r.chain, int(r.resnum))
            if key in lookup.index:
                ids.append(int(lookup.loc[key]))
        active_idx = np.asarray(ids, dtype=int)

    # Ground truth (optional)
    gt_idx = None
    if ground_truth_csv is None:
        guess = nodes_csv.parent / "ground_truth.csv"
        if guess.exists():
            ground_truth_csv = guess
    if ground_truth_csv is not None and Path(ground_truth_csv).exists():
        df = pd.read_csv(ground_truth_csv)
        lookup = nodes.set_index(["chain", "resnum"])["idx"]
        ids = []
        for _, r in df.iterrows():
            key = (r.chain, int(r.resnum))
            if key in lookup.index:
                ids.append(int(lookup.loc[key]))
        gt_idx = np.asarray(ids, dtype=int) if ids else None

    return ProteinGraph(
        nodes=nodes,
        adjacency_binary=A_bin,
        adjacency_weighted=A_w,
        active_idx=active_idx,
        ground_truth_idx=gt_idx,
        name=name,
    )
