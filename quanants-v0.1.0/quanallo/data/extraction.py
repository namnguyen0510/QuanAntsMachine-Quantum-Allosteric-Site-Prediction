"""
Build a :class:`~quanallo.ProteinGraph` from a PDB file.

This module is the QuanAllo equivalent of ``stage1_consolidated.py`` — it parses
the structure, extracts residue Cα coordinates, builds the contact graph,
computes per-residue SASA/surface flag, and detects the active-site residues
via one of several supported strategies.

Public API
----------
- :func:`build_graph_from_pdb` — high-level: PDB → ProteinGraph
- :func:`detect_active_site` — flexible active-site detection
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")

# Optional BioPython import — guarded
try:
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    _HAVE_BIOPYTHON = True
except ImportError:  # pragma: no cover
    _HAVE_BIOPYTHON = False


# ----------------------------------------------------------------------
# Constants (same conventions as stage1_consolidated.py)
# ----------------------------------------------------------------------
PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
NUCLEIC_RESIDUES = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "T", "DI"}
IONS_AND_BUFFERS = {
    "HOH", "MG", "NA", "CL", "K", "ZN", "CA", "MN", "SO4", "PO4",
    "BR", "I", "CD", "CO", "CU", "FE", "NI",
}


def _require_biopython():
    if not _HAVE_BIOPYTHON:
        raise ImportError(
            "BioPython is required for PDB extraction. Install with: "
            "pip install biopython"
        )


# ----------------------------------------------------------------------
# Parsing & extraction
# ----------------------------------------------------------------------
def parse_pdb(path: str | Path):
    """Parse a PDB file with BioPython. Returns ``Structure`` object."""
    _require_biopython()
    return PDBParser(QUIET=True).get_structure("x", str(path))


def get_ca_table(structure, chains: Sequence[str],
                 resnum_range: Optional[tuple] = None) -> pd.DataFrame:
    """Extract per-residue Cα coordinates as a DataFrame with columns
    ``idx, chain, resnum, resname, x, y, z``."""
    rows = []
    for ch in structure[0]:
        if ch.id not in chains:
            continue
        for res in ch:
            if res.resname.strip() not in PROTEIN_RESIDUES:
                continue
            if "CA" not in res:
                continue
            if resnum_range is not None:
                lo, hi = resnum_range
                if not (lo <= res.id[1] <= hi):
                    continue
            ca = res["CA"]
            rows.append({
                "chain": ch.id,
                "resnum": int(res.id[1]),
                "resname": res.resname.strip(),
                "x": float(ca.coord[0]),
                "y": float(ca.coord[1]),
                "z": float(ca.coord[2]),
            })
    df = pd.DataFrame(rows).reset_index(drop=True)
    df.insert(0, "idx", df.index)
    return df


def build_contact_graph(nodes: pd.DataFrame,
                         r_cutoff: float = 8.0,
                         weighting: str = "gaussian",
                         sigma: float = 5.0):
    """Build binary and weighted adjacency matrices from Cα coords.

    Parameters
    ----------
    nodes : DataFrame with columns ``x, y, z``
    r_cutoff : float
        Contact radius in Å. Pairs ≤ ``r_cutoff`` are considered in contact.
    weighting : {"binary", "gaussian"}
        ``"binary"`` → all contacts weight 1.
        ``"gaussian"`` → weight = exp(-d²/2σ²) for d ≤ r_cutoff, else 0.
    sigma : float
        Gaussian width parameter (Å).

    Returns
    -------
    (A_binary, A_weighted, D) : tuple of np.ndarray
    """
    coords = nodes[["x", "y", "z"]].values
    D = squareform(pdist(coords))
    np.fill_diagonal(D, np.inf)
    A = (D <= r_cutoff).astype(int)
    if weighting == "binary":
        W = A.astype(float)
    elif weighting == "gaussian":
        W = np.exp(-D**2 / (2 * sigma**2)) * A
    else:
        raise ValueError(f"weighting must be 'binary' or 'gaussian', got {weighting!r}")
    return A, W, D


def compute_sasa(structure, chains: Sequence[str],
                 probe_radius: float = 1.4) -> dict:
    """Compute per-residue SASA via the Shrake-Rupley algorithm.
    Returns a dict mapping ``(chain, resnum) -> SASA``."""
    _require_biopython()
    ShrakeRupley(probe_radius=probe_radius).compute(structure[0], level="R")
    out = {}
    for ch in structure[0]:
        if ch.id not in chains:
            continue
        for res in ch:
            if res.resname.strip() in PROTEIN_RESIDUES:
                out[(ch.id, res.id[1])] = float(res.sasa)
    return out


# ----------------------------------------------------------------------
# Active-site / ground-truth detection
# ----------------------------------------------------------------------
def _residues_near_atoms(structure, protein_chains, atom_coords, radius):
    if len(atom_coords) == 0:
        return []
    coords = np.asarray(atom_coords)
    r2 = radius ** 2
    hits = []
    for chain in structure[0]:
        if chain.id not in protein_chains:
            continue
        for residue in chain:
            if residue.resname.strip() not in PROTEIN_RESIDUES:
                continue
            for atom in residue:
                if (np.sum((coords - atom.coord) ** 2, axis=1).min()) <= r2:
                    hits.append((chain.id, int(residue.id[1]),
                                 residue.resname.strip()))
                    break
    return hits


def detect_active_site(
    structure,
    chains: Sequence[str],
    *,
    ligand_name: Optional[str] = None,
    explicit_residues: Optional[Sequence[tuple]] = None,
    dna_contact: bool = False,
    radius: float = 4.5,
) -> list[tuple]:
    """Detect active-site residues by one of three strategies.

    Exactly one of the three keyword arguments must be supplied.

    Parameters
    ----------
    ligand_name : str
        Detect residues with any atom within ``radius`` of any atom of a HETATM
        residue named ``ligand_name`` (e.g. ``"GDP"``, ``"MOV"``).
    explicit_residues : list of (chain, resnum)
        Use a user-supplied list of residue identifiers directly.
    dna_contact : bool
        Detect residues within ``radius`` of any DNA/RNA atom (for transcription
        factors like c-Myc).

    Returns
    -------
    list of (chain, resnum, resname) tuples.
    """
    _require_biopython()
    given = sum([ligand_name is not None, explicit_residues is not None, dna_contact])
    if given != 1:
        raise ValueError(
            "Specify exactly one of: ligand_name, explicit_residues, dna_contact"
        )

    if ligand_name is not None:
        lig = ligand_name.upper()
        coords = []
        for chain in structure[0]:
            for res in chain:
                if res.resname.strip().upper() == lig:
                    for atom in res:
                        coords.append(atom.coord)
        if not coords:
            warnings.warn(f"No HETATM residue named {lig!r} found in structure.")
        return _residues_near_atoms(structure, chains, coords, radius)

    if explicit_residues is not None:
        lookup = {}
        for chain in structure[0]:
            if chain.id not in chains:
                continue
            for res in chain:
                if res.resname.strip() in PROTEIN_RESIDUES:
                    lookup[(chain.id, int(res.id[1]))] = res.resname.strip()
        out = []
        for entry in explicit_residues:
            c, n = entry[0], int(entry[1])
            if (c, n) in lookup:
                out.append((c, n, lookup[(c, n)]))
        return out

    if dna_contact:
        coords = []
        for chain in structure[0]:
            for res in chain:
                if res.resname.strip() in NUCLEIC_RESIDUES:
                    for atom in res:
                        coords.append(atom.coord)
        return _residues_near_atoms(structure, chains, coords, radius)
    return []  # unreachable


# ----------------------------------------------------------------------
# High-level: PDB → ProteinGraph
# ----------------------------------------------------------------------
def build_graph_from_pdb(
    pdb_path: str | Path,
    *,
    chains: Optional[Sequence[str]] = None,
    auto_active_site_ligand: Optional[str] = None,
    explicit_active_site: Optional[Sequence[tuple]] = None,
    dna_contact_active_site: bool = False,
    active_site_radius: float = 4.5,
    ground_truth_ligand: Optional[str] = None,
    ground_truth_radius: float = 4.5,
    resnum_range: Optional[tuple] = None,
    r_cutoff: float = 8.0,
    weighting: str = "gaussian",
    sigma: float = 5.0,
    sasa_probe_radius: float = 1.4,
    surface_threshold: float = 20.0,
    name: Optional[str] = None,
) -> "ProteinGraph":
    """
    Build a fully-populated :class:`ProteinGraph` from a PDB file.

    Parameters
    ----------
    pdb_path : str or Path
        Path to the PDB file.
    chains : list of str, optional
        Chain IDs to keep. If ``None``, keeps all protein chains.
    auto_active_site_ligand : str, optional
        Auto-detect active site as residues within ``active_site_radius`` Å of
        any HETATM residue with this name (e.g. ``"GDP"`` for KRAS).
    explicit_active_site : list of (chain, resnum), optional
        Use these residues as the active site directly.
    dna_contact_active_site : bool
        If True, define active site as DNA-contacting residues.
    active_site_radius : float
        Distance cutoff for ligand-proximity detection.
    ground_truth_ligand : str, optional
        If supplied, also detect ground-truth residues (e.g. drug-binding site
        in HOLO structure). Stored as ``graph.ground_truth_idx``.
    ground_truth_radius : float
        Distance cutoff for ground-truth detection.
    resnum_range : (lo, hi) tuple, optional
        Only keep residues whose resnum is in [lo, hi].
    r_cutoff : float
        Contact-graph distance cutoff (Å). Defaults to 8.0.
    weighting : {"binary", "gaussian"}
        Edge-weighting scheme.
    sigma : float
        Gaussian width (only used for ``weighting="gaussian"``).
    sasa_probe_radius : float
        Solvent probe radius for SASA computation.
    surface_threshold : float
        Minimum SASA (Å²) to count a residue as surface-exposed.
    name : str, optional
        Identifier for the graph. Defaults to the PDB filename stem.

    Returns
    -------
    ProteinGraph
    """
    from quanallo.data.schemas import ProteinGraph

    pdb_path = Path(pdb_path)
    structure = parse_pdb(pdb_path)
    if name is None:
        name = pdb_path.stem

    # Determine chains
    if chains is None:
        # Keep all chains with at least 5 protein residues
        chains = []
        for ch in structure[0]:
            count = sum(1 for r in ch if r.resname.strip() in PROTEIN_RESIDUES)
            if count >= 5:
                chains.append(ch.id)
    chains = list(chains)

    # Cα table
    nodes = get_ca_table(structure, chains, resnum_range=resnum_range)
    if len(nodes) == 0:
        raise ValueError(f"No Cα atoms found in {pdb_path} for chains {chains}")

    # Contact graph
    A_bin, A_w, D = build_contact_graph(nodes, r_cutoff=r_cutoff,
                                          weighting=weighting, sigma=sigma)

    # SASA + surface flag + degree
    sasa_map = compute_sasa(structure, chains, probe_radius=sasa_probe_radius)
    nodes["sasa"] = nodes.apply(
        lambda r: sasa_map.get((r.chain, int(r.resnum)), float("nan")), axis=1
    )
    nodes["is_surface"] = nodes["sasa"] >= surface_threshold
    nodes["degree"] = A_bin.sum(axis=1)
    nodes["weighted_degree"] = A_w.sum(axis=1)

    # Active-site detection
    active_residues = []
    if auto_active_site_ligand is not None:
        active_residues = detect_active_site(
            structure, chains, ligand_name=auto_active_site_ligand,
            radius=active_site_radius
        )
    elif explicit_active_site is not None:
        active_residues = detect_active_site(
            structure, chains, explicit_residues=explicit_active_site
        )
    elif dna_contact_active_site:
        active_residues = detect_active_site(
            structure, chains, dna_contact=True, radius=active_site_radius
        )
    else:
        warnings.warn(
            "No active-site detection mode specified. "
            "Graph created without active_idx — most methods will fail. "
            "Pass auto_active_site_ligand=, explicit_active_site=, or "
            "dna_contact_active_site=True.",
            stacklevel=2,
        )

    # Filter active residues by resnum_range if needed
    if resnum_range is not None:
        lo, hi = resnum_range
        active_residues = [(c, n, rn) for (c, n, rn) in active_residues
                           if lo <= n <= hi]

    # Map active residues to indices
    nodes_lookup = nodes.set_index(["chain", "resnum"])["idx"]
    active_idx = []
    for (c, n, _) in active_residues:
        if (c, n) in nodes_lookup.index:
            active_idx.append(int(nodes_lookup.loc[(c, n)]))
    active_idx = np.asarray(active_idx, dtype=int)

    # Mark active residues
    nodes["is_marked"] = nodes["idx"].isin(active_idx.tolist())

    # Distance to nearest active residue (3D, not graph hops)
    if len(active_idx) > 0:
        d2 = D[:, active_idx].min(axis=1)
        d2[active_idx] = 0.0
        nodes["dist_to_marked"] = d2
    else:
        nodes["dist_to_marked"] = float("nan")

    # Optional ground-truth detection
    gt_idx = None
    if ground_truth_ligand is not None:
        gt_residues = detect_active_site(
            structure, chains, ligand_name=ground_truth_ligand,
            radius=ground_truth_radius,
        )
        if resnum_range is not None:
            lo, hi = resnum_range
            gt_residues = [(c, n, rn) for (c, n, rn) in gt_residues
                            if lo <= n <= hi]
        gt = []
        for (c, n, _) in gt_residues:
            if (c, n) in nodes_lookup.index:
                gt.append(int(nodes_lookup.loc[(c, n)]))
        gt_idx = np.asarray(gt, dtype=int) if gt else None

    return ProteinGraph(
        nodes=nodes,
        adjacency_binary=A_bin.astype(float),
        adjacency_weighted=A_w,
        active_idx=active_idx,
        ground_truth_idx=gt_idx,
        name=name,
    )
