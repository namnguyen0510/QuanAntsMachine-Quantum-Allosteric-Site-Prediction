"""
QuanAllo — Quantum walker-based allosteric site prediction for proteins.

Public API:

    >>> from quanallo import AllostericPredictor, ProteinGraph
    >>> predictor = AllostericPredictor(method="dqaw_lifetime")
    >>> result = predictor.predict_from_pdb("4OBE.pdb", auto_active_site_ligand="GDP")

For lower-level control, import directly from submodules:

    >>> from quanallo.methods import DQAWLifetime
    >>> from quanallo.quanant import QuanAntColony, AdaptiveQuanAnt
"""

from quanallo._version import __version__

# High-level
from quanallo.pipeline.predictor import AllostericPredictor, PredictionResult

# Data structures
from quanallo.data.schemas import ProteinGraph

# Method registry (for convenience)
from quanallo.methods import (
    QSVD,
    DQAWTimeAvg,
    DQAWLifetime,
    QPageRank,
    HeatKernel,
    CTQW,
    CommuteTime,
    GNM,
    MetaLearner,
    METHOD_REGISTRY,
)

# QuanAnt
from quanallo.quanant.colony import QuanAntColony, AdaptiveQuanAnt

# Core utilities (frequently used)
from quanallo.core.utils import evaluate, hop_distances, fuzzy_gt_credit
from quanallo.core.selection import mmr_top_k, argmax_top_k
from quanallo.core.ensemble import rrf_combine, weighted_combine, trimmed_mean_combine

__all__ = [
    "__version__",
    # Top level
    "AllostericPredictor",
    "PredictionResult",
    # Data
    "ProteinGraph",
    # Methods
    "QSVD",
    "DQAWTimeAvg",
    "DQAWLifetime",
    "QPageRank",
    "HeatKernel",
    "CTQW",
    "CommuteTime",
    "GNM",
    "MetaLearner",
    "METHOD_REGISTRY",
    # QuanAnt
    "QuanAntColony",
    "AdaptiveQuanAnt",
    # Core utilities
    "evaluate",
    "hop_distances",
    "fuzzy_gt_credit",
    "mmr_top_k",
    "argmax_top_k",
    "rrf_combine",
    "weighted_combine",
    "trimmed_mean_combine",
]
