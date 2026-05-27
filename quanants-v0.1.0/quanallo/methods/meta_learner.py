"""
MetaLearner — supervised hybrid classifier on quantum + geometric features.

Trains a regularized logistic regression on per-residue features:

  * Quantum scores: QSVD, DQAW-TimeAvg, DQAW-Lifetime, QPageRank
  * Geometric: hop-to-active, SASA, surface flag, degree, 3D distance to
    active centroid, surface-neighbour density

Target = ground-truth membership on the training graph.

Typical use: ``fit(apo_graph)`` (uses APO GT), then ``predict(holo_graph)``.
This is the "supervised meta-learner" from v4 of the development.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from quanallo.methods.base import AllosteryMethod
from quanallo.data.schemas import ProteinGraph
from quanallo.core.utils import hop_distances


# ----------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------
def build_feature_matrix(graph: ProteinGraph,
                          quantum_scores: dict) -> np.ndarray:
    """
    Build per-residue feature matrix.

    Parameters
    ----------
    graph : ProteinGraph
    quantum_scores : dict
        Must contain keys ``qsvd``, ``dqaw_t``, ``dqaw_l``, ``qpr``.

    Returns
    -------
    (N, F) array — F = 10 features.
    """
    A = graph.adjacency_weighted
    N = graph.N
    active = graph.active_idx
    nodes = graph.nodes
    xyz = graph.coords

    h_act = hop_distances(A, active, N).astype(float)
    sasa = nodes["sasa"].values.astype(float) if "sasa" in nodes.columns else np.zeros(N)
    surf = nodes["is_surface"].values.astype(float) if "is_surface" in nodes.columns else np.zeros(N)
    deg = nodes["degree"].values.astype(float) if "degree" in nodes.columns else A.sum(axis=1)
    if len(active) > 0:
        act_centroid = xyz[active].mean(axis=0)
    else:
        act_centroid = xyz.mean(axis=0)
    dist3d = np.linalg.norm(xyz - act_centroid, axis=1)
    surf_neigh = np.zeros(N)
    surf_mask = surf.astype(bool)
    for i in range(N):
        d = np.linalg.norm(xyz - xyz[i], axis=1)
        surf_neigh[i] = ((d < 8.0) & surf_mask).sum()

    return np.column_stack([
        quantum_scores.get("qsvd", np.zeros(N)),    # 0
        quantum_scores.get("dqaw_t", np.zeros(N)),  # 1
        quantum_scores.get("dqaw_l", np.zeros(N)),  # 2
        quantum_scores.get("qpr", np.zeros(N)),     # 3
        h_act,                                       # 4
        sasa,                                        # 5
        surf,                                        # 6
        deg,                                         # 7
        dist3d,                                      # 8
        surf_neigh,                                  # 9
    ])


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
@dataclass
class MetaLearner(AllosteryMethod):
    """
    Logistic regression on quantum + geometric features.

    Workflow
    --------
    1. ``compute_quantum_features(graph)`` runs QSVD, DQAW-TimeAvg, DQAW-Lifetime,
       QPageRank to get per-residue feature scores.
    2. :meth:`fit` trains LogReg on a graph with ``ground_truth_idx`` populated.
    3. :meth:`compute` returns predicted probabilities on any other graph.

    Parameters
    ----------
    C : float
        Inverse-regularization strength for LogisticRegression.
    feature_methods : list of str, optional
        Subset of ['qsvd', 'dqaw_t', 'dqaw_l', 'qpr'] to include. Default = all.
    exclude_active_from_training : bool
        If True (default), active-site residues are excluded from the training set.
    """

    C: float = 0.5
    feature_methods: tuple = ("qsvd", "dqaw_t", "dqaw_l", "qpr")
    exclude_active_from_training: bool = True
    name: str = "meta_learner"
    kind: str = "hybrid"
    requires_active_site: bool = True

    # Trained state (set after fit())
    _clf: Optional[LogisticRegression] = field(default=None, repr=False)
    _scaler: Optional[StandardScaler] = field(default=None, repr=False)

    @staticmethod
    def compute_quantum_features(graph: ProteinGraph) -> dict:
        """Run the underlying quantum methods (default hyperparameters) and
        return ``{name: score_vector}``."""
        from quanallo.methods.qsvd import QSVD
        from quanallo.methods.dqaw import DQAWTimeAvg, DQAWLifetime
        from quanallo.methods.qpagerank import QPageRank
        return {
            "qsvd":   QSVD().compute(graph),
            "dqaw_t": DQAWTimeAvg().compute(graph),
            "dqaw_l": DQAWLifetime().compute(graph),
            "qpr":    QPageRank().compute(graph),
        }

    def fit(self, graph: ProteinGraph) -> "MetaLearner":
        """Train the underlying LogReg on ``graph``. Requires
        ``graph.ground_truth_idx`` to be populated."""
        if graph.ground_truth_idx is None:
            raise ValueError(
                "MetaLearner.fit requires graph.ground_truth_idx to be set."
            )
        features = self.compute_quantum_features(graph)
        X = build_feature_matrix(graph, features)
        y = np.zeros(graph.N, dtype=int)
        y[graph.ground_truth_idx] = 1

        train_mask = np.ones(graph.N, dtype=bool)
        if self.exclude_active_from_training:
            train_mask[graph.active_idx] = False
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X[train_mask])
        self._clf = LogisticRegression(
            class_weight="balanced", C=self.C,
            max_iter=2000, solver="lbfgs",
        )
        self._clf.fit(Xs, y[train_mask])
        return self

    def compute(self, graph: ProteinGraph) -> np.ndarray:
        if self._clf is None or self._scaler is None:
            raise RuntimeError(
                "MetaLearner.compute called before fit. Call .fit(apo_graph) "
                "with a ground-truth-annotated graph first."
            )
        features = self.compute_quantum_features(graph)
        X = build_feature_matrix(graph, features)
        Xs = self._scaler.transform(X)
        return self._clf.predict_proba(Xs)[:, 1]
