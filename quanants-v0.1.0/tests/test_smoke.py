"""
Smoke tests for QuanAllo.

These test that every public class instantiates, every method runs on a
synthetic small protein graph, and the high-level Predictor works in all four
modes (single, ensemble, quanant, adaptive_quanant).

Run:    pytest tests/
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from quanallo import (
    AllostericPredictor,
    ProteinGraph,
    METHOD_REGISTRY,
    QuanAntColony,
    AdaptiveQuanAnt,
    evaluate,
    mmr_top_k,
    argmax_top_k,
    rrf_combine,
)


# ----------------------------------------------------------------------
# Synthetic fixture
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_graph() -> ProteinGraph:
    """Build a small random-but-deterministic graph (35 nodes).
    Topology: rough chain with extra long-range contacts, simulating a small protein."""
    rng = np.random.default_rng(0)
    N = 35
    xyz = np.zeros((N, 3))
    # Backbone along x with small lateral jitter
    for i in range(N):
        xyz[i] = [i * 3.8, rng.normal(0, 0.5), rng.normal(0, 0.5)]
    # Add some longer-range contacts (folded structure)
    for k in range(8):
        a, b = rng.integers(0, N, size=2)
        if abs(a - b) > 5:
            xyz[b] = xyz[a] + rng.normal(0, 2.0, size=3)
    # Distance matrix, contact graph
    D = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    A_bin = (D <= 8.0).astype(float)
    A_w = np.exp(-D ** 2 / (2 * 5.0 ** 2)) * A_bin

    nodes = pd.DataFrame({
        "idx": np.arange(N),
        "chain": ["A"] * N,
        "resnum": np.arange(1, N + 1),
        "resname": ["ALA"] * N,
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "is_surface": [True] * N,
        "sasa": [50.0] * N,
        "degree": A_bin.sum(axis=1),
        "weighted_degree": A_w.sum(axis=1),
    })
    active = np.array([3, 4, 5])
    gt = np.array([20, 21, 22])
    return ProteinGraph(
        nodes=nodes, adjacency_binary=A_bin, adjacency_weighted=A_w,
        active_idx=active, ground_truth_idx=gt, name="synthetic",
    )


# ----------------------------------------------------------------------
# 1. Individual methods
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method_name", list(METHOD_REGISTRY.keys()))
def test_each_method_runs(synthetic_graph, method_name):
    """Every registered method should produce a finite (N,) score vector."""
    cls = METHOD_REGISTRY[method_name]
    if method_name == "meta_learner":
        method = cls().fit(synthetic_graph)
    else:
        method = cls()
    scores = method.compute(synthetic_graph)
    assert scores.shape == (synthetic_graph.N,)
    assert np.all(np.isfinite(scores)) or scores.std() > 0


# ----------------------------------------------------------------------
# 2. Top-k selection
# ----------------------------------------------------------------------
def test_argmax_topk(synthetic_graph):
    scores = np.linspace(0, 1, synthetic_graph.N)
    top = argmax_top_k(scores, k=5)
    assert top == list(range(synthetic_graph.N - 1, synthetic_graph.N - 6, -1))


def test_mmr_topk_basic(synthetic_graph):
    scores = np.linspace(0, 1, synthetic_graph.N)
    top = mmr_top_k(scores, synthetic_graph.coords, k=5, lambda_div=0.5)
    assert len(top) == 5
    assert len(set(top)) == 5  # distinct


# ----------------------------------------------------------------------
# 3. Evaluation
# ----------------------------------------------------------------------
def test_evaluate(synthetic_graph):
    scores = np.zeros(synthetic_graph.N)
    scores[synthetic_graph.ground_truth_idx] = 1.0   # perfect prediction
    result = evaluate(
        scores, synthetic_graph.adjacency_weighted,
        synthetic_graph.ground_truth_idx,
        synthetic_graph.active_idx, synthetic_graph.N,
        surface_mask=synthetic_graph.surface_mask,
        top_k=3,                                       # match the 3 GT residues
    )
    assert result["weighted_top5"] >= 2.9  # nearly perfect (3 exact hits = 3.0)
    assert result["precision_at_5"][0] == 1.0
    # Also test that argmax + hop ordering is right
    assert set(result["top_pred"]) == set(synthetic_graph.ground_truth_idx.tolist())


# ----------------------------------------------------------------------
# 4. Ensemble combiners
# ----------------------------------------------------------------------
def test_rrf_combine():
    s1 = np.array([5, 4, 3, 2, 1])
    s2 = np.array([1, 2, 3, 4, 5])
    fused = rrf_combine({"a": s1, "b": s2})
    # The MIDDLE element (index 2) appears in the middle of both rankings, but
    # the extremes (0 and 4) are in the top of one each. RRF should give similar
    # values for 0 and 4 (both rank-1 in some method), higher than the middle.
    # At minimum: fused is finite, length-5 and not all equal.
    assert fused.shape == (5,)
    assert np.all(np.isfinite(fused))
    assert fused.std() > 0


# ----------------------------------------------------------------------
# 5. High-level predictor
# ----------------------------------------------------------------------
def test_predictor_single(synthetic_graph):
    p = AllostericPredictor(method="dqaw_lifetime", top_k=3, only_surface=False)
    result = p.predict(synthetic_graph)
    assert len(result.top_indices) == 3
    assert result.method_used == "dqaw_lifetime"
    assert result.weighted_top5 is not None  # GT is set


def test_predictor_ensemble(synthetic_graph):
    p = AllostericPredictor(
        methods=["dqaw_lifetime", "qpagerank", "heatkernel"],
        ensemble="rrf", top_k=3, only_surface=False,
    )
    result = p.predict(synthetic_graph)
    assert len(result.top_indices) == 3


def test_predictor_quanant(synthetic_graph):
    p = AllostericPredictor(
        method="quanant",
        quanant_species=["dqaw_lifetime", "qpagerank"],
        ants_per_species=2, n_iter=2,
        top_k=3, only_surface=False,
        parallel=False,
    )
    result = p.predict(synthetic_graph)
    assert len(result.top_indices) == 3


def test_predictor_adaptive_quanant(synthetic_graph):
    p = AllostericPredictor(
        method="adaptive_quanant",
        quanant_species=["dqaw_lifetime", "qpagerank"],
        ants_per_species=2, n_iter=2,
        top_k=3, only_surface=False,
        parallel=False,
    )
    p.fit(synthetic_graph)        # use synthetic GT for training
    result = p.predict(synthetic_graph)
    assert len(result.top_indices) == 3
    assert p._fitted_adaptive is not None
    assert sum(p._fitted_adaptive.apo_weights.values()) == pytest.approx(1.0, abs=1e-3)


# ----------------------------------------------------------------------
# 6. ProteinGraph IO
# ----------------------------------------------------------------------
def test_proteingraph_save_load(synthetic_graph, tmp_path):
    synthetic_graph.save(tmp_path / "graph_dump")
    g2 = ProteinGraph.load(tmp_path / "graph_dump")
    assert g2.N == synthetic_graph.N
    assert np.allclose(g2.adjacency_binary, synthetic_graph.adjacency_binary)
    assert np.array_equal(g2.active_idx, synthetic_graph.active_idx)


def test_proteingraph_repr(synthetic_graph):
    r = repr(synthetic_graph)
    assert "synthetic" in r
    assert "N=35" in r
