"""
run_full_pipeline.py — comprehensive QuanAllo evaluation on KRAS G12C
======================================================================

End-to-end pipeline showcasing every capability of the quanallo package:

  1. Hyperparameter grid-search per method (APO GT supervised).
  2. Default vs. tuned single-species comparison on APO + HOLO.
  3. APO-HOLO transferability scatter (which methods generalize?).
  4. Species selection — forward greedy (legitimate) + all-subsets (oracle).
  5. Ensemble comparison — RRF, weighted-mean, trimmed-mean.
  6. QuanAnt machines — single, heterogeneous shared, multi-pheromone.
  7. Adaptive QuanAnt — APO-trained species weights + online updates.
  8. **Submission folder** — APO + HOLO score matrices and best-model hitlist.

Outputs (under `OUT/`):

  matrices/
    long_form.csv               every evaluation result in long form
    pivot_summary.csv           best method per stage (wide form)
    score_matrix_apo.npy        N×M score matrix on APO
    score_matrix_holo.npy       N×M score matrix on HOLO
    tuned_hyperparams.json      best params per method
  hit_lists/
    *.csv                       top-5 predictions per method
  plots/
    01–15.png                   comparison plots (see below)
  submission/                   ★ official challenge-submission folder
    final_hitlist.csv           the top-5 from the best model (APO graph)
    apo_score_matrix.csv        verbose per-residue scores on APO
    holo_score_matrix.csv       verbose per-residue scores on HOLO + GT
    apo_predictions_top5.csv    APO top-5 with full metadata
    holo_predictions_top5.csv   HOLO top-5 with GT-hop info
    metadata.json               model identity, hyperparameters, scores
    README.md                   human-readable overview

Usage:
  python run_full_pipeline.py [--out OUTPUT_DIR] [--quick]
"""
from __future__ import annotations
import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------------------------------------------------
# Suppress noisy warnings & set publication-style defaults
# -----------------------------------------------------------------------
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.dpi": 130,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
})

# -----------------------------------------------------------------------
# QuanAllo imports
# -----------------------------------------------------------------------
from quanallo import (
    ProteinGraph,
    AllostericPredictor,
    METHOD_REGISTRY,
    QuanAntColony,
    AdaptiveQuanAnt,
    evaluate,
    hop_distances,
    rrf_combine,
    weighted_combine,
    trimmed_mean_combine,
)
from quanallo.methods import (
    QSVD, DQAWTimeAvg, DQAWLifetime, QPageRank, HeatKernel,
    CTQW, CommuteTime, GNM, MetaLearner,
)
from quanallo.core.ensemble import softmax_with_floor
from quanallo.visualization import KIND_COLORS


# =======================================================================
# DATA LOADING
# =======================================================================
def _build_weighted(edges_df: pd.DataFrame, N: int) -> np.ndarray:
    A = np.zeros((N, N))
    for _, r in edges_df.iterrows():
        i, j = int(r["i"]), int(r["j"])
        w = float(r["weight"])
        A[i, j] = w
        A[j, i] = w
    return A


def load_kras(data_dir: Path) -> tuple[ProteinGraph, ProteinGraph]:
    """Load the preprocessed KRAS G12C APO + HOLO graphs."""

    def _load(side: str) -> ProteinGraph:
        prefix = "apo" if side == "apo" else "holo"
        nodes = pd.read_csv(data_dir / f"{prefix}_nodes.csv")
        edges = pd.read_csv(data_dir / f"{prefix}_edges.csv")
        A_bin = np.load(data_dir / f"{prefix}_adjacency.npy").astype(float)
        A_w = _build_weighted(edges, len(nodes))
        key = nodes.set_index(["chain", "resnum"])["idx"]

        def _to_idx(df: pd.DataFrame) -> np.ndarray:
            return np.asarray(
                [int(key.loc[(r.chain, int(r.resnum))])
                 for _, r in df.iterrows()
                 if (r.chain, int(r.resnum)) in key.index],
                dtype=int,
            )

        active_csv = "apo_active_site.csv" if side == "apo" else "apo_active_site.csv"
        gt_csv = "holo_ground_truth.csv" if side == "apo" else "holo_ground_truth.csv"
        return ProteinGraph(
            nodes=nodes,
            adjacency_binary=A_bin,
            adjacency_weighted=A_w,
            active_idx=_to_idx(pd.read_csv(data_dir / active_csv)),
            ground_truth_idx=_to_idx(pd.read_csv(data_dir / gt_csv)),
            name=f"Myosin_{side.upper()}",
        )

    return _load("apo"), _load("holo")


# =======================================================================
# UTILITIES — score a method on a graph, return a flat dict for the matrix
# =======================================================================
def score_to_row(
    scores: np.ndarray,
    graph: ProteinGraph,
    *,
    method: str,
    stage: str,
    side: str,
    extra: dict | None = None,
) -> dict:
    """Run evaluate() and return a flat dict suitable for the results table."""
    ev = evaluate(
        scores,
        graph.adjacency_weighted,
        graph.ground_truth_idx,
        graph.active_idx,
        graph.N,
        surface_mask=graph.surface_mask,
    )
    top_resnums = [int(graph.nodes.iloc[p]["resnum"]) for p in ev["top_pred"]]
    row = {
        "stage": stage,
        "method": method,
        "side": side,
        "weighted_top5": round(ev["weighted_top5"], 4),
        "P5_k0": round(ev["precision_at_5"][0], 3),
        "P5_k1": round(ev["precision_at_5"][1], 3),
        "P5_k2": round(ev["precision_at_5"][2], 3),
        "P5_k3": round(ev["precision_at_5"][3], 3),
        "top5_idx": ",".join(map(str, ev["top_pred"])),
        "top5_resnums": ",".join(map(str, top_resnums)),
        "hops_to_gt": ",".join(map(str, ev["hops_to_gt"])),
    }
    if extra:
        row.update(extra)
    return row


# =======================================================================
# STAGE 1 — Hyperparameter grid search per method (APO-supervised)
# =======================================================================
def stage1_grid_search(apo: ProteinGraph, holo: ProteinGraph,
                       *, quick: bool = False, log) -> tuple[dict, list, dict]:
    """
    Grid-search hyperparameters for each method using APO weighted_top5 as the
    objective. Returns (best_params, all_rows, grid_results).
    """
    log("\n[STAGE 1] Hyperparameter grid search (APO-supervised)")
    log("-" * 70)

    if quick:
        dqaw_l_grid = list(itertools.product(
            [0.4],
            [0.2, 0.4],
            [2.0],
            [1.8, 2.3],
            [8],
        ))
        dqaw_t_grid = list(itertools.product(
            [0.4],
            [2.0],
            [0.3, 0.6],
            [2.0],
            [1.0, 1.6],
        ))
        qsvd_grid = [10, 15, 20]
        qpr_grid = list(itertools.product([0.85], [2.0, 2.5, 3.0], [1.8]))
        heat_grid = [1.0, 2.0, 3.0]
        ctqw_grid = [5.0, 15.0]
        commute_grid = [None]
        gnm_grid = [8, 12]
    else:
        dqaw_l_grid = list(itertools.product(
            [0.3, 0.4, 0.5],
            [0.1, 0.2, 0.4],
            [1.5, 2.0, 2.5],
            [1.8, 2.3, 2.8],
            [6, 8, 10],
        ))
        dqaw_t_grid = list(itertools.product(
            [0.3, 0.4, 0.5],
            [1.5, 2.0, 2.5],
            [0.3, 0.6, 0.9],
            [1.5, 2.0, 2.5],
            [1.0, 1.3, 1.6],
        ))
        qsvd_grid = [8, 12, 15, 18, 22]
        qpr_grid = list(itertools.product(
            [0.80, 0.85, 0.90],
            [2.0, 2.5, 3.0],
            [1.5, 1.8, 2.1],
        ))
        heat_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
        ctqw_grid = [5.0, 10.0, 15.0, 20.0]
        commute_grid = [None]
        gnm_grid = [6, 8, 10, 12]

    grid_results: dict = {}
    best_params: dict = {}
    all_rows: list = []

    # --- DQAWLifetime ---
    log(f"  DQAWLifetime: {len(dqaw_l_grid)} configs ...")
    scores_grid = []
    for alpha, gamma, mu, sigma, n_modes in dqaw_l_grid:
        m = DQAWLifetime(alpha=alpha, gamma_absorb=gamma,
                          hop_mu=mu, hop_sigma=sigma, n_modes=n_modes)
        s = m.compute(apo)
        ev = evaluate(s, apo.adjacency_weighted, apo.ground_truth_idx,
                       apo.active_idx, apo.N, surface_mask=apo.surface_mask)
        scores_grid.append({
            "method": "dqaw_lifetime",
            "alpha": alpha, "gamma_absorb": gamma,
            "hop_mu": mu, "hop_sigma": sigma, "n_modes": n_modes,
            "apo_wt5": ev["weighted_top5"],
        })
    df = pd.DataFrame(scores_grid).sort_values("apo_wt5", ascending=False)
    grid_results["dqaw_lifetime"] = df
    best = df.iloc[0]
    best_params["dqaw_lifetime"] = dict(
        alpha=float(best["alpha"]),
        gamma_absorb=float(best["gamma_absorb"]),
        hop_mu=float(best["hop_mu"]),
        hop_sigma=float(best["hop_sigma"]),
        n_modes=int(best["n_modes"]),
    )
    log(f"    best: alpha={best['alpha']} gamma={best['gamma_absorb']} "
        f"mu={best['hop_mu']} sigma={best['hop_sigma']} "
        f"n_modes={best['n_modes']}  →  APO {best['apo_wt5']:.3f}")

    # --- DQAWTimeAvg ---
    log(f"  DQAWTimeAvg : {len(dqaw_t_grid)} configs ...")
    scores_grid = []
    for alpha, beta, gamma, mu, sigma in dqaw_t_grid:
        m = DQAWTimeAvg(alpha=alpha, beta_well=beta, gamma_absorb=gamma,
                         hop_mu=mu, hop_sigma=sigma)
        s = m.compute(apo)
        ev = evaluate(s, apo.adjacency_weighted, apo.ground_truth_idx,
                       apo.active_idx, apo.N, surface_mask=apo.surface_mask)
        scores_grid.append({
            "method": "dqaw_timeavg",
            "alpha": alpha, "beta_well": beta, "gamma_absorb": gamma,
            "hop_mu": mu, "hop_sigma": sigma,
            "apo_wt5": ev["weighted_top5"],
        })
    df = pd.DataFrame(scores_grid).sort_values("apo_wt5", ascending=False)
    grid_results["dqaw_timeavg"] = df
    best = df.iloc[0]
    best_params["dqaw_timeavg"] = dict(
        alpha=float(best["alpha"]), beta_well=float(best["beta_well"]),
        gamma_absorb=float(best["gamma_absorb"]),
        hop_mu=float(best["hop_mu"]), hop_sigma=float(best["hop_sigma"]),
    )
    log(f"    best APO {best['apo_wt5']:.3f}")

    # --- QSVD ---
    log(f"  QSVD        : {len(qsvd_grid)} configs ...")
    scores_grid = []
    for n_comp in qsvd_grid:
        m = QSVD(n_components=n_comp)
        s = m.compute(apo)
        ev = evaluate(s, apo.adjacency_weighted, apo.ground_truth_idx,
                       apo.active_idx, apo.N, surface_mask=apo.surface_mask)
        scores_grid.append({"method": "qsvd", "n_components": n_comp,
                            "apo_wt5": ev["weighted_top5"]})
    df = pd.DataFrame(scores_grid).sort_values("apo_wt5", ascending=False)
    grid_results["qsvd"] = df
    best = df.iloc[0]
    best_params["qsvd"] = dict(n_components=int(best["n_components"]))
    log(f"    best n_components={best['n_components']}  →  APO {best['apo_wt5']:.3f}")

    # --- QPageRank ---
    log(f"  QPageRank   : {len(qpr_grid)} configs ...")
    scores_grid = []
    for damping, mu, sigma in qpr_grid:
        m = QPageRank(damping=damping, hop_mu=mu, hop_sigma=sigma)
        s = m.compute(apo)
        ev = evaluate(s, apo.adjacency_weighted, apo.ground_truth_idx,
                       apo.active_idx, apo.N, surface_mask=apo.surface_mask)
        scores_grid.append({"method": "qpagerank", "damping": damping,
                            "hop_mu": mu, "hop_sigma": sigma,
                            "apo_wt5": ev["weighted_top5"]})
    df = pd.DataFrame(scores_grid).sort_values("apo_wt5", ascending=False)
    grid_results["qpagerank"] = df
    best = df.iloc[0]
    best_params["qpagerank"] = dict(
        damping=float(best["damping"]),
        hop_mu=float(best["hop_mu"]), hop_sigma=float(best["hop_sigma"]),
    )
    log(f"    best APO {best['apo_wt5']:.3f}")

    # --- HeatKernel ---
    log(f"  HeatKernel  : {len(heat_grid)} configs ...")
    scores_grid = []
    for t in heat_grid:
        m = HeatKernel(t=t)
        s = m.compute(apo)
        ev = evaluate(s, apo.adjacency_weighted, apo.ground_truth_idx,
                       apo.active_idx, apo.N, surface_mask=apo.surface_mask)
        scores_grid.append({"method": "heatkernel", "t": t,
                            "apo_wt5": ev["weighted_top5"]})
    df = pd.DataFrame(scores_grid).sort_values("apo_wt5", ascending=False)
    grid_results["heatkernel"] = df
    best = df.iloc[0]
    best_params["heatkernel"] = dict(t=float(best["t"]))
    log(f"    best t={best['t']}  →  APO {best['apo_wt5']:.3f}")

    # --- CTQW ---
    log(f"  CTQW        : {len(ctqw_grid)} configs ...")
    scores_grid = []
    for T_max in ctqw_grid:
        m = CTQW(T_max=T_max)
        s = m.compute(apo)
        ev = evaluate(s, apo.adjacency_weighted, apo.ground_truth_idx,
                       apo.active_idx, apo.N, surface_mask=apo.surface_mask)
        scores_grid.append({"method": "ctqw", "T_max": T_max,
                            "apo_wt5": ev["weighted_top5"]})
    df = pd.DataFrame(scores_grid).sort_values("apo_wt5", ascending=False)
    grid_results["ctqw"] = df
    best = df.iloc[0]
    best_params["ctqw"] = dict(T_max=float(best["T_max"]))
    log(f"    best T_max={best['T_max']}  →  APO {best['apo_wt5']:.3f}")

    # --- CommuteTime (no params) ---
    log("  CommuteTime : 1 config (no params) ...")
    m = CommuteTime()
    s = m.compute(apo)
    ev = evaluate(s, apo.adjacency_weighted, apo.ground_truth_idx,
                   apo.active_idx, apo.N, surface_mask=apo.surface_mask)
    grid_results["commute_time"] = pd.DataFrame([{
        "method": "commute_time", "apo_wt5": ev["weighted_top5"]
    }])
    best_params["commute_time"] = {}
    log(f"    APO {ev['weighted_top5']:.3f}")

    # --- GNM ---
    log(f"  GNM         : {len(gnm_grid)} configs ...")
    scores_grid = []
    for k in gnm_grid:
        m = GNM(n_slow_modes=k)
        s = m.compute(apo)
        ev = evaluate(s, apo.adjacency_weighted, apo.ground_truth_idx,
                       apo.active_idx, apo.N, surface_mask=apo.surface_mask)
        scores_grid.append({"method": "gnm", "n_slow_modes": k,
                            "apo_wt5": ev["weighted_top5"]})
    df = pd.DataFrame(scores_grid).sort_values("apo_wt5", ascending=False)
    grid_results["gnm"] = df
    best = df.iloc[0]
    best_params["gnm"] = dict(n_slow_modes=int(best["n_slow_modes"]))
    log(f"    best n_slow_modes={best['n_slow_modes']}  →  APO {best['apo_wt5']:.3f}")

    # --- MetaLearner ---
    log("  MetaLearner : trains on APO GT (logreg, 10 features) ...")
    meta = MetaLearner(C=0.5).fit(apo)
    s_apo = meta.compute(apo)
    ev = evaluate(s_apo, apo.adjacency_weighted, apo.ground_truth_idx,
                   apo.active_idx, apo.N, surface_mask=apo.surface_mask)
    grid_results["meta_learner"] = pd.DataFrame([{
        "method": "meta_learner", "C": 0.5, "apo_wt5": ev["weighted_top5"]
    }])
    best_params["meta_learner"] = dict(C=0.5, _trained=True)
    log(f"    APO {ev['weighted_top5']:.3f}")

    log("\n[STAGE 1] Grid search complete.")
    return best_params, all_rows, grid_results


# =======================================================================
# STAGE 2 — Default vs tuned single-method, on APO + HOLO
# =======================================================================
def instantiate_tuned(method: str, params: dict, apo: ProteinGraph) -> object:
    """Build a method instance from saved best params. Returns the method,
    pre-fitted if it's the meta-learner."""
    if method == "meta_learner":
        # Re-fit on APO since MetaLearner is stateful
        return MetaLearner(C=params.get("C", 0.5)).fit(apo)
    if method == "qsvd":
        return QSVD(**{k: v for k, v in params.items() if k != "_trained"})
    if method == "dqaw_timeavg":
        return DQAWTimeAvg(**{k: v for k, v in params.items() if k != "_trained"})
    if method == "dqaw_lifetime":
        return DQAWLifetime(**{k: v for k, v in params.items() if k != "_trained"})
    if method == "qpagerank":
        return QPageRank(**{k: v for k, v in params.items() if k != "_trained"})
    if method == "heatkernel":
        return HeatKernel(**{k: v for k, v in params.items() if k != "_trained"})
    if method == "ctqw":
        return CTQW(**{k: v for k, v in params.items() if k != "_trained"})
    if method == "commute_time":
        return CommuteTime()
    if method == "gnm":
        return GNM(**{k: v for k, v in params.items() if k != "_trained"})
    raise ValueError(f"unknown method {method!r}")


def stage2_default_vs_tuned(apo: ProteinGraph, holo: ProteinGraph,
                              best_params: dict, log) -> tuple[list, dict, dict]:
    """Compute default + tuned scores on APO and HOLO. Return rows + score matrices."""
    log("\n[STAGE 2] Default vs Tuned single-method (APO + HOLO)")
    log("-" * 70)
    rows = []
    apo_scores: dict[str, np.ndarray] = {}
    holo_scores: dict[str, np.ndarray] = {}

    for method in METHOD_REGISTRY:
        log(f"  {method:<16}", end="")
        # --- DEFAULT ---
        cls = METHOD_REGISTRY[method]
        if method == "meta_learner":
            inst = cls().fit(apo)
        else:
            inst = cls()
        s_a, s_h = inst.compute(apo), inst.compute(holo)
        rows.append(score_to_row(s_a, apo, method=f"{method}_default",
                                   stage="default", side="APO"))
        rows.append(score_to_row(s_h, holo, method=f"{method}_default",
                                   stage="default", side="HOLO"))
        apo_scores[f"{method}_default"] = s_a
        holo_scores[f"{method}_default"] = s_h

        # --- TUNED ---
        if best_params.get(method):
            inst_t = instantiate_tuned(method, best_params[method], apo)
            s_a_t, s_h_t = inst_t.compute(apo), inst_t.compute(holo)
            rows.append(score_to_row(s_a_t, apo, method=f"{method}_tuned",
                                       stage="tuned", side="APO"))
            rows.append(score_to_row(s_h_t, holo, method=f"{method}_tuned",
                                       stage="tuned", side="HOLO"))
            apo_scores[f"{method}_tuned"] = s_a_t
            holo_scores[f"{method}_tuned"] = s_h_t
            log(f"  APO {rows[-3]['weighted_top5']:.3f} → {rows[-1]['weighted_top5']:.3f}  "
                f"|  HOLO {rows[-2]['weighted_top5']:.3f}")
        else:
            log(f"  APO {rows[-2]['weighted_top5']:.3f}  |  HOLO {rows[-1]['weighted_top5']:.3f}")

    return rows, apo_scores, holo_scores


# =======================================================================
# STAGE 3 — Species selection (forward greedy + all subsets)
# =======================================================================
def stage3_species_selection(
    apo_scores: dict, holo_scores: dict, apo: ProteinGraph, holo: ProteinGraph,
    log,
) -> tuple[list, dict]:
    """Pick the best subset of (tuned) single methods to ensemble.

    - Forward greedy on APO (legitimate transfer-learning).
    - All-subsets on both APO and HOLO (for the oracle gap).
    """
    log("\n[STAGE 3] Species selection (forward greedy + all-subsets)")
    log("-" * 70)
    candidates = [m for m in apo_scores if m.endswith("_tuned")]
    log(f"  candidate methods: {candidates}")

    def ensemble_score(selected: list[str], side: str = "APO") -> float:
        scores = apo_scores if side == "APO" else holo_scores
        sub = {m: scores[m] for m in selected}
        fused = rrf_combine(sub)
        g = apo if side == "APO" else holo
        ev = evaluate(fused, g.adjacency_weighted, g.ground_truth_idx,
                       g.active_idx, g.N, surface_mask=g.surface_mask)
        return ev["weighted_top5"]

    # --- Forward greedy on APO ---
    selected: list[str] = []
    remaining = list(candidates)
    forward_log = []
    best_score = -np.inf
    log("\n  Forward greedy (APO-supervised, applied to HOLO):")
    while remaining:
        scores_at_step = {}
        for c in remaining:
            trial = selected + [c]
            scores_at_step[c] = ensemble_score(trial, side="APO")
        best_c = max(scores_at_step, key=scores_at_step.get)
        new_score = scores_at_step[best_c]
        if new_score < best_score - 0.02:
            break
        selected.append(best_c)
        remaining.remove(best_c)
        holo_at_step = ensemble_score(selected, side="HOLO")
        forward_log.append({
            "step": len(selected),
            "added": best_c,
            "selected": ",".join(selected),
            "apo_wt5": round(new_score, 3),
            "holo_wt5": round(holo_at_step, 3),
        })
        log(f"    +{best_c:<24}  APO {new_score:.3f}   HOLO {holo_at_step:.3f}")
        best_score = max(best_score, new_score)
        if len(selected) >= 5:
            break
    forward_best = selected

    # --- All subsets up to size 5 ---
    log("\n  All-subsets exhaustive (computing oracle bound):")
    if len(candidates) > 7:
        candidates_small = candidates[:7]
    else:
        candidates_small = candidates
    all_rows = []
    for size in range(1, min(6, len(candidates_small) + 1)):
        for subset in itertools.combinations(candidates_small, size):
            subset = list(subset)
            apo_s = ensemble_score(subset, side="APO")
            holo_s = ensemble_score(subset, side="HOLO")
            all_rows.append({
                "size": size,
                "subset": ",".join(subset),
                "apo_wt5": apo_s,
                "holo_wt5": holo_s,
            })
    df_all = pd.DataFrame(all_rows)
    # best per size on APO and HOLO
    log("    best by size:")
    for size in sorted(df_all["size"].unique()):
        d = df_all[df_all["size"] == size]
        best_apo = d.loc[d["apo_wt5"].idxmax()]
        best_holo = d.loc[d["holo_wt5"].idxmax()]
        log(f"      size {size}: APO-best={best_apo['apo_wt5']:.3f}  "
            f"HOLO-best (oracle)={best_holo['holo_wt5']:.3f}")
    summary = {
        "forward_log": forward_log,
        "all_subsets": df_all,
        "forward_best": forward_best,
    }
    return forward_log, summary


# =======================================================================
# STAGE 4 — Ensemble methods comparison
# =======================================================================
def stage4_ensemble_methods(
    apo_scores: dict, holo_scores: dict, apo: ProteinGraph, holo: ProteinGraph,
    log,
) -> list:
    """Compare RRF, weighted-mean, trimmed-mean on the tuned method set."""
    log("\n[STAGE 4] Ensemble methods comparison")
    log("-" * 70)
    tuned = {m: holo_scores[m] for m in holo_scores if m.endswith("_tuned")
              and not m.startswith("meta_learner")}
    tuned_apo = {m: apo_scores[m] for m in apo_scores if m.endswith("_tuned")
                  and not m.startswith("meta_learner")}

    rows = []

    # APO-trained weights for the weighted ensemble
    apo_ws = {m: max(evaluate(apo_scores[m], apo.adjacency_weighted,
                                 apo.ground_truth_idx, apo.active_idx, apo.N,
                                 surface_mask=apo.surface_mask)["weighted_top5"], 0.1)
                for m in tuned}
    apo_ws = softmax_with_floor(apo_ws, temperature=1.0, floor=0.05)
    log(f"  APO-learned ensemble weights (softmax T=1.0): "
        f"{ {k: round(v, 3) for k, v in apo_ws.items()} }")

    # RRF (unweighted)
    fused_h = rrf_combine(tuned)
    fused_a = rrf_combine(tuned_apo)
    rows.append(score_to_row(fused_a, apo, method="ensemble_RRF",
                               stage="ensemble", side="APO"))
    rows.append(score_to_row(fused_h, holo, method="ensemble_RRF",
                               stage="ensemble", side="HOLO"))
    log(f"  RRF        APO {rows[-2]['weighted_top5']:.3f}  HOLO {rows[-1]['weighted_top5']:.3f}")

    # RRF weighted by APO score
    fused_h_w = rrf_combine(tuned, weights=apo_ws)
    fused_a_w = rrf_combine(tuned_apo, weights=apo_ws)
    rows.append(score_to_row(fused_a_w, apo, method="ensemble_RRF_APOweighted",
                               stage="ensemble", side="APO"))
    rows.append(score_to_row(fused_h_w, holo, method="ensemble_RRF_APOweighted",
                               stage="ensemble", side="HOLO"))
    log(f"  RRF-APOw   APO {rows[-2]['weighted_top5']:.3f}  HOLO {rows[-1]['weighted_top5']:.3f}")

    # Weighted mean (with APO weights)
    fused_h_m = weighted_combine(tuned, weights=apo_ws)
    fused_a_m = weighted_combine(tuned_apo, weights=apo_ws)
    rows.append(score_to_row(fused_a_m, apo, method="ensemble_weighted_mean",
                               stage="ensemble", side="APO"))
    rows.append(score_to_row(fused_h_m, holo, method="ensemble_weighted_mean",
                               stage="ensemble", side="HOLO"))
    log(f"  Weighted   APO {rows[-2]['weighted_top5']:.3f}  HOLO {rows[-1]['weighted_top5']:.3f}")

    # Trimmed mean
    fused_h_t = trimmed_mean_combine(tuned, trim_lo=1, trim_hi=0)
    fused_a_t = trimmed_mean_combine(tuned_apo, trim_lo=1, trim_hi=0)
    rows.append(score_to_row(fused_a_t, apo, method="ensemble_trimmed_mean",
                               stage="ensemble", side="APO"))
    rows.append(score_to_row(fused_h_t, holo, method="ensemble_trimmed_mean",
                               stage="ensemble", side="HOLO"))
    log(f"  Trimmed    APO {rows[-2]['weighted_top5']:.3f}  HOLO {rows[-1]['weighted_top5']:.3f}")

    return rows


# =======================================================================
# STAGE 5 — QuanAnt mode comparison
# =======================================================================
def stage5_quanant_modes(apo: ProteinGraph, holo: ProteinGraph,
                          *, quick: bool, log) -> tuple[list, dict]:
    """Run several QuanAnt configurations on HOLO and compare."""
    log("\n[STAGE 5] QuanAnt mode comparison")
    log("-" * 70)
    rows = []
    pheromones: dict[str, np.ndarray] = {}

    ants = 3 if quick else 5
    iters = 3 if quick else 5
    species5 = ["qsvd", "dqaw_timeavg", "dqaw_lifetime", "qpagerank", "heatkernel"]

    # --- Single-species QuanAnt (the v4 baseline approach) ---
    log(f"  Single-species DQAW-Lifetime QuanAnt ({ants} ants × {iters} iter)...")
    t0 = time.time()
    colony = QuanAntColony(species=["dqaw_lifetime"], ants_per_species=ants * 4,
                            n_iter=iters, aggregation="shared_pheromone",
                            parallel=True, verbose=False)
    r = colony.run(holo)
    rows.append(score_to_row(r.final_score, holo,
                               method="quanant_single_dqaw_lifetime",
                               stage="quanant", side="HOLO",
                               extra={"runtime_s": round(time.time() - t0, 1)}))
    pheromones["quanant_single_dqaw_lifetime"] = r.final_score
    log(f"    HOLO {rows[-1]['weighted_top5']:.3f}  ({rows[-1]['runtime_s']}s)")

    # --- Heterogeneous shared pheromone ---
    log(f"  Heterogeneous shared-pheromone ({len(species5)} species × {ants} ants × {iters} iter)...")
    t0 = time.time()
    colony = QuanAntColony(species=species5, ants_per_species=ants,
                            n_iter=iters, aggregation="shared_pheromone",
                            parallel=True, verbose=False)
    r = colony.run(holo)
    rows.append(score_to_row(r.final_score, holo,
                               method="quanant_hetero_shared",
                               stage="quanant", side="HOLO",
                               extra={"runtime_s": round(time.time() - t0, 1)}))
    pheromones["quanant_hetero_shared"] = r.final_score
    log(f"    HOLO {rows[-1]['weighted_top5']:.3f}  ({rows[-1]['runtime_s']}s)")

    # --- Heterogeneous multi-pheromone (each species own field) ---
    log(f"  Heterogeneous multi-pheromone (independent fields)...")
    t0 = time.time()
    colony = QuanAntColony(species=species5, ants_per_species=ants,
                            n_iter=iters, aggregation="multi_pheromone",
                            parallel=True, verbose=False)
    r = colony.run(holo)
    rows.append(score_to_row(r.final_score, holo,
                               method="quanant_hetero_multi",
                               stage="quanant", side="HOLO",
                               extra={"runtime_s": round(time.time() - t0, 1)}))
    pheromones["quanant_hetero_multi"] = r.final_score
    log(f"    HOLO {rows[-1]['weighted_top5']:.3f}  ({rows[-1]['runtime_s']}s)")

    return rows, pheromones


# =======================================================================
# STAGE 6 — Adaptive QuanAnt (APO→HOLO transfer)
# =======================================================================
def stage6_adaptive_quanant(apo: ProteinGraph, holo: ProteinGraph,
                             *, quick: bool, log) -> tuple[list, dict, list]:
    """APO-trained species weights + online updates."""
    log("\n[STAGE 6] Adaptive QuanAnt (APO→HOLO transfer)")
    log("-" * 70)
    ants = 3 if quick else 5
    iters = 3 if quick else 5
    rows = []
    pheromones: dict[str, np.ndarray] = {}

    species5 = ["qsvd", "dqaw_timeavg", "dqaw_lifetime", "qpagerank", "heatkernel"]
    adaptive = AdaptiveQuanAnt(
        species=species5, ants_per_species=ants, n_iter=iters,
        softmax_temperature=1.0, weight_floor=0.10, adaptive_momentum=0.40,
        parallel=True, verbose=False,
    )
    t0 = time.time()
    adaptive.fit(apo)
    log("  APO-learned species weights:")
    for sp, w in adaptive.apo_weights.items():
        log(f"    {sp:<16} weight={w:.3f}  "
            f"(raw={adaptive._apo_species_scores[sp]:.3f})")

    result = adaptive.predict(holo)
    rt = time.time() - t0
    rows.append(score_to_row(result.final_score, holo,
                               method="adaptive_quanant",
                               stage="adaptive", side="HOLO",
                               extra={"runtime_s": round(rt, 1)}))
    pheromones["adaptive_quanant"] = result.final_score
    log(f"  Adaptive HOLO {rows[-1]['weighted_top5']:.3f}  ({rt:.1f}s)")
    return rows, pheromones, result.species_weights_history


# =======================================================================
# STAGE 7 — Build submission folder (APO + HOLO matrices + best hitlist)
# =======================================================================
def stage7_create_submission(
    apo: ProteinGraph,
    holo: ProteinGraph,
    all_rows: list,
    apo_scores: dict,
    holo_scores: dict,
    pheromones: dict,
    best_params: dict,
    out_dir: Path,
    log,
) -> Path:
    """
    Build the official submission folder.

    Selects the best-performing model (highest HOLO weighted_top5) that has
    per-residue scores on **both** APO and HOLO, then writes:

        submission/
            apo_score_matrix.csv      — full per-residue scores on APO
            holo_score_matrix.csv     — full per-residue scores on HOLO
            apo_predictions_top5.csv  — APO top-5 (verbose)
            holo_predictions_top5.csv — HOLO top-5 (verbose, with GT check)
            final_hitlist.csv         — THE submission: clean top-5 from APO
            metadata.json             — model details, hyperparams, scores
            README.md                 — human-readable overview

    A "best model" is chosen with this rule:
      1. Sort all HOLO predictions by weighted_top5 (descending).
      2. Walk down until we find one that has per-residue scores on BOTH
         APO and HOLO (this excludes QuanAnt-only entries, which only have
         a HOLO pheromone vector).
      3. Tie-break by the corresponding APO score (higher wins).

    Returns the path to the submission folder.
    """
    log("\n" + "=" * 70)
    log("[STAGE 7] Building submission folder")
    log("=" * 70)

    sub_dir = out_dir / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Identify the best model with full APO+HOLO scores ----
    df_all = pd.DataFrame(all_rows)
    df_holo_sorted = (
        df_all[df_all["side"] == "HOLO"]
        .sort_values("weighted_top5", ascending=False)
    )
    best_method, best_holo_row = None, None
    for _, row in df_holo_sorted.iterrows():
        if row["method"] in apo_scores and row["method"] in holo_scores:
            best_method = row["method"]
            best_holo_row = row
            break

    # If no method has both, fall back to top HOLO regardless
    fell_back_to_pheromone = False
    if best_method is None:
        best_holo_row = df_holo_sorted.iloc[0]
        best_method = best_holo_row["method"]
        fell_back_to_pheromone = True
        log(f"  ⚠ Top model {best_method!r} is QuanAnt-only "
            "(no APO scores). Submission uses HOLO pheromone only.")

    # Find matching APO row (may not exist for QuanAnt entries)
    apo_row_match = df_all[
        (df_all["side"] == "APO") & (df_all["method"] == best_method)
    ]
    apo_row = apo_row_match.iloc[0] if len(apo_row_match) > 0 else None

    log(f"\n  Selected model: {best_method}")
    log(f"  HOLO weighted_top5: {best_holo_row['weighted_top5']:.3f}")
    if apo_row is not None:
        log(f"  APO  weighted_top5: {apo_row['weighted_top5']:.3f}")

    # ---- 2) Get per-residue score vectors ----
    s_apo = apo_scores.get(best_method)
    s_holo = holo_scores.get(best_method, pheromones.get(best_method))
    if s_apo is None:
        # Method only has HOLO pheromone — duplicate it as APO fallback
        # (only happens for QuanAnt-only entries; we warn above)
        s_apo = pheromones.get(best_method)

    # ---- 3) Build full per-residue matrices ----
    def build_matrix(graph: ProteinGraph, scores: np.ndarray) -> pd.DataFrame:
        """Build a verbose per-residue score table for one graph."""
        n = graph.N
        df = graph.nodes[
            ["idx", "chain", "resnum", "resname", "x", "y", "z"]
        ].copy()
        df["is_active_site"] = df["idx"].isin(graph.active_idx.tolist())
        df["is_surface"] = graph.surface_mask
        df["raw_score"] = np.asarray(scores, dtype=float)

        # Normalize to [0, 1]
        smin, smax = float(df["raw_score"].min()), float(df["raw_score"].max())
        df["normalized_score"] = (df["raw_score"] - smin) / (smax - smin + 1e-12)

        # Rank only among eligible (surface, non-active) residues
        eligible = (df["is_surface"] & ~df["is_active_site"]).values
        masked = df["raw_score"].values.copy()
        masked[~eligible] = -np.inf
        order = np.argsort(masked)[::-1]
        rank = np.full(n, -1, dtype=int)
        rk = 1
        for i in order:
            if eligible[i] and np.isfinite(masked[i]):
                rank[i] = rk
                rk += 1
        df["rank"] = rank
        df["in_top5"] = (df["rank"] >= 1) & (df["rank"] <= 5)

        # Add GT info if available (HOLO and APO both have it in our setup)
        if graph.ground_truth_idx is not None:
            gt_set = set(graph.ground_truth_idx.tolist())
            df["is_ground_truth"] = df["idx"].isin(gt_set)
            h2g = hop_distances(graph.adjacency_weighted,
                                graph.ground_truth_idx, n)
            df["hop_to_gt"] = h2g.astype(int)
        return df

    apo_matrix = build_matrix(apo, s_apo)
    holo_matrix = build_matrix(holo, s_holo)

    apo_top5 = (
        apo_matrix[apo_matrix["in_top5"]].sort_values("rank").reset_index(drop=True)
    )
    holo_top5 = (
        holo_matrix[holo_matrix["in_top5"]].sort_values("rank").reset_index(drop=True)
    )

    # ---- 4) Write CSV files ----
    apo_matrix.to_csv(sub_dir / "apo_score_matrix.csv", index=False)
    holo_matrix.to_csv(sub_dir / "holo_score_matrix.csv", index=False)
    apo_top5.to_csv(sub_dir / "apo_predictions_top5.csv", index=False)
    holo_top5.to_csv(sub_dir / "holo_predictions_top5.csv", index=False)

    # The FINAL hitlist — clean, minimal columns, APO predictions
    # (the challenge takes APO as input, so this is the legitimate submission)
    final_cols = [
        "rank", "chain", "resnum", "resname",
        "raw_score", "normalized_score", "x", "y", "z",
    ]
    final_hitlist = apo_top5[final_cols].copy()
    final_hitlist.to_csv(sub_dir / "final_hitlist.csv", index=False)

    log(f"\n  Files written to {sub_dir}/:")
    log(f"    apo_score_matrix.csv      ({len(apo_matrix)} rows × {len(apo_matrix.columns)} cols)")
    log(f"    holo_score_matrix.csv     ({len(holo_matrix)} rows × {len(holo_matrix.columns)} cols)")
    log(f"    apo_predictions_top5.csv  ({len(apo_top5)} rows)")
    log(f"    holo_predictions_top5.csv ({len(holo_top5)} rows)")
    log(f"    final_hitlist.csv         ({len(final_hitlist)} rows — THE submission)")

    # ---- 5) Metadata ----
    method_clean = best_method.replace("_tuned", "").replace("_default", "")
    method_class_map = {
        "qsvd": "QSVD", "dqaw_timeavg": "DQAWTimeAvg",
        "dqaw_lifetime": "DQAWLifetime", "qpagerank": "QPageRank",
        "heatkernel": "HeatKernel", "ctqw": "CTQW",
        "commute_time": "CommuteTime", "gnm": "GNM",
        "meta_learner": "MetaLearner",
    }
    is_tuned = best_method.endswith("_tuned")
    hyperparams = best_params.get(method_clean, {}) if is_tuned else "(library defaults)"

    metadata = {
        "submission_for": "Cleveland Clinic Quantum + AI Challenge 2026",
        "target": "KRAS G12C",
        "apo_pdb": "4OBE",
        "holo_pdb": "6OIM",
        "drug": "Sotorasib (PDB ligand: MOV)",
        "model": {
            "name": best_method,
            "class": method_class_map.get(method_clean, method_clean),
            "tuned": is_tuned,
            "hyperparameters": hyperparams,
        },
        "performance": {
            "APO_weighted_top5": (float(apo_row["weighted_top5"])
                                   if apo_row is not None else None),
            "HOLO_weighted_top5": float(best_holo_row["weighted_top5"]),
            "HOLO_precision_at_5_k0": float(best_holo_row["P5_k0"]),
            "HOLO_precision_at_5_k1": float(best_holo_row["P5_k1"]),
            "HOLO_precision_at_5_k2": float(best_holo_row["P5_k2"]),
            "HOLO_precision_at_5_k3": float(best_holo_row["P5_k3"]),
        },
        "predictions": {
            "APO_top5": [
                {"rank": int(r["rank"]), "chain": r["chain"],
                 "resnum": int(r["resnum"]), "resname": r["resname"],
                 "raw_score": float(r["raw_score"])}
                for _, r in apo_top5.iterrows()
            ],
            "HOLO_top5": [
                {"rank": int(r["rank"]), "chain": r["chain"],
                 "resnum": int(r["resnum"]), "resname": r["resname"],
                 "raw_score": float(r["raw_score"]),
                 "is_ground_truth": bool(r.get("is_ground_truth", False)),
                 "hop_to_gt": int(r.get("hop_to_gt", -1))}
                for _, r in holo_top5.iterrows()
            ],
            "HOLO_GT_exact_hits": (int(holo_top5["is_ground_truth"].sum())
                                    if "is_ground_truth" in holo_top5.columns
                                    else None),
        },
        "baseline_comparison_HOLO": {
            "v4_baseline (tuned DQAW-Lifetime, manual)": 3.000,
            "v6_consensus (heterogeneous, uniform weights)": 2.062,
            "v7_adaptive (online Jaccard updates)": 2.062,
            "this_submission": float(best_holo_row["weighted_top5"]),
        },
    }
    with open(sub_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    log(f"    metadata.json             ({len(json.dumps(metadata))} bytes)")

    # ---- 6) README ----
    readme_lines = [
        "# QuanAllo — Submission folder",
        "",
        "Allosteric pocket predictions for the **Cleveland Clinic Quantum + AI",
        "Challenge 2026**, KRAS G12C target.",
        "",
        f"- **APO structure**: 4OBE  ({apo.N} residues, {len(apo.active_idx)} active-site residues)",
        f"- **HOLO structure**: 6OIM  ({holo.N} residues, GT drug: Sotorasib/MOV)",
        f"- **Best model**: `{best_method}`",
        f"- **HOLO weighted top-5**: **{best_holo_row['weighted_top5']:.3f} / 5.0**"
        f"  (P@5 within 0/1/2 hops = {best_holo_row['P5_k0']:.0%} / "
        f"{best_holo_row['P5_k1']:.0%} / {best_holo_row['P5_k2']:.0%})",
        "",
        "## Folder contents",
        "",
        "| File | Description |",
        "|---|---|",
        "| `final_hitlist.csv` | **THE submission** — top-5 allosteric residues from the APO graph |",
        "| `apo_score_matrix.csv` | Per-residue scores on APO (input structure) |",
        "| `holo_score_matrix.csv` | Per-residue scores on HOLO + GT validation |",
        "| `apo_predictions_top5.csv` | APO top-5 with full metadata |",
        "| `holo_predictions_top5.csv` | HOLO top-5 with GT-hop information |",
        "| `metadata.json` | Model identity, hyperparameters, scores |",
        "",
        "## Model details",
        "",
        f"**`{best_method}`** — class `{method_class_map.get(method_clean, method_clean)}`",
        "",
        "### Hyperparameters",
        "",
        "```json",
        json.dumps(hyperparams, indent=2),
        "```",
        "",
        "Hyperparameters discovered by grid search using the APO active-site",
        "ground truth as the supervisory signal (Stage 1 of the benchmark).",
        "",
        "## Performance vs prior baselines (HOLO weighted top-5)",
        "",
        "| Iteration | HOLO score | Notes |",
        "|---|---|---|",
        "| v4 baseline | 3.000 | tuned DQAW-Lifetime, manual hyperparameters |",
        "| v6 consensus | 2.062 | heterogeneous QuanAnt, uniform weights |",
        "| v7 adaptive | 2.062 | online Jaccard updates |",
        f"| **this submission** | **{best_holo_row['weighted_top5']:.3f}** | **{best_method}** |",
        "",
        "## Final hitlist (top-5 from APO)",
        "",
        "| Rank | Chain | Resnum | Resname | Raw score | Normalized score |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in apo_top5.iterrows():
        readme_lines.append(
            f"| {int(r['rank'])} | {r['chain']} | {int(r['resnum'])} | "
            f"{r['resname']} | {r['raw_score']:.4f} | {r['normalized_score']:.4f} |"
        )
    readme_lines += [
        "",
        "## HOLO predictions (validation against drug-binding ground truth)",
        "",
        "| Rank | Chain | Resnum | Resname | Raw score | Is GT? | Hop to GT |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in holo_top5.iterrows():
        is_gt = "**yes**" if bool(r.get("is_ground_truth", False)) else "no"
        hop = int(r.get("hop_to_gt", -1))
        readme_lines.append(
            f"| {int(r['rank'])} | {r['chain']} | {int(r['resnum'])} | "
            f"{r['resname']} | {r['raw_score']:.4f} | {is_gt} | {hop} |"
        )
    readme_lines += [
        "",
        "## Score matrix columns",
        "",
        "Each row of `*_score_matrix.csv` represents one Cα atom of the protein.",
        "",
        "- `idx` — 0-based residue index inside the graph",
        "- `chain`, `resnum`, `resname` — PDB identifiers",
        "- `x`, `y`, `z` — Cα coordinates (Å)",
        "- `is_active_site` — input active site (masked from prediction)",
        "- `is_surface` — SASA ≥ 20 Å² (eligible for prediction)",
        "- `raw_score` — model output (higher = more allosteric)",
        "- `normalized_score` — min-max normalized to [0, 1]",
        "- `rank` — position among eligible residues (1 = best), -1 = ineligible",
        "- `in_top5` — boolean: this residue is in the predicted top-5",
        "- `is_ground_truth` *(HOLO only)* — true allosteric pocket residue",
        "- `hop_to_gt` — shortest graph distance to any GT residue (HOLO and APO both)",
    ]
    with open(sub_dir / "README.md", "w") as fh:
        fh.write("\n".join(readme_lines) + "\n")
    log(f"    README.md")

    # ---- 7) Print the final hitlist to console ----
    log("\n  ★ FINAL HITLIST (top-5 from APO graph)")
    log("  " + "-" * 60)
    for _, r in apo_top5.iterrows():
        marker = ""
        if "is_ground_truth" in apo_top5.columns and r.get("is_ground_truth"):
            marker = "  ← GT match!"
        elif "hop_to_gt" in apo_top5.columns:
            h = int(r["hop_to_gt"])
            if 1 <= h <= 2:
                marker = f"  ← {h} hop(s) to GT"
        log(f"  {int(r['rank'])}. {r['chain']}{int(r['resnum']):<3} {r['resname']}  "
            f"score={r['raw_score']:.4f}  norm={r['normalized_score']:.4f}{marker}")

    return sub_dir


# =======================================================================
# PLOTTING
# =======================================================================
def _save(fig, path: Path, label: str, log):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log(f"    plot saved: {label} → {path.name}")


def plot_grid_search(grid_results: dict, out_dir: Path, log):
    """Plot 01 — heatmaps of APO score across hyperparameter grid per method."""
    log("\n[PLOTS] grid-search heatmaps")
    methods_with_grid = [m for m, df in grid_results.items()
                          if len(df) > 1 and m != "meta_learner"]
    n = len(methods_with_grid)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 9))
    axes = axes.flatten()
    for i, m in enumerate(methods_with_grid):
        ax = axes[i]
        df = grid_results[m].sort_values("apo_wt5")
        ax.barh(range(len(df)), df["apo_wt5"].values,
                 color=KIND_COLORS.get("quantum", "#888"))
        ax.set_yticks(range(len(df)))
        # Build a compact label for each row
        param_cols = [c for c in df.columns if c not in ("method", "apo_wt5")]
        labels = []
        for _, r in df.iterrows():
            labels.append(",".join(f"{c[:3]}={r[c]}" for c in param_cols))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("APO weighted_top5")
        ax.set_title(f"{m}  ({len(df)} configs)", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        # Highlight best
        ax.barh(len(df) - 1, df["apo_wt5"].iloc[-1], color="#e07b39",
                 edgecolor="black", linewidth=1.5)
    for j in range(len(methods_with_grid), len(axes)):
        axes[j].axis("off")
    _save(fig, out_dir / "01_grid_search_per_method.png",
          "grid search heatmaps", log)


def plot_default_vs_tuned(rows_default_tuned: list, out_dir: Path, log):
    """Plot 02 — improvement from tuning (APO + HOLO)."""
    log("[PLOTS] default-vs-tuned bars")
    df = pd.DataFrame(rows_default_tuned)
    methods = sorted(set(m.split("_default")[0].split("_tuned")[0]
                         for m in df["method"]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, side in zip(axes, ["APO", "HOLO"]):
        x = np.arange(len(methods))
        defaults, tuneds = [], []
        for m in methods:
            d = df[(df["method"].str.startswith(m)) & (df["side"] == side)]
            d_def = d[d["method"].str.endswith("_default")]
            d_tun = d[d["method"].str.endswith("_tuned")]
            defaults.append(float(d_def["weighted_top5"].iloc[0])
                              if len(d_def) else 0)
            tuneds.append(float(d_tun["weighted_top5"].iloc[0])
                            if len(d_tun) else 0)
        ax.bar(x - 0.2, defaults, 0.4, color="#5e6c75",
                edgecolor="black", linewidth=0.4, label="Default")
        ax.bar(x + 0.2, tuneds, 0.4, color="#c1272d",
                edgecolor="black", linewidth=0.4, label="Tuned")
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Weighted top-5")
        ax.set_title(f"{side}: default vs APO-tuned")
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
        # Improvement arrows
        for i, (d, t) in enumerate(zip(defaults, tuneds)):
            if t > d + 0.05:
                ax.annotate("", xy=(i + 0.2, t + 0.05), xytext=(i - 0.2, d + 0.05),
                              arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))
    _save(fig, out_dir / "02_default_vs_tuned.png",
          "default vs tuned", log)


def plot_apo_holo_scatter(rows: list, out_dir: Path, log):
    """Plot 03 — APO vs HOLO scatter (transferability)."""
    log("[PLOTS] APO-HOLO transferability scatter")
    df = pd.DataFrame(rows)
    methods = sorted(df["method"].unique())
    fig, ax = plt.subplots(figsize=(8, 7))
    for m in methods:
        rec = df[df["method"] == m]
        if {"APO", "HOLO"}.issubset(set(rec["side"])):
            xa = rec[rec["side"] == "APO"]["weighted_top5"].iloc[0]
            xh = rec[rec["side"] == "HOLO"]["weighted_top5"].iloc[0]
            kind = "tuned" if "_tuned" in m else "default"
            color = "#c1272d" if kind == "tuned" else "#5e6c75"
            ax.scatter(xa, xh, color=color, s=85,
                        edgecolor="black", linewidths=0.6, zorder=3)
            ax.annotate(m.replace("_tuned", "*").replace("_default", ""),
                          (xa, xh), fontsize=7,
                          xytext=(5, 3), textcoords="offset points")
    lo, hi = 0, max(df["weighted_top5"].max() + 0.5, 3.5)
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="perfect transfer")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("APO weighted_top5")
    ax.set_ylabel("HOLO weighted_top5")
    ax.set_title("APO → HOLO transferability\n(* = tuned;  above-diagonal = HOLO > APO)")
    ax.grid(alpha=0.3)
    from matplotlib.patches import Patch as P
    ax.legend(handles=[
        P(facecolor="#5e6c75", label="default"),
        P(facecolor="#c1272d", label="APO-tuned"),
    ], loc="upper left")
    _save(fig, out_dir / "03_apo_holo_transfer_scatter.png",
          "APO-HOLO scatter", log)


def plot_species_selection(forward_log: list, summary: dict,
                              out_dir: Path, log):
    """Plot 04 — forward selection + all-subsets bound."""
    log("[PLOTS] species selection trajectory")
    if not forward_log:
        return
    df = pd.DataFrame(forward_log)
    df_all = summary["all_subsets"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(df["step"], df["apo_wt5"], "o-",
             color="#1f4e79", linewidth=2.5, markersize=8,
             label="APO (training)")
    ax.plot(df["step"], df["holo_wt5"], "o-",
             color="#c1272d", linewidth=2.5, markersize=8,
             label="HOLO (deployment)")
    for _, r in df.iterrows():
        ax.annotate(r["added"].replace("_tuned", ""),
                     (r["step"], max(r["apo_wt5"], r["holo_wt5"])),
                     fontsize=7, ha="center",
                     xytext=(0, 8), textcoords="offset points")
    ax.set_xlabel("Step (number of species in ensemble)")
    ax.set_ylabel("Weighted top-5")
    ax.set_title("Forward-greedy species selection (APO-supervised)")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    sizes = sorted(df_all["size"].unique())
    best_apo = [df_all[df_all["size"] == s]["apo_wt5"].max() for s in sizes]
    best_holo = [df_all[df_all["size"] == s]["holo_wt5"].max() for s in sizes]
    median_apo = [df_all[df_all["size"] == s]["apo_wt5"].median() for s in sizes]
    median_holo = [df_all[df_all["size"] == s]["holo_wt5"].median() for s in sizes]
    ax.fill_between(sizes, median_apo, best_apo, color="#1f4e79", alpha=0.15)
    ax.plot(sizes, best_apo, "o-", color="#1f4e79", linewidth=2.5, label="APO best")
    ax.plot(sizes, median_apo, ":", color="#1f4e79", alpha=0.7, label="APO median")
    ax.fill_between(sizes, median_holo, best_holo, color="#c1272d", alpha=0.15)
    ax.plot(sizes, best_holo, "o-", color="#c1272d", linewidth=2.5, label="HOLO best")
    ax.plot(sizes, median_holo, ":", color="#c1272d", alpha=0.7, label="HOLO median")
    ax.set_xlabel("Subset size")
    ax.set_ylabel("Weighted top-5")
    ax.set_title(f"All-subsets exhaustive ({len(df_all)} subsets)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")
    _save(fig, out_dir / "04_species_selection.png",
          "species selection", log)


def plot_ensemble_comparison(rows_all: list, out_dir: Path, log):
    """Plot 05 — final summary bar of all approaches on HOLO."""
    log("[PLOTS] ensemble + final HOLO comparison")
    df = pd.DataFrame(rows_all)
    df_holo = df[df["side"] == "HOLO"].copy()
    df_holo = df_holo.sort_values("weighted_top5", ascending=True)
    if len(df_holo) > 30:
        df_holo = df_holo.tail(30)

    def _kind(name: str) -> str:
        if name.startswith("adaptive") or name.startswith("quanant"):
            return "quanant"
        if name.startswith("ensemble"):
            return "ensemble"
        if "meta_learner" in name:
            return "hybrid"
        if any(x in name for x in ("dqaw", "ctqw")):
            return "quantum"
        if any(x in name for x in ("qsvd", "qpagerank", "heatkernel")):
            return "quantum_inspired"
        if any(x in name for x in ("commute", "gnm")):
            return "classical"
        return "quantum"

    colors = [KIND_COLORS.get(_kind(m), "#888") for m in df_holo["method"]]
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(df_holo) + 2))
    y = np.arange(len(df_holo))
    ax.barh(y, df_holo["weighted_top5"], color=colors,
            edgecolor="black", linewidth=0.4)
    for i, v in enumerate(df_holo["weighted_top5"]):
        ax.text(v + 0.03, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(df_holo["method"], fontsize=8)
    ax.set_xlabel("HOLO weighted_top5")
    ax.set_title("Final HOLO comparison (every method)")
    ax.grid(axis="x", alpha=0.25)
    ax.axvline(3.0, ls="--", color="darkgreen", alpha=0.6)
    ax.text(3.0, 0, " v4 baseline (3.0)", color="darkgreen", fontsize=8, va="bottom")
    legend = [Patch(facecolor=c, label=k) for k, c in KIND_COLORS.items()]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    _save(fig, out_dir / "05_final_holo_comparison.png",
          "final HOLO comparison", log)


def plot_precision_curves(holo_scores: dict, holo: ProteinGraph,
                            out_dir: Path, log):
    """Plot 06 — precision@5 curves for top methods."""
    log("[PLOTS] precision@5 curves")
    candidates = list(holo_scores.keys())
    scored = []
    for m in candidates:
        ev = evaluate(holo_scores[m], holo.adjacency_weighted,
                       holo.ground_truth_idx, holo.active_idx, holo.N,
                       surface_mask=holo.surface_mask)
        scored.append((m, ev["weighted_top5"], ev["precision_at_5"]))
    scored.sort(key=lambda x: -x[1])
    top = scored[:8]
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["#c1272d", "#e07b39", "#1f4e79", "#2a9d8f", "#7b3294",
                "#8b1a1a", "#5e6c75", "#f15a29"]
    for i, (m, _, p) in enumerate(top):
        ks = sorted(p.keys()); ys = [p[k] for k in ks]
        ax.plot(ks, ys, marker="o", color=palette[i],
                 linewidth=2, label=f"{m.replace('_tuned', '*')} ({scored[i][1]:.2f})")
    ax.set_xlabel("k-hop tolerance to ground truth")
    ax.set_ylabel("Precision@5")
    ax.set_title("Precision@5 vs k-hop tolerance (top-8 by HOLO wt5)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out_dir / "06_precision_at_k_curves.png",
          "precision@5 curves", log)


def plot_top5_jaccard(holo_scores: dict, holo: ProteinGraph,
                        out_dir: Path, log):
    """Plot 07 — pairwise Jaccard top-5 between methods."""
    log("[PLOTS] top-5 Jaccard heatmap")
    methods = [m for m in holo_scores if m.endswith("_tuned")]
    if len(methods) < 2:
        return
    tops = {}
    for m in methods:
        ev = evaluate(holo_scores[m], holo.adjacency_weighted,
                       holo.ground_truth_idx, holo.active_idx, holo.N,
                       surface_mask=holo.surface_mask)
        tops[m] = set(ev["top_pred"])
    n = len(methods)
    M = np.zeros((n, n))
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            ai, bj = tops[a], tops[b]
            M[i, j] = len(ai & bj) / max(1, len(ai | bj))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short = [m.replace("_tuned", "") for m in methods]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                     color="white" if M[i, j] < 0.5 else "black", fontsize=8)
    ax.set_title("Top-5 Jaccard agreement between methods (HOLO)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Jaccard")
    _save(fig, out_dir / "07_top5_jaccard_heatmap.png",
          "top-5 Jaccard heatmap", log)


def plot_3d_best(rows_all: list, holo_scores: dict, pheromones: dict,
                   holo: ProteinGraph, out_dir: Path, log):
    """Plot 08 — 3D structure with best HOLO prediction."""
    log("[PLOTS] 3D structure of best HOLO method")
    df = pd.DataFrame(rows_all)
    df_holo = df[df["side"] == "HOLO"]
    best = df_holo.loc[df_holo["weighted_top5"].idxmax()]
    best_name = best["method"]
    if best_name in holo_scores:
        scores = holo_scores[best_name]
    elif best_name in pheromones:
        scores = pheromones[best_name]
    else:
        return
    xyz = holo.coords
    s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    other = np.ones(holo.N, dtype=bool)
    other[holo.active_idx] = False
    if holo.ground_truth_idx is not None:
        other[holo.ground_truth_idx] = False
    top_pred = [int(p) for p in best["top5_idx"].split(",")]

    sc = ax.scatter(xyz[other, 0], xyz[other, 1], xyz[other, 2],
                     c=s[other], cmap="viridis", s=45, alpha=0.85,
                     edgecolors="k", linewidths=0.3)
    ax.scatter(xyz[holo.active_idx, 0], xyz[holo.active_idx, 1],
                xyz[holo.active_idx, 2],
                c="red", s=130, marker="o", edgecolors="black",
                linewidths=0.8, label="Active site")
    ax.scatter(xyz[holo.ground_truth_idx, 0], xyz[holo.ground_truth_idx, 1],
                xyz[holo.ground_truth_idx, 2],
                c="gold", s=160, marker="*", edgecolors="black",
                linewidths=0.6, label="Allosteric GT")
    ax.scatter(xyz[top_pred, 0], xyz[top_pred, 1], xyz[top_pred, 2],
                c="#2a9d8f", s=220, marker="X", edgecolors="black",
                linewidths=1.2, label=f"Top-5 prediction")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
    ax.set_title(f"KRAS G12C HOLO — {best_name}\n"
                  f"weighted_top5 = {best['weighted_top5']:.3f}")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Score (normalized)")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, out_dir / "08_3d_structure_best_holo.png",
          "3D structure best HOLO", log)


def plot_adaptive_trajectory(weight_history: list, apo_weights: dict,
                              out_dir: Path, log):
    """Plot 09 — adaptive species weight trajectory."""
    log("[PLOTS] adaptive QuanAnt weight trajectory")
    if not weight_history:
        return
    species = list(weight_history[0].keys())
    iters = list(range(1, len(weight_history) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sps = list(apo_weights.keys())
    ws = [apo_weights[s] for s in sps]
    ax.bar(range(len(sps)), ws, color=[KIND_COLORS["quanant"]] * len(sps),
            edgecolor="black")
    ax.set_xticks(range(len(sps)))
    ax.set_xticklabels([s.replace("_", "\n") for s in sps], fontsize=9)
    ax.set_ylabel("Weight"); ax.set_title("APO-learned species weights")
    ax.grid(axis="y", alpha=0.3)
    for i, w in enumerate(ws):
        ax.text(i, w + 0.01, f"{w:.2f}", ha="center", fontsize=9)

    ax = axes[1]
    species_palette = {
        "qsvd":          "#c1272d",
        "dqaw_timeavg":  "#d62728",
        "dqaw_lifetime": "#8b1a1a",
        "qpagerank":     "#e07b39",
        "heatkernel":    "#a83232",
    }
    for sp in species:
        ax.plot(iters, [h[sp] for h in weight_history], "o-",
                 color=species_palette.get(sp, "#888"),
                 label=sp, linewidth=2)
    ax.set_xlabel("HOLO iteration"); ax.set_ylabel("Species weight")
    ax.set_title("Adaptive online updates (Jaccard agreement)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    _save(fig, out_dir / "09_adaptive_weight_trajectory.png",
          "adaptive weight trajectory", log)


def plot_per_residue_heatmap(holo_scores: dict, holo: ProteinGraph,
                               out_dir: Path, log):
    """Plot 10 — per-residue normalized score heatmap (top-25 residues × methods)."""
    log("[PLOTS] per-residue score heatmap")
    methods = [m for m in holo_scores if m.endswith("_tuned")]
    if len(methods) < 2:
        return
    M = np.stack([holo_scores[m] for m in methods], axis=0)
    M_norm = (M - M.min(axis=1, keepdims=True)) / (
        M.max(axis=1, keepdims=True) - M.min(axis=1, keepdims=True) + 1e-12)
    # Pick top-25 residues by mean (across methods) score
    top_idx = np.argsort(-M_norm.mean(axis=0))[:25]
    M_show = M_norm[:, top_idx]
    resnums = [int(holo.nodes.iloc[i]["resnum"]) for i in top_idx]
    is_gt = [int(i) in set(holo.ground_truth_idx.tolist()) for i in top_idx]

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(M_show, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.replace("_tuned", "") for m in methods], fontsize=9)
    ax.set_xticks(range(len(top_idx)))
    labs = [f"{rn}*" if g else str(rn) for rn, g in zip(resnums, is_gt)]
    ax.set_xticklabels(labs, fontsize=8)
    for i, g in enumerate(is_gt):
        if g:
            ax.get_xticklabels()[i].set_color("limegreen")
            ax.get_xticklabels()[i].set_fontweight("bold")
    ax.set_xlabel("Residue resnum  (green* = ground truth)")
    ax.set_title("Per-residue normalized scores across methods (HOLO, top-25 residues)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Normalized score")
    _save(fig, out_dir / "10_per_residue_heatmap.png",
          "per-residue heatmap", log)


def plot_score_matrix_correlation(apo_scores: dict, holo_scores: dict,
                                    out_dir: Path, log):
    """Plot 11 — Pearson correlation of per-residue scores between methods (APO + HOLO)."""
    log("[PLOTS] method-score correlation heatmaps")
    methods = [m for m in apo_scores if m.endswith("_tuned")]
    if len(methods) < 2:
        return

    def _corr(scores_dict):
        M = np.stack([scores_dict[m] for m in methods], axis=0)
        n = len(methods)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = np.corrcoef(M[i], M[j])[0, 1]
        return C

    Ca = _corr(apo_scores); Ch = _corr(holo_scores)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, C, title in zip(axes, [Ca, Ch], ["APO", "HOLO"]):
        im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(methods)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels([m.replace("_tuned", "") for m in methods],
                            rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels([m.replace("_tuned", "") for m in methods],
                            fontsize=9)
        for i in range(len(methods)):
            for j in range(len(methods)):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                         color="white" if abs(C[i, j]) > 0.5 else "black",
                         fontsize=7)
        ax.set_title(f"{title} — score correlations")
        fig.colorbar(im, ax=ax, shrink=0.7)
    _save(fig, out_dir / "11_method_correlations.png",
          "method correlations", log)


def plot_pareto_front(rows_all: list, out_dir: Path, log):
    """Plot 12 — Pareto front showing APO-vs-HOLO frontier."""
    log("[PLOTS] APO/HOLO Pareto front")
    df = pd.DataFrame(rows_all)
    pivot = df.pivot_table(index="method", columns="side",
                            values="weighted_top5", aggfunc="first")
    pivot = pivot.dropna()
    if pivot.empty or "APO" not in pivot.columns or "HOLO" not in pivot.columns:
        return
    pts = pivot[["APO", "HOLO"]].values
    names = pivot.index.tolist()
    # Compute Pareto front (max APO, max HOLO)
    is_pareto = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        for j, q in enumerate(pts):
            if i == j: continue
            if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                is_pareto[i] = False; break
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(pts[~is_pareto, 0], pts[~is_pareto, 1],
                c="#5e6c75", s=70, alpha=0.7, edgecolor="black",
                linewidths=0.5, label="dominated")
    ax.scatter(pts[is_pareto, 0], pts[is_pareto, 1],
                c="#c1272d", s=130, marker="*", edgecolor="black",
                linewidths=0.6, label="Pareto front")
    pareto = sorted([(p[0], p[1], n) for p, n in zip(pts[is_pareto], np.array(names)[is_pareto])],
                     key=lambda x: x[0])
    if len(pareto) > 1:
        xs = [p[0] for p in pareto]; ys = [p[1] for p in pareto]
        ax.plot(xs, ys, "r--", alpha=0.4)
    for x, y, n in zip(pts[:, 0], pts[:, 1], names):
        ax.annotate(n.replace("_tuned", "*"), (x, y), fontsize=6,
                     xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("APO weighted_top5")
    ax.set_ylabel("HOLO weighted_top5")
    ax.set_title("APO vs HOLO Pareto front")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, out_dir / "12_pareto_front.png",
          "Pareto front", log)


def plot_score_distributions(holo_scores: dict, holo: ProteinGraph,
                              out_dir: Path, log):
    """Plot 13 — per-method score-distribution boxplots, split by GT vs non-GT."""
    log("[PLOTS] score distributions (GT vs non-GT)")
    methods = [m for m in holo_scores if m.endswith("_tuned")][:9]
    if len(methods) < 3:
        return
    gt_set = set(holo.ground_truth_idx.tolist())
    is_gt = np.array([i in gt_set for i in range(holo.N)])
    fig, ax = plt.subplots(figsize=(12, 5))
    data, labels, colors = [], [], []
    for m in methods:
        s = holo_scores[m]
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-12)
        data.append(s_norm[is_gt]); labels.append(f"{m.replace('_tuned','')}\nGT")
        colors.append("#2a9d8f")
        data.append(s_norm[~is_gt]); labels.append(f"non-GT")
        colors.append("#5e6c75")
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Normalized score")
    ax.set_title("Score distributions: GT residues vs non-GT (HOLO, tuned methods)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir / "13_score_distributions.png",
          "score distributions", log)


def plot_runtime(rows_all: list, out_dir: Path, log):
    """Plot 14 — runtime per QuanAnt mode (if recorded)."""
    log("[PLOTS] QuanAnt runtimes")
    df = pd.DataFrame(rows_all)
    df = df[df["stage"].isin(["quanant", "adaptive"]) & df["side"].eq("HOLO")
             & df["weighted_top5"].notna()].copy()
    if "runtime_s" not in df or df["runtime_s"].isna().all():
        return
    df = df.dropna(subset=["runtime_s"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#7b3294" if "adaptive" in m else "#1f4e79" for m in df["method"]]
    ax.barh(df["method"], df["runtime_s"], color=colors,
             edgecolor="black", linewidth=0.4)
    for y, (rt, sc) in enumerate(zip(df["runtime_s"], df["weighted_top5"])):
        ax.text(rt + 0.3, y, f"{rt:.1f}s  →  HOLO {sc:.3f}",
                 va="center", fontsize=8)
    ax.set_xlabel("Runtime (s)"); ax.set_title("QuanAnt runtime vs HOLO score")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_dir / "14_runtime_per_quanant.png",
          "runtimes", log)


def plot_summary_table(rows_all: list, out_dir: Path, log):
    """Plot 15 — summary table image."""
    log("[PLOTS] summary table figure")
    df = pd.DataFrame(rows_all)
    df_holo = df[df["side"] == "HOLO"].copy().sort_values(
        "weighted_top5", ascending=False)
    top10 = df_holo.head(10)
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(top10) + 2))
    ax.axis("off")
    cell_text = []
    for _, r in top10.iterrows():
        cell_text.append([
            r["method"][:36],
            f"{r['weighted_top5']:.3f}",
            f"{r['P5_k0']:.2f}",
            f"{r['P5_k2']:.2f}",
            r["top5_resnums"][:36],
        ])
    table = ax.table(
        cellText=cell_text,
        colLabels=["Method", "wt5", "P@5_k0", "P@5_k2", "Top-5 resnums"],
        loc="center", cellLoc="left", colLoc="center",
        colWidths=[0.30, 0.10, 0.10, 0.10, 0.40],
    )
    table.auto_set_font_size(False); table.set_fontsize(9)
    table.scale(1, 1.6)
    for i in range(5):
        table[(0, i)].set_facecolor("#1f4e79")
        table[(0, i)].set_text_props(color="white", weight="bold")
    for r in range(1, len(top10) + 1):
        for c in range(5):
            table[(r, c)].set_facecolor("#f7f7f7" if r % 2 else "white")
    ax.set_title("Final HOLO leaderboard — top 10 methods", pad=15, fontsize=13)
    _save(fig, out_dir / "15_top10_leaderboard.png",
          "top-10 leaderboard", log)


# =======================================================================
# MAIN ORCHESTRATOR
# =======================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="input_BCR_ABL1",
                          help="Directory with preprocessed BCR_ABL1 data.")
    parser.add_argument("--out", default="output_BCR_ABL1",
                          help="Output directory.")
    parser.add_argument("--quick", action="store_true",
                          help="Run smaller grids and fewer ants (~3 min).")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "matrices").mkdir(parents=True, exist_ok=True)
    (out / "hit_lists").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    # Logger that writes to both stdout and a log file
    log_path = out / "run.log"
    log_fh = open(log_path, "w", encoding="utf-8")

    def log(*args_print, end="\n"):
        msg = " ".join(str(a) for a in args_print) + (end if end != "\n" else "")
        if end == "\n":
            print(msg); log_fh.write(msg + "\n")
        else:
            print(msg, end=""); log_fh.write(msg)
        log_fh.flush()

    t_start = time.time()
    log("=" * 70)
    log("QuanAllo full pipeline — KRAS G12C")
    log(f"  data : {args.data}")
    log(f"  out  : {out}")
    log(f"  quick: {args.quick}")
    log("=" * 70)

    # --- Load data ---
    apo, holo = load_kras(Path(args.data))
    log(f"\n[load] APO  = {apo}")
    log(f"[load] HOLO = {holo}")

    all_rows: list = []

    # --- STAGE 1 ---
    best_params, _, grid_results = stage1_grid_search(
        apo, holo, quick=args.quick, log=log)
    with open(out / "matrices" / "tuned_hyperparams.json", "w") as fh:
        json.dump(best_params, fh, indent=2)
    log(f"\n[save] tuned_hyperparams.json")

    # Save full grids
    for m, df in grid_results.items():
        df.to_csv(out / "matrices" / f"grid_{m}.csv", index=False)

    # --- STAGE 2 ---
    rows_dt, apo_scores, holo_scores = stage2_default_vs_tuned(
        apo, holo, best_params, log=log)
    all_rows += rows_dt

    # Save score matrices
    methods = sorted(apo_scores.keys())
    M_apo = np.stack([apo_scores[m] for m in methods], axis=1)
    M_holo = np.stack([holo_scores[m] for m in methods], axis=1)
    np.save(out / "matrices" / "score_matrix_apo.npy", M_apo)
    np.save(out / "matrices" / "score_matrix_holo.npy", M_holo)
    with open(out / "matrices" / "score_matrix_columns.txt", "w") as fh:
        fh.write("\n".join(methods))
    log(f"\n[save] score_matrix_apo.npy  shape={M_apo.shape}")
    log(f"[save] score_matrix_holo.npy shape={M_holo.shape}")

    # --- STAGE 3 ---
    forward_log, sel_summary = stage3_species_selection(
        apo_scores, holo_scores, apo, holo, log=log)
    pd.DataFrame(forward_log).to_csv(
        out / "matrices" / "species_selection_forward.csv", index=False)
    sel_summary["all_subsets"].to_csv(
        out / "matrices" / "species_selection_all_subsets.csv", index=False)

    # --- STAGE 4 ---
    rows_ens = stage4_ensemble_methods(
        apo_scores, holo_scores, apo, holo, log=log)
    all_rows += rows_ens

    # --- STAGE 5 ---
    rows_qa, pher_qa = stage5_quanant_modes(
        apo, holo, quick=args.quick, log=log)
    all_rows += rows_qa

    # --- STAGE 6 ---
    rows_aq, pher_aq, weight_history = stage6_adaptive_quanant(
        apo, holo, quick=args.quick, log=log)
    all_rows += rows_aq

    # Merge for downstream plotting
    pheromones = {**pher_qa, **pher_aq}
    holo_scores_plus = {**holo_scores, **pheromones}

    # --- Save final matrices ---
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(out / "matrices" / "long_form.csv", index=False)
    log(f"\n[save] long_form.csv ({len(df_all)} rows)")

    # Pivot summary: best per stage on each side
    pivot_rows = []
    for stage in df_all["stage"].unique():
        for side in ["APO", "HOLO"]:
            d = df_all[(df_all["stage"] == stage) & (df_all["side"] == side)]
            if len(d) == 0:
                continue
            best = d.loc[d["weighted_top5"].idxmax()]
            pivot_rows.append({
                "stage": stage, "side": side,
                "best_method": best["method"],
                "weighted_top5": best["weighted_top5"],
                "P5_k2": best["P5_k2"],
                "top5_resnums": best["top5_resnums"],
            })
    df_pivot = pd.DataFrame(pivot_rows)
    df_pivot.to_csv(out / "matrices" / "pivot_summary.csv", index=False)
    log(f"[save] pivot_summary.csv ({len(df_pivot)} rows)")

    # --- Save hit lists ---
    log("\n[save] hit lists per method")
    for _, r in df_all[df_all["side"] == "HOLO"].iterrows():
        top_idx = [int(x) for x in r["top5_idx"].split(",")]
        hits = []
        for rank, p in enumerate(top_idx, start=1):
            info = holo.residue_at(p)
            hits.append({
                "rank": rank, "method": r["method"],
                **info,
                "is_GT": int(p) in set(holo.ground_truth_idx.tolist()),
                "hop_to_gt": int(r["hops_to_gt"].split(",")[rank - 1]),
            })
        fname = r["method"].replace("/", "_") + ".csv"
        pd.DataFrame(hits).to_csv(out / "hit_lists" / fname, index=False)

    # ====== PLOTS ======
    log("\n" + "=" * 70)
    log("[plots] generating 15 comparison plots")
    log("=" * 70)
    plots_dir = out / "plots"

    plot_grid_search(grid_results, plots_dir, log)
    plot_default_vs_tuned(rows_dt, plots_dir, log)
    plot_apo_holo_scatter(rows_dt, plots_dir, log)
    plot_species_selection(forward_log, sel_summary, plots_dir, log)
    plot_ensemble_comparison(all_rows, plots_dir, log)
    plot_precision_curves(holo_scores_plus, holo, plots_dir, log)
    plot_top5_jaccard(holo_scores, holo, plots_dir, log)
    plot_3d_best(all_rows, holo_scores, pheromones, holo, plots_dir, log)
    # Adaptive trajectory
    try:
        # we need apo_weights from the adaptive run — recompute from row metadata
        species5 = ["qsvd", "dqaw_timeavg", "dqaw_lifetime",
                     "qpagerank", "heatkernel"]
        apo_weights_for_plot = weight_history[0] if weight_history else {}
        plot_adaptive_trajectory(weight_history, apo_weights_for_plot,
                                  plots_dir, log)
    except Exception as e:
        log(f"  (skipped adaptive trajectory plot: {e})")
    plot_per_residue_heatmap(holo_scores, holo, plots_dir, log)
    plot_score_matrix_correlation(apo_scores, holo_scores, plots_dir, log)
    plot_pareto_front(all_rows, plots_dir, log)
    plot_score_distributions(holo_scores, holo, plots_dir, log)
    plot_runtime(all_rows, plots_dir, log)
    plot_summary_table(all_rows, plots_dir, log)

    # ====== STAGE 7 — Submission folder ======
    submission_dir = stage7_create_submission(
        apo=apo, holo=holo, all_rows=all_rows,
        apo_scores=apo_scores, holo_scores=holo_scores,
        pheromones=pheromones, best_params=best_params,
        out_dir=out, log=log,
    )

    # ====== FINAL SUMMARY ======
    log("\n" + "=" * 70)
    log("FINAL HOLO LEADERBOARD")
    log("=" * 70)
    df_holo = df_all[df_all["side"] == "HOLO"].sort_values(
        "weighted_top5", ascending=False)
    log(df_holo.head(10)[
        ["method", "weighted_top5", "P5_k2", "top5_resnums"]
    ].to_string(index=False))

    log(f"\n[done] total runtime: {time.time() - t_start:.1f} s")
    log(f"[done] outputs in: {out}")
    log(f"[done] submission folder: {submission_dir}")
    log_fh.close()


if __name__ == "__main__":
    main()
