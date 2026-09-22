"""Structural affinity from hyperedges (T2): a weighted clique expansion.

This is a pairwise projection of the hypergraph, with each hyperedge's
weight spread over its pairs so large hyperedges don't dominate.
Weighting rationale: DESIGN_NOTES.md section 3.
Naive-projection comparison rationale: DESIGN_NOTES.md section 4.
"""
from collections import defaultdict
import numpy as np
import scipy.sparse as sp


def build_structural_affinity(snap, weighted=True):
    """Sparse |concept_ids| x |concept_ids| structural affinity matrix.

    weighted=True -> 1/(arity-1) per pair, arity counted over concept members.
    weighted=False -> unweighted clique expansion (1.0 per pair), for comparison.
    Every edge with at least 2 concept members contributes, restricted to
    those members; article/author members are dropped from the edge.
    """
    ids = sorted(snap.concept_ids)
    idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)

    pair_weight = defaultdict(float)
    n_used_edges = 0
    n_skipped_context_edges = 0
    for e in snap.hyperedges:
        members = [m for m in e.get("members", [])]
        concept_members = [m for m in members if m in idx]
        if len(concept_members) < 2:
            continue
        if len(concept_members) != len(members):
            n_skipped_context_edges += 1  # edge also touches article/author
        arity = len(concept_members)
        share = 1.0 / (arity - 1) if weighted else 1.0
        n_used_edges += 1
        for i in range(arity):
            for j in range(i + 1, arity):
                a, b = idx[concept_members[i]], idx[concept_members[j]]
                if a > b:
                    a, b = b, a
                pair_weight[(a, b)] += share

    if not pair_weight:
        empty_stats = {
            "n_concept_nodes": n, "n_used_edges": 0, "n_skipped_context_edges": 0,
            "n_nonzero_pairs": 0, "density": 0.0,
        }
        return sp.csr_matrix((n, n)), ids, empty_stats

    rows, cols, vals = [], [], []
    for (a, b), w in pair_weight.items():
        rows += [a, b]
        cols += [b, a]
        vals += [w, w]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    stats = {
        "n_concept_nodes": n,
        "n_used_edges": n_used_edges,
        "n_skipped_context_edges": n_skipped_context_edges,
        "n_nonzero_pairs": len(pair_weight),
        "density": len(pair_weight) / (n * (n - 1) / 2) if n > 1 else 0.0,
    }
    return A, ids, stats


def projection_loss_report(snap):
    """Compare weighted vs. unweighted clique expansion by how much pairwise
    weight comes from high-arity (>10) hyperedges under each. Both are
    projections; this measures weight distribution, not clustering quality.
    See DESIGN_NOTES.md section 4."""
    A_native, ids, stats_native = build_structural_affinity(snap, weighted=True)
    A_naive, _, stats_naive = build_structural_affinity(snap, weighted=False)

    # mass contributed by hyperedges of arity > 10, native vs naive
    high_arity_native_mass = 0.0
    high_arity_naive_mass = 0.0
    total_native_mass = 0.0
    total_naive_mass = 0.0
    idx = {nid: i for i, nid in enumerate(ids)}
    for e in snap.hyperedges:
        concept_members = [m for m in e.get("members", []) if m in idx]
        arity = len(concept_members)
        if arity < 2:
            continue
        n_pairs = arity * (arity - 1) / 2
        naive_mass = n_pairs * 1.0
        native_mass = n_pairs * (1.0 / (arity - 1))
        total_naive_mass += naive_mass
        total_native_mass += native_mass
        if arity > 10:
            high_arity_naive_mass += naive_mass
            high_arity_native_mass += native_mass

    return {
        "native_affinity_stats": stats_native,
        "naive_affinity_stats": stats_naive,
        "high_arity_share_of_total_mass_naive": (
            high_arity_naive_mass / total_naive_mass if total_naive_mass else 0.0),
        "high_arity_share_of_total_mass_native": (
            high_arity_native_mass / total_native_mass if total_native_mass else 0.0),
        "interpretation": (
            "Under the unweighted clique expansion, hyperedges of arity>10 hold "
            "most of the pairwise weight. The 1/(arity-1) weighting keeps each "
            "hyperedge's total contribution linear in arity instead of quadratic."
        ),
    }
