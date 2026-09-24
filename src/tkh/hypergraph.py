"""Structural affinity from hyperedges (T2): a weighted clique expansion.

Each hyperedge with k concept members adds 1/(k-1) to each of its pairs,
so a big hyperedge counts for more than a small one, but linearly in k
rather than quadratically. Why: DESIGN_NOTES.md sections 3 and 4.
"""
from collections import defaultdict

import scipy.sparse as sp


def clique_expansion(groups, n, weights=None):
    """Symmetric n x n sparse matrix where each group (a list of row
    indices) adds weight/(k-1) to every pair of its k members. Groups with
    fewer than 2 members add nothing. weights default to 1 per group."""
    pair_weight = defaultdict(float)
    for g, members in enumerate(groups):
        k = len(members)
        if k < 2:
            continue
        share = (1 if weights is None else weights[g]) / (k - 1)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = sorted((members[i], members[j]))
                pair_weight[(a, b)] += share
    rows, cols, vals = [], [], []
    for (a, b), w in pair_weight.items():
        rows += [a, b]
        cols += [b, a]
        vals += [w, w]
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def concept_members(snap, edge):
    """The edge's members that get clustered (article/author members dropped)."""
    return [m for m in edge["members"] if m in snap.concept_ids]


def build_structural_affinity(snap):
    """(A, ids): the clique expansion over the snapshot's concept nodes,
    rows in sorted id order."""
    ids = sorted(snap.concept_ids)
    idx = {nid: i for i, nid in enumerate(ids)}
    groups = [[idx[m] for m in concept_members(snap, e)] for e in snap.hyperedges]
    return clique_expansion(groups, len(ids)), ids


def high_arity_weight_share(snap, min_arity=11):
    """Share of all pair weight that comes from hyperedges with at least
    min_arity concept members, as (unweighted, with the 1/(k-1) weighting).
    An edge of k members carries k(k-1)/2 unweighted and k/2 weighted.
    DESIGN_NOTES.md section 4."""
    ks = [len(concept_members(snap, e)) for e in snap.hyperedges]
    ks = [k for k in ks if k >= 2]
    unweighted = [k * (k - 1) / 2 for k in ks]
    weighted = [k / 2 for k in ks]
    return (sum(u for u, k in zip(unweighted, ks) if k >= min_arity) / sum(unweighted),
            sum(w for w, k in zip(weighted, ks) if k >= min_arity) / sum(weighted))
