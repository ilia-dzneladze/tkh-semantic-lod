import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.collapse import coarse_structural_affinity
from tkh.io import Snapshot
from tkh.pipeline import build_levels


def test_coarse_affinity_spreads_weight_over_spanned_supernodes():
    supers = ["A", "B", "C"]
    edges = [
        {"members": ["A", "B", "C"], "weight": 2},   # 3 super-nodes -> 2/2 per pair
        {"members": ["A", "art_1"], "weight": 5},    # 1 super-node after dropping context
        {"members": ["B", "C", "art_2"], "weight": 1},  # 2 super-nodes -> 1/1
    ]
    A = coarse_structural_affinity(edges, supers).toarray()
    assert np.allclose(A, A.T)
    assert A[0, 1] == 1.0 and A[0, 2] == 1.0
    assert A[1, 2] == 2.0
    assert A[0, 0] == 0.0


def _toy_snapshot(n_nodes=60, n_edges=80, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"m_{i:03d}" for i in range(n_nodes)]
    nodes = {nid: {"id": nid, "type": "method", "surface_form": nid} for nid in ids}
    nodes["art_0"] = {"id": "art_0", "type": "article", "surface_form": "paper"}
    edges = []
    for e in range(n_edges):
        k = int(rng.integers(2, 6))
        members = list(rng.choice(ids, size=k, replace=False))
        if e % 3 == 0:
            members.append("art_0")
        edges.append({"id": f"e{e}", "members": members, "relation_type": "uses"})
    return Snapshot(cutoff_year=2020, nodes=nodes, hyperedges=edges, concept_ids=set(ids)), sorted(ids)


@pytest.mark.parametrize("coarsening", ["multilevel", "multilevel_sizenorm"])
def test_multilevel_is_laminar_and_respects_budget(coarsening):
    snap, ids = _toy_snapshot()
    emb = np.random.default_rng(1).normal(size=(len(ids), 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    targets = [3, 8, 20]
    out = build_levels(snap, ids, emb, level_targets=targets, coarsening=coarsening)
    labels = out["labels_by_level"]
    for level, k in enumerate(targets):
        assert len(set(labels[level])) <= k
    for level in range(len(targets) - 1):
        parent_of = {}
        for fine, coarse in zip(labels[level + 1], labels[level]):
            assert parent_of.setdefault(fine, coarse) == coarse


def test_finest_level_identical_across_variants():
    snap, ids = _toy_snapshot()
    emb = np.random.default_rng(2).normal(size=(len(ids), 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    targets = [3, 8, 20]
    a = build_levels(snap, ids, emb, level_targets=targets, coarsening="dendrogram")
    b = build_levels(snap, ids, emb, level_targets=targets, coarsening="multilevel")
    assert np.array_equal(a["labels_by_level"][2], b["labels_by_level"][2])


def test_weighted_upgma_keeps_point_counts_in_linkage():
    import scipy.sparse as sp
    from tkh.cluster import sparse_upgma
    A = sp.csr_matrix(np.array([[0, .9, .1], [.9, 0, .2], [.1, .2, 0]]))
    Z, _ = sparse_upgma(A, 3, sizes=np.array([5.0, 1.0, 2.0]))
    assert Z[-1, 3] == 3  # scipy expects point counts, not weights
    # weighted average: (5*.1 + 1*.2) / 6
    assert np.isclose(1 - Z[-1, 2], (5 * .1 + 1 * .2) / 6)
