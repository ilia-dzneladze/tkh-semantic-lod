import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.hypergraph import clique_expansion, build_structural_affinity, high_arity_weight_share


def test_each_member_spreads_one_unit_over_its_co_members():
    A = clique_expansion([[0, 1, 2], [0, 1]], 4).toarray()
    assert np.allclose(A, A.T)
    assert A[0, 1] == 0.5 + 1.0  # half from the 3-edge, all of the 2-edge
    assert A[0, 2] == A[1, 2] == 0.5
    assert A[3].sum() == 0 and np.all(np.diag(A) == 0)
    # the 3-edge adds 3/2 in total, not 3 pairs' worth
    assert np.isclose(clique_expansion([[0, 1, 2]], 3).toarray().sum() / 2, 1.5)


def test_weights_scale_and_small_groups_add_nothing():
    A = clique_expansion([[0, 1], [2], []], 3, weights=[4, 9, 9]).toarray()
    assert A[0, 1] == 4 and A[2].sum() == 0


def test_structural_affinity_drops_articles_and_authors():
    snap = SimpleNamespace(
        concept_ids={"m1", "m2", "m3"},
        hyperedges=[{"members": ["m1", "m2", "art_1"]}, {"members": ["m3", "auth_1"]}])
    A, ids = build_structural_affinity(snap)
    assert ids == ["m1", "m2", "m3"]
    A = A.toarray()
    assert A[0, 1] == 1.0  # a 2-concept edge once the article is dropped
    assert A[2].sum() == 0  # one concept left, so no pair


def test_high_arity_share():
    big = {"members": [f"c{i}" for i in range(11)]}
    small = {"members": ["c0", "c1"]}
    snap = SimpleNamespace(concept_ids={f"c{i}" for i in range(11)}, hyperedges=[big, small])
    unweighted, weighted = high_arity_weight_share(snap)
    assert np.isclose(unweighted, 55 / 56) and np.isclose(weighted, 5.5 / 6.5)
