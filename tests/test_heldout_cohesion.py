import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.eval.coherence import heldout_edge_cohesion


def test_perfectly_aligned_edges_give_full_cohesion():
    ids = ["a", "b", "c", "d"]
    labels = [0, 0, 1, 1]
    obs, exp, lift = heldout_edge_cohesion(labels, ids, [["a", "b"], ["c", "d"]])
    assert obs == 1.0
    assert np.isclose(exp, (2 * 1 + 2 * 1) / (4 * 3))
    assert np.isclose(lift, 3.0)


def test_edges_split_across_clusters_and_unknown_members():
    ids = ["a", "b", "c", "d"]
    labels = [0, 0, 1, 1]
    # (a,c) never shares a cluster; "zz" is not in ids, so the 2nd edge is just (a,b)
    obs, _, _ = heldout_edge_cohesion(labels, ids, [["a", "c"], ["a", "b", "zz"], ["zz"]])
    assert obs == 0.5
