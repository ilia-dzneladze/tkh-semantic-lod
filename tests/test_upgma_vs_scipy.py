"""sparse_upgma should be plain average linkage with missing edges read as
similarity 0, so it has to agree with scipy's dense implementation."""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.cluster import sparse_upgma, cut_to_k_clusters


def random_similarity(n, density, seed):
    rng = np.random.default_rng(seed)
    S = np.triu(rng.uniform(0.05, 0.95, (n, n)) * (rng.random((n, n)) < density), 1)
    return S + S.T


def scipy_average(S):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return linkage(squareform(D, checks=False), method="average")


def test_matches_scipy_on_connected_graph():
    for seed in range(5):
        S = random_similarity(40, 0.5, seed)
        Z, forced = sparse_upgma(sp.csr_matrix(S), 40)
        Zs = scipy_average(S)
        assert not forced.any()
        np.testing.assert_allclose(Z[:, 2], Zs[:, 2], atol=1e-12)
        for k in (2, 5, 12):
            a = cut_to_k_clusters(Z, 40, k)
            b = cut_to_k_clusters(Zs, 40, k)
            assert adjusted_rand_score(a, b) == 1.0


def test_matches_scipy_heights_on_sparse_graph_with_forced_merges():
    # disconnected pieces: forced merges happen at distance 1 in both, merge
    # order among ties can differ, so compare heights only
    for seed in range(5):
        S = random_similarity(40, 0.06, seed)
        Z, forced = sparse_upgma(sp.csr_matrix(S), 40)
        Zs = scipy_average(S)
        assert forced.any()
        np.testing.assert_allclose(np.sort(Z[:, 2]), np.sort(Zs[:, 2]), atol=1e-12)
        assert np.all(np.diff(Z[:, 2]) >= -1e-12)  # no inversions


def test_sizes_equal_replicated_points():
    # a point with weight w should behave like w identical copies
    S = random_similarity(12, 0.7, 7)
    sizes = np.array([1, 3, 1, 2, 1, 1, 4, 1, 1, 2, 1, 1])
    Z_w, _ = sparse_upgma(sp.csr_matrix(S), 12, sizes=sizes)
    idx = np.repeat(np.arange(12), sizes)
    S_rep = S[np.ix_(idx, idx)]
    S_rep[idx[:, None] == idx[None, :]] = 1.0
    np.fill_diagonal(S_rep, 0.0)
    Z_rep = scipy_average(S_rep)
    heights_rep = Z_rep[:, 2][Z_rep[:, 2] > 1e-12]
    np.testing.assert_allclose(Z_w[:, 2], heights_rep, atol=1e-12)
