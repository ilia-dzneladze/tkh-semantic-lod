import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.eval.stability import cross_snapshot_stability  # noqa: E402


def _hier(groups):
    """groups: list of member lists, all at level 0."""
    return {"super_nodes": [{"id": f"S{i}", "level": 0, "member_ids": list(g)}
                            for i, g in enumerate(groups)]}


def test_identical_partitions_give_ari_one_and_a_degenerate_ci():
    groups = [[f"n{i}" for i in range(0, 20)], [f"n{i}" for i in range(20, 40)]]
    hs = {2020: _hier(groups), 2022: _hier(groups), 2024: _hier(groups)}
    r = cross_snapshot_stability(hs, 1, n_boot=200)["by_level"][0]
    assert r["mean_ari"] == 1.0 and r["ci95"] == [1.0, 1.0]


def test_resampling_ci_brackets_the_estimate_and_is_seeded():
    rng = np.random.default_rng(3)
    ids = [f"n{i}" for i in range(300)]

    def noisy(p):
        lab = [i % 4 if rng.random() > p else int(rng.integers(4)) for i in range(300)]
        return _hier([[ids[i] for i in range(300) if lab[i] == k] for k in range(4)])

    hs = {2020: noisy(0.0), 2022: noisy(0.3), 2024: noisy(0.3)}
    a = cross_snapshot_stability(hs, 1, n_boot=300, seed=1)
    b = cross_snapshot_stability(hs, 1, n_boot=300, seed=1)
    r = a["by_level"][0]
    assert r["ci95"] == b["by_level"][0]["ci95"]
    assert r["ci95"][0] < r["mean_ari"] < r["ci95"][1]
    assert all(t["ci95"][0] <= t["ari"] <= t["ci95"][1] for t in a["detail"])
    # the old interval is kept alongside, and three wildly different
    # transitions make it far wider than the node bootstrap
    lo_t, hi_t = r["ci95_t_over_transitions"]
    assert hi_t - lo_t > r["ci95"][1] - r["ci95"][0]


def test_nodes_new_at_t_plus_1_are_ignored():
    # the 2022 partition adds a new group of new nodes; the shared nodes
    # keep their grouping, so agreement on shared nodes is perfect
    g2020 = [["a", "b", "c"], ["d", "e", "f"]]
    g2022 = [["a", "b", "c"], ["d", "e", "f"], ["x", "y", "z"]]
    r = cross_snapshot_stability({2020: _hier(g2020), 2022: _hier(g2022)}, 1, n_boot=50)
    assert r["detail"][0]["n_common_nodes"] == 6 and r["detail"][0]["ari"] == 1.0


def test_many_small_clusters_do_not_bias_the_interval():
    # 200 clusters of 7: resampling WITH replacement put duplicates in the
    # same cluster in both partitions and pushed the whole interval above
    # the real ARI (seen on the level-2 data). Half-sampling must not.
    rng = np.random.default_rng(0)
    n, k = 1400, 200
    ids = [f"n{i}" for i in range(n)]
    base = np.arange(n) % k
    moved = np.where(rng.random(n) < 0.35, rng.integers(0, k, n), base)
    h0 = _hier([[ids[i] for i in range(n) if base[i] == c] for c in range(k)])
    h1 = _hier([[ids[i] for i in range(n) if moved[i] == c] for c in range(k)])
    r = cross_snapshot_stability({2020: h0, 2022: h1}, 1, n_boot=300)["detail"][0]
    assert r["ci95"][0] < r["ari"] < r["ci95"][1]
