"""T6 stability: perturbation-based (rebuild after removing 10% of
hyperedges, several seeds) and real cross-snapshot (from T3's persistent
ids). Both reported as mean + CI, not a single point estimate, per the
brief's instruction to not trust a lone number.
"""
import dataclasses
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import adjusted_rand_score

from tkh.hypergraph import build_structural_affinity
from tkh.embeddings import semantic_knn_graph
from tkh.cluster import combine_affinities, sparse_upgma, cut_to_k_clusters


def labels_from_hierarchy(hierarchy, level, ids):
    """Persistent super-node id each node in `ids` belongs to at `level`,
    used as the cluster "label" for ARI (arbitrary hashable labels are
    fine for sklearn's adjusted_rand_score, they don't need to be ints)."""
    node_to_pid = {}
    for sn in hierarchy["super_nodes"]:
        if sn["level"] != level:
            continue
        for nid in sn["member_ids"]:
            node_to_pid[nid] = sn["id"]
    return [node_to_pid.get(nid) for nid in ids]


def perturb_snapshot(snap, remove_frac=0.10, seed=0):
    rng = np.random.default_rng(seed)
    edges = snap.hyperedges
    n = len(edges)
    n_remove = int(round(n * remove_frac))
    drop_idx = set(rng.choice(n, size=n_remove, replace=False).tolist())
    kept = [e for i, e in enumerate(edges) if i not in drop_idx]
    return dataclasses.replace(snap, hyperedges=kept)


def _mean_ci95(values):
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, 0.0, [mean, mean]
    std = float(arr.std(ddof=1))
    sem = std / np.sqrt(len(arr))
    if sem == 0:
        ci = [mean, mean]
    else:
        lo, hi = scipy_stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=sem)
        ci = [float(lo), float(hi)]
    return mean, std, ci


def perturbation_stability(snap, embedding_cache, level_targets, alpha,
                            original_hierarchy, n_seeds=5, remove_frac=0.10,
                            base_seed=1000):
    """Rebuild the clustering n_seeds times, each time with remove_frac of
    the snapshot's hyperedges dropped at random, and compare each rebuild
    to the original via ARI. Node set is held fixed (only edges are
    perturbed), so label arrays line up index-for-index with no need to
    restrict to a common subset."""
    A_struct0, ids, _ = build_structural_affinity(snap, weighted=True)
    original_labels = {
        level_idx: labels_from_hierarchy(original_hierarchy, level_idx, ids)
        for level_idx in range(len(level_targets))
    }

    emb = np.stack([embedding_cache[nid] for nid in ids])
    A_sem = semantic_knn_graph(emb, k=15)  # unaffected by edge removal, computed once

    per_level_aris = {level_idx: [] for level_idx in range(len(level_targets))}
    n_forced_by_seed = []

    for s in range(n_seeds):
        pert = perturb_snapshot(snap, remove_frac=remove_frac, seed=base_seed + s)
        A_struct, pert_ids, _ = build_structural_affinity(pert, weighted=True)
        assert pert_ids == ids, "perturbation must not change the node set"

        A_combined = combine_affinities(A_struct, A_sem, alpha=alpha)
        Z, forced = sparse_upgma(A_combined, len(ids))
        n_forced_by_seed.append(int(forced.sum()))

        for level_idx, k in enumerate(level_targets):
            labels = cut_to_k_clusters(Z, len(ids), k)
            ari = adjusted_rand_score(original_labels[level_idx], labels)
            per_level_aris[level_idx].append(float(ari))

    summary = {}
    for level_idx, aris in per_level_aris.items():
        mean, std, ci = _mean_ci95(aris)
        summary[level_idx] = {
            "mean_ari": mean, "std_ari": std, "ci95": ci,
            "values": aris, "n_seeds": len(aris),
        }
    return {
        "remove_frac": remove_frac, "n_seeds": n_seeds,
        "n_forced_merges_by_seed": n_forced_by_seed,
        "by_level": summary,
    }


def cross_snapshot_stability(hierarchies_by_year, n_levels):
    """ARI between each pair of consecutive snapshots, per level, restricted
    to the node ids present in both (a node introduced at t+1 can't count
    toward or against agreement it wasn't there to participate in)."""
    years = sorted(hierarchies_by_year)
    transitions = list(zip(years[:-1], years[1:]))

    per_level = {level_idx: [] for level_idx in range(n_levels)}
    detail = []

    for y0, y1 in transitions:
        h0, h1 = hierarchies_by_year[y0], hierarchies_by_year[y1]
        for level_idx in range(n_levels):
            ids0 = {nid for sn in h0["super_nodes"] if sn["level"] == level_idx
                     for nid in sn["member_ids"]}
            ids1 = {nid for sn in h1["super_nodes"] if sn["level"] == level_idx
                     for nid in sn["member_ids"]}
            common = sorted(ids0 & ids1)
            if len(common) < 2:
                continue
            labels0 = labels_from_hierarchy(h0, level_idx, common)
            labels1 = labels_from_hierarchy(h1, level_idx, common)
            ari = adjusted_rand_score(labels0, labels1)
            per_level[level_idx].append(float(ari))
            detail.append({"from_year": y0, "to_year": y1, "level": level_idx,
                            "n_common_nodes": len(common), "ari": float(ari)})

    summary = {}
    for level_idx, aris in per_level.items():
        mean, std, ci = _mean_ci95(aris)
        summary[level_idx] = {
            "mean_ari": mean, "std_ari": std, "ci95": ci,
            "values": aris, "n_transitions": len(aris),
        }
    return {"by_level": summary, "detail": detail}
