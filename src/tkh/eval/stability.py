"""T6 stability, both as a mean with a CI: rebuild after removing 10% of
the hyperedges (several seeds), and agreement between consecutive
snapshots on the nodes they share.
"""
import dataclasses

import numpy as np
from sklearn.metrics import adjusted_rand_score

from tkh.embeddings import semantic_knn_graph
from tkh.eval.stats import mean_ci95


def labels_from_hierarchy(hierarchy, level, ids):
    """The super-node id each node in `ids` belongs to at `level` (None if
    it's in none), to use as cluster labels for ARI."""
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


def perturbation_stability(snap, embedding_cache, level_targets, alpha,
                            original_hierarchy, n_seeds=5, remove_frac=0.10,
                            base_seed=1000, coarsening="dendrogram"):
    """Rebuild the clustering n_seeds times, each time with remove_frac of
    the hyperedges dropped at random, and compare each rebuild with the
    original by ARI, per level. Only edges are removed, so the node set is
    the same. Only the structural term is disturbed, which ties this
    result to alpha: DESIGN_NOTES.md section 17."""
    from tkh.pipeline import build_levels, KNN_K

    ids = sorted(snap.concept_ids)
    original_labels = {
        level_idx: labels_from_hierarchy(original_hierarchy, level_idx, ids)
        for level_idx in range(len(level_targets))
    }

    emb = np.stack([embedding_cache[nid] for nid in ids])
    A_sem = semantic_knn_graph(emb, k=KNN_K)  # unaffected by edge removal, computed once

    per_level_aris = {level_idx: [] for level_idx in range(len(level_targets))}
    n_forced_by_seed = []

    for s in range(n_seeds):
        pert = perturb_snapshot(snap, remove_frac=remove_frac, seed=base_seed + s)
        built = build_levels(pert, ids, emb, alpha=alpha, level_targets=level_targets,
                             coarsening=coarsening, A_sem=A_sem)
        n_forced_by_seed.append(built["forced_by_level"])

        for level_idx, labels in built["labels_by_level"].items():
            ari = adjusted_rand_score(original_labels[level_idx], labels)
            per_level_aris[level_idx].append(float(ari))

    summary = {}
    for level_idx, aris in per_level_aris.items():
        mean, std, ci = mean_ci95(aris)
        summary[level_idx] = {
            "mean_ari": mean, "std_ari": std, "ci95": ci,
            "values": aris, "n_seeds": len(aris),
        }
    return {
        "remove_frac": remove_frac, "n_seeds": n_seeds,
        "n_forced_merges_by_seed": n_forced_by_seed,
        "by_level": summary,
    }


def _half_sample_ari(labels0, labels1, rng, n_boot):
    """ARI on n_boot random halves (without replacement) of the shared
    nodes. Not an ordinary bootstrap: resampling with replacement puts
    duplicate nodes in the same cluster in both partitions, which inflates
    ARI when clusters are small. See DESIGN_NOTES.md section 20."""
    a, b = np.asarray(labels0, dtype=object), np.asarray(labels1, dtype=object)
    n = len(a)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n // 2, replace=False)
        out[i] = adjusted_rand_score(a[idx], b[idx])
    return out


def cross_snapshot_stability(hierarchies_by_year, n_levels, n_boot=1000, seed=0):
    """ARI between consecutive snapshots, per level, on the nodes present in
    both. CIs come from n_boot random halves of the shared nodes, per
    transition and for the mean over transitions; the older t-interval
    over the transition values is kept as ci95_t_over_transitions.
    DESIGN_NOTES.md section 20."""
    years = sorted(hierarchies_by_year)
    transitions = list(zip(years[:-1], years[1:]))
    rng = np.random.default_rng(seed)

    per_level = {level_idx: [] for level_idx in range(n_levels)}
    boot_by_level = {level_idx: [] for level_idx in range(n_levels)}
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
            row = {"from_year": y0, "to_year": y1, "level": level_idx,
                   "n_common_nodes": len(common), "ari": float(ari)}
            if n_boot:
                boot = _half_sample_ari(labels0, labels1, rng, n_boot)
                boot_by_level[level_idx].append(boot)
                row["ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            detail.append(row)

    summary = {}
    for level_idx, aris in per_level.items():
        mean, std, ci_t = mean_ci95(aris)
        summary[level_idx] = {
            "mean_ari": mean, "std_ari": std, "values": aris, "n_transitions": len(aris),
            "ci95_t_over_transitions": ci_t,
        }
        if n_boot and boot_by_level[level_idx]:
            pooled = np.mean(np.vstack(boot_by_level[level_idx]), axis=0)
            summary[level_idx]["ci95"] = [float(np.percentile(pooled, 2.5)),
                                          float(np.percentile(pooled, 97.5))]
            summary[level_idx]["ci95_method"] = f"half-sampling of shared nodes within each transition, {n_boot} resamples"
        else:
            summary[level_idx]["ci95"] = ci_t
            summary[level_idx]["ci95_method"] = "t-interval over transitions"
    return {"by_level": summary, "detail": detail}
