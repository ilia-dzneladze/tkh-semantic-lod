"""T6 coherence, measured with a signal the clustering never used.

Clusters are built from MPNet embeddings and hyperedges. Coherence here is
TF-IDF similarity among a cluster's members, compared with the same
number for random clusters of the same sizes. DESIGN_NOTES.md section 13.
"""
import numpy as np
from tkh.embeddings import encode_lexical_tfidf


def fit_tfidf(snap):
    """(X, ids, id_to_row): TF-IDF rows for the concept nodes, sorted by id."""
    ids = sorted(snap.concept_ids)
    X = encode_lexical_tfidf([snap.nodes[nid]["surface_form"] or "" for nid in ids])
    return X, ids, {nid: i for i, nid in enumerate(ids)}


def labels_as_hierarchy(labels, ids, level):
    """A minimal hierarchy dict for one level of labels, so the coherence
    functions can score a clustering that was never written out."""
    clusters = {}
    for nid, lab in zip(ids, labels):
        clusters.setdefault(int(lab), []).append(nid)
    return {"super_nodes": [{"id": str(lab), "level": level, "member_ids": members}
                            for lab, members in clusters.items()]}


def _mean_offdiag_similarity(X, rows):
    """Mean cosine similarity over all pairs of distinct rows (None below 2)."""
    n = len(rows)
    if n < 2:
        return None
    sub = X[rows]
    S = sub @ sub.T
    return float((S.sum() - S.diagonal().sum()) / (n * (n - 1)))


def _member_rows(hierarchy, level, id_to_row):
    """{super-node id: TF-IDF rows of its members} at `level`."""
    return {sn["id"]: [id_to_row[nid] for nid in sn["member_ids"] if nid in id_to_row]
            for sn in hierarchy["super_nodes"] if sn["level"] == level}


def cluster_coherence(hierarchy, level, X, id_to_row):
    """{super-node id: (coherence, n members scored)} at `level`, for
    super-nodes with at least 2 members in the TF-IDF index."""
    out = {}
    for sid, rows in _member_rows(hierarchy, level, id_to_row).items():
        c = _mean_offdiag_similarity(X, rows)
        if c is not None:
            out[sid] = (c, len(rows))
    return out


def weighted_mean_coherence(coherence_by_id):
    """Mean coherence weighted by cluster size (None if nothing scored)."""
    if not coherence_by_id:
        return None
    total_w = sum(w for _, w in coherence_by_id.values())
    return sum(c * w for c, w in coherence_by_id.values()) / total_w


def random_labels_null(X, member_rows_by_cluster, n_trials=30, rng=None):
    """Weighted mean coherence of n_trials random clusterings with the
    same cluster sizes."""
    rng = rng or np.random.default_rng(0)
    all_rows = [r for rows in member_rows_by_cluster for r in rows]
    sizes = [len(rows) for rows in member_rows_by_cluster]
    null_values = []
    for _ in range(n_trials):
        perm = rng.permutation(all_rows)
        trial_coherence = {}
        i = 0
        for k, size in enumerate(sizes):
            chunk = perm[i:i + size].tolist()
            i += size
            c = _mean_offdiag_similarity(X, chunk)
            if c is not None:
                trial_coherence[k] = (c, size)
        null_values.append(weighted_mean_coherence(trial_coherence))
    return [v for v in null_values if v is not None]


def coherence_vs_null(hierarchy, level, X, id_to_row, n_trials=30, seed=0):
    coh = cluster_coherence(hierarchy, level, X, id_to_row)
    observed = weighted_mean_coherence(coh)
    member_rows_by_cluster = [rows for rows in _member_rows(hierarchy, level, id_to_row).values()
                              if len(rows) >= 2]

    null_vals = random_labels_null(X, member_rows_by_cluster, n_trials=n_trials,
                                    rng=np.random.default_rng(seed))
    null_mean = float(np.mean(null_vals)) if null_vals else None
    null_std = float(np.std(null_vals)) if null_vals else None
    z = ((observed - null_mean) / null_std
         if observed is not None and null_std else None)
    percentile_at_or_above = (
        sum(1 for v in null_vals if v >= observed) / len(null_vals)
        if observed is not None and null_vals else None)

    return {
        "level": level,
        "n_clusters_scored": len(coh),
        "observed_coherence": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_n_trials": len(null_vals),
        "z_score": z,
        "null_fraction_at_or_above_observed": percentile_at_or_above,
        "per_cluster": {k: v[0] for k, v in coh.items()},
    }


def heldout_edge_cohesion(labels, ids, heldout_edges):
    """How often members of hyperedges the clustering never saw land in the
    same cluster, against a random partition with the same cluster sizes.

    labels: cluster label per id. heldout_edges: lists of member ids; only
    members in ids count, and edges with fewer than 2 of them are skipped.
    Each edge contributes the fraction of its member pairs that share a
    cluster, and edges are averaged with equal weight. Under a random
    partition any pair shares a cluster with probability
    sum n_c(n_c-1) / (N(N-1)), which is the expected value. Returns
    (observed, expected, lift = observed / expected).
    See DESIGN_NOTES.md section 13."""
    lab = dict(zip(ids, labels))
    fractions = []
    for e in heldout_edges:
        members = [lab[m] for m in e if m in lab]
        k = len(members)
        if k < 2:
            continue
        same = sum(1 for i in range(k) for j in range(i + 1, k) if members[i] == members[j])
        fractions.append(same / (k * (k - 1) / 2))
    sizes = np.bincount(np.asarray(labels))
    n = len(labels)
    expected = float((sizes * (sizes - 1)).sum() / (n * (n - 1)))
    observed = float(np.mean(fractions)) if fractions else float("nan")
    return observed, expected, observed / expected if expected else float("nan")
