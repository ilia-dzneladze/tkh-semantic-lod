"""T6 coherence, with the circularity trap neutralized.

Clustering is driven by MPNet embeddings + hypergraph structure. Coherence
here is measured with TF-IDF instead, a lexical signal never used to build
the clusters, then compared against a random-labels null model so the
number means something relative to chance. See DESIGN_NOTES.md section 13.
"""
import numpy as np
from tkh.embeddings import encode_lexical_tfidf


def fit_tfidf(snap):
    ids = sorted(snap.concept_ids)
    texts = [snap.nodes[nid]["surface_form"] or "" for nid in ids]
    X = encode_lexical_tfidf(texts)  # sparse, L2-normalized rows
    id_to_row = {nid: i for i, nid in enumerate(ids)}
    return X, ids, id_to_row


def _mean_offdiag_similarity(X, rows):
    n = len(rows)
    if n < 2:
        return None
    sub = X[rows]
    S = (sub @ sub.T)
    total = S.sum()
    diag = S.diagonal().sum()
    denom = n * (n - 1)
    return float((total - diag) / denom) if denom else None


def cluster_coherence(hierarchy, level, X, id_to_row, min_size=2):
    """{super_node_id: coherence} for every super-node at `level` with at
    least min_size members that are in the TF-IDF index (raw ids only,
    context nodes aren't indexed and are skipped)."""
    out = {}
    for sn in hierarchy["super_nodes"]:
        if sn["level"] != level:
            continue
        rows = [id_to_row[nid] for nid in sn["member_ids"] if nid in id_to_row]
        c = _mean_offdiag_similarity(X, rows)
        if c is not None:
            out[sn["id"]] = (c, len(rows))
    return out


def weighted_mean_coherence(coherence_by_id):
    if not coherence_by_id:
        return None
    total_w = sum(w for _, w in coherence_by_id.values())
    if total_w == 0:
        return None
    return sum(c * w for c, w in coherence_by_id.values()) / total_w


def random_labels_null(X, member_rows_by_cluster, n_trials=30, rng=None):
    """Same cluster SIZES, node identities reshuffled at random. Returns
    a list of n_trials weighted-mean-coherence values under this null."""
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
    member_rows_by_cluster = []
    for sn in hierarchy["super_nodes"]:
        if sn["level"] != level:
            continue
        rows = [id_to_row[nid] for nid in sn["member_ids"] if nid in id_to_row]
        if len(rows) >= 2:
            member_rows_by_cluster.append(rows)

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
