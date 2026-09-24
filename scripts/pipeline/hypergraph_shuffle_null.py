"""A second coherence null that keeps the hypergraph's shape
(DESIGN_NOTES.md section 13).

The random-labels null ignores structure. This one shuffles the 2026
hypergraph by swapping members between hyperedges, which keeps every
concept node's degree and every hyperedge's size exactly (the only two
things the structural affinity depends on). Each shuffled hypergraph is
clustered with the real semantic graph and scored with the same TF-IDF
coherence as the real clustering. Writes outputs/hypergraph_shuffle_null.json
and the same result into metrics.json under
"coherence_hypergraph_shuffle_null".
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, load_hierarchy, update_metrics, DATA_PATH, OUTPUTS  # noqa: E402
from tkh.pipeline import ALPHA, LEVEL_TARGETS, KNN_K  # noqa: E402
from tkh.hypergraph import clique_expansion, concept_members  # noqa: E402
from tkh.embeddings import encode_semantic, semantic_knn_graph  # noqa: E402
from tkh.cluster import combine_affinities, sparse_upgma, cut_to_k_clusters  # noqa: E402
from tkh.eval.coherence import (  # noqa: E402
    fit_tfidf, cluster_coherence, weighted_mean_coherence, labels_as_hierarchy)

YEAR = 2026
OUT_PATH = OUTPUTS / "hypergraph_shuffle_null.json"
N_TRIALS = 20
SWAP_MULTIPLIER = 10

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def qualifying_edges(snap):
    """Each hyperedge's concept members, for edges with at least two."""
    edges = [concept_members(snap, e) for e in snap.hyperedges]
    return [e for e in edges if len(e) >= 2]


def shuffle_bipartite(edges, seed, swap_multiplier=SWAP_MULTIPLIER):
    """Swap members between random pairs of edges, swap_multiplier times
    per membership, skipping swaps that would repeat a node in an edge.
    Returns (shuffled edges, successful swaps, attempts)."""
    rng = np.random.default_rng(seed)
    edges = [list(e) for e in edges]
    edge_sets = [set(e) for e in edges]
    incidence = [(ei, pos) for ei, e in enumerate(edges) for pos in range(len(e))]
    n_incidence = len(incidence)
    n_attempts = swap_multiplier * n_incidence
    n_success = 0
    for _ in range(n_attempts):
        i1 = rng.integers(0, n_incidence)
        i2 = rng.integers(0, n_incidence)
        e1, p1 = incidence[i1]
        e2, p2 = incidence[i2]
        if e1 == e2:
            continue
        n1, n2 = edges[e1][p1], edges[e2][p2]
        if n1 == n2 or n1 in edge_sets[e2] or n2 in edge_sets[e1]:
            continue
        edges[e1][p1], edges[e2][p2] = n2, n1
        edge_sets[e1].discard(n1); edge_sets[e1].add(n2)
        edge_sets[e2].discard(n2); edge_sets[e2].add(n1)
        n_success += 1
    return edges, n_success, n_attempts


def degree_sequence(edges):
    return sorted(Counter(nid for e in edges for nid in e).values())


def assert_invariant_preserved(orig_edges, shuf_edges):
    assert sorted(len(e) for e in orig_edges) == sorted(len(e) for e in shuf_edges), \
        "arity sequence changed"
    assert degree_sequence(orig_edges) == degree_sequence(shuf_edges), \
        "degree sequence changed"
    for e in shuf_edges:
        assert len(set(e)) == len(e), "duplicate member within one shuffled edge"


def main():
    snap = build_snapshot(load_tkh(DATA_PATH), YEAR)
    real_hierarchy = load_hierarchy(YEAR)

    orig_edges = qualifying_edges(snap)
    log(f"{len(orig_edges)} qualifying edges, {sum(len(e) for e in orig_edges)} incidence pairs")

    ids = sorted(snap.concept_ids)
    idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)
    emb = encode_semantic([snap.nodes[nid]["surface_form"] or "" for nid in ids], show_progress_bar=True)
    A_sem = semantic_knn_graph(emb, k=KNN_K)  # the shuffle doesn't touch it
    X, _, id_to_row = fit_tfidf(snap)

    real_observed = {level: weighted_mean_coherence(cluster_coherence(real_hierarchy, level, X, id_to_row))
                     for level in range(len(LEVEL_TARGETS))}
    log(f"real observed coherence: {real_observed}")

    null_values = {level: [] for level in range(len(LEVEL_TARGETS))}
    n_forced_by_trial = []
    swap_success_rates = []

    for trial in range(N_TRIALS):
        shuf_edges, n_success, n_attempts = shuffle_bipartite(orig_edges, seed=2000 + trial)
        assert_invariant_preserved(orig_edges, shuf_edges)
        swap_success_rates.append(n_success / n_attempts)

        A_struct = clique_expansion([[idx[m] for m in e] for e in shuf_edges], n)
        Z, forced = sparse_upgma(combine_affinities(A_struct, A_sem, alpha=ALPHA), n)
        n_forced_by_trial.append(int(forced.sum()))

        for level, k in enumerate(LEVEL_TARGETS):
            hier = labels_as_hierarchy(cut_to_k_clusters(Z, n, k), ids, level)
            val = weighted_mean_coherence(cluster_coherence(hier, level, X, id_to_row))
            if val is not None:
                null_values[level].append(val)

        log(f"trial {trial} done (swap success rate {n_success/n_attempts:.2f}, "
            f"{int(forced.sum())} forced merges)")

    results = {}
    for level in range(len(LEVEL_TARGETS)):
        vals = null_values[level]
        null_mean = float(np.mean(vals))
        null_std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        z = (real_observed[level] - null_mean) / null_std if null_std else None
        results[level] = {
            "real_observed_coherence": real_observed[level],
            "shuffled_null_mean": null_mean, "shuffled_null_std": null_std,
            "shuffled_null_values": vals, "z_score": z,
            "n_trials": len(vals),
        }
        log(f"level {level}: real={real_observed[level]:.5f} "
            f"shuffled_null_mean={null_mean:.5f} z={z}")

    out = {
        "n_trials": N_TRIALS, "swap_multiplier": SWAP_MULTIPLIER,
        "mean_swap_success_rate": float(np.mean(swap_success_rates)),
        "n_forced_merges_by_trial": n_forced_by_trial,
        "by_level": results,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    update_metrics("coherence_hypergraph_shuffle_null", {"year": YEAR, "alpha": ALPHA, **out})
    log(f"wrote {OUT_PATH} and metrics.json")


if __name__ == "__main__":
    main()
