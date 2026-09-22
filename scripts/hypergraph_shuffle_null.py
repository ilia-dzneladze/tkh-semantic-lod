"""A second, structurally-independent coherence null (DESIGN_NOTES.md section 13).

The shipped null (random-labels, same cluster sizes) doesn't control for
hypergraph structure at all. This one does: shuffle the 2026 snapshot's
hypergraph via bipartite double-edge-swaps on the (node, hyperedge)
incidence structure, preserving each concept node's hyperedge-degree and
each qualifying hyperedge's concept-arity exactly (the only two things
build_structural_affinity actually reads), rebuild structural affinity on
the shuffled hypergraph, cluster with the SAME real semantic affinity, and
score THAT clustering's coherence with the exact same functions the real
number already uses. If the real clustering still looks far more coherent
than clusterings built from degree/arity-matched random hypergraphs, that's
stronger evidence than the random-labels null alone. Writes
outputs/hypergraph_shuffle_null.json and copies the result into
outputs/metrics.json under "coherence_hypergraph_shuffle_null".
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot  # noqa: E402
from tkh.hypergraph import build_structural_affinity  # noqa: E402
from tkh.embeddings import encode_semantic, semantic_knn_graph  # noqa: E402
from tkh.cluster import combine_affinities, sparse_upgma, cut_to_k_clusters  # noqa: E402
from tkh.eval.coherence import fit_tfidf, cluster_coherence, weighted_mean_coherence  # noqa: E402

DATA_PATH = ROOT / "data" / "tkh_collection10.json"
OUT_PATH = ROOT / "outputs" / "hypergraph_shuffle_null.json"
ALPHA = 0.3
LEVEL_TARGETS = [12, 50, 200]
N_TRIALS = 20
SWAP_MULTIPLIER = 10

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def qualifying_edges(snap):
    idx = set(snap.concept_ids)
    return [[m for m in e.get("members", []) if m in idx]
            for e in snap.hyperedges
            if len([m for m in e.get("members", []) if m in idx]) >= 2]


def shuffle_bipartite(edges, seed, swap_multiplier=SWAP_MULTIPLIER):
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
    deg = {}
    for e in edges:
        for nid in e:
            deg[nid] = deg.get(nid, 0) + 1
    return sorted(deg.values())


def assert_invariant_preserved(orig_edges, shuf_edges):
    assert sorted(len(e) for e in orig_edges) == sorted(len(e) for e in shuf_edges), \
        "arity sequence changed"
    assert degree_sequence(orig_edges) == degree_sequence(shuf_edges), \
        "degree sequence changed"
    for e in shuf_edges:
        assert len(set(e)) == len(e), "duplicate member within one shuffled edge"


def main():
    data = load_tkh(DATA_PATH)
    snap = build_snapshot(data, 2026)
    real_hierarchy = json.loads((ROOT / "outputs" / "snapshots" / "2026" / "hierarchy.json")
                                 .read_text(encoding="utf-8"))

    orig_edges = qualifying_edges(snap)
    log(f"{len(orig_edges)} qualifying edges, {sum(len(e) for e in orig_edges)} incidence pairs")

    ids = sorted(snap.concept_ids)
    texts = [snap.nodes[nid]["surface_form"] or "" for nid in ids]
    emb = encode_semantic(texts, show_progress_bar=True)
    embedding_cache = dict(zip(ids, emb))
    emb_matrix = np.stack([embedding_cache[nid] for nid in ids])
    A_sem = semantic_knn_graph(emb_matrix, k=15)
    log("semantic affinity built (reused across every trial, unaffected by hypergraph shuffle)")

    X, tfidf_ids, id_to_row = fit_tfidf(snap)

    # real clustering's observed coherence, via the exact same functions
    real_observed = {}
    for level in range(len(LEVEL_TARGETS)):
        coh = cluster_coherence(real_hierarchy, level, X, id_to_row)
        real_observed[level] = weighted_mean_coherence(coh)
    log(f"real observed coherence: {real_observed}")

    null_values = {level: [] for level in range(len(LEVEL_TARGETS))}
    n_forced_by_trial = []
    swap_success_rates = []

    for trial in range(N_TRIALS):
        shuf_edges, n_success, n_attempts = shuffle_bipartite(orig_edges, seed=2000 + trial)
        assert_invariant_preserved(orig_edges, shuf_edges)
        swap_success_rates.append(n_success / n_attempts)

        # rebuild structural affinity directly from the shuffled edge list,
        # same weight formula as build_structural_affinity (1/(arity-1) per pair)
        idx = {nid: i for i, nid in enumerate(ids)}
        n = len(ids)
        import scipy.sparse as sp
        from collections import defaultdict
        pair_weight = defaultdict(float)
        for e in shuf_edges:
            arity = len(e)
            share = 1.0 / (arity - 1)
            for i in range(arity):
                for j in range(i + 1, arity):
                    a, b = idx[e[i]], idx[e[j]]
                    if a > b:
                        a, b = b, a
                    pair_weight[(a, b)] += share
        rows, cols, vals = [], [], []
        for (a, b), w in pair_weight.items():
            rows += [a, b]; cols += [b, a]; vals += [w, w]
        A_struct_shuf = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

        A_combined = combine_affinities(A_struct_shuf, A_sem, alpha=ALPHA)
        Z, forced = sparse_upgma(A_combined, n)
        n_forced_by_trial.append(int(forced.sum()))

        for level, k in enumerate(LEVEL_TARGETS):
            labels = cut_to_k_clusters(Z, n, k)
            from collections import defaultdict as dd
            clusters = dd(list)
            for nid, lab in zip(ids, labels):
                clusters[int(lab)].append(nid)
            hier = {"super_nodes": [{"id": str(lab), "level": level, "member_ids": members}
                                     for lab, members in clusters.items()]}
            coh = cluster_coherence(hier, level, X, id_to_row)
            val = weighted_mean_coherence(coh)
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
    log(f"wrote {OUT_PATH}")

    metrics_path = ROOT / "outputs" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["coherence_hypergraph_shuffle_null"] = {"year": 2026, "alpha": ALPHA, **out}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"updated {metrics_path}")


if __name__ == "__main__":
    main()
