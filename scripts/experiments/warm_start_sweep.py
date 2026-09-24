"""Warm-start check (DESIGN_NOTES.md sections 10 and 15): does pulling each
snapshot's clustering toward the previous snapshot's level-2 partition
buy real cross-snapshot stability, and what does it cost in coherence?

A = (1 - gamma) * A_task + gamma * A_prior, where A_task is the shipped
alpha=0.3 blend and A_prior(i, j) = 1 if i and j shared a level-2 cluster
in the previous snapshot (0 for nodes that weren't there). gamma=0 must
reproduce the shipped hierarchy. Years run in order, since each depends
on the previous year's (possibly warm-started) result. Writes
outputs/warm_start_sweep.json.

Usage:
    python scripts/experiments/warm_start_sweep.py      # full gamma grid
    python scripts/experiments/warm_start_sweep.py 0    # smoke test only
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, SNAPSHOT_CUTOFFS, DATA_PATH  # noqa: E402
from tkh.pipeline import ALPHA, LEVEL_TARGETS, KNN_K  # noqa: E402
from tkh.hypergraph import build_structural_affinity  # noqa: E402
from tkh.embeddings import encode_semantic, semantic_knn_graph, encode_lexical_tfidf  # noqa: E402
from tkh.cluster import combine_affinities, sparse_upgma, cut_to_k_clusters  # noqa: E402
from tkh.eval.coherence import coherence_vs_null  # noqa: E402
from tkh.eval.stability import cross_snapshot_stability  # noqa: E402

OUT_PATH = ROOT / "outputs" / "warm_start_sweep.json"
PRIOR_LEVEL = 2
DEFAULT_GAMMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def lightweight_hierarchy(ids, labels_by_level):
    super_nodes = []
    for level, labels in labels_by_level.items():
        clusters = defaultdict(list)
        for nid, lab in zip(ids, labels):
            clusters[int(lab)].append(nid)
        for lab, members in clusters.items():
            super_nodes.append({
                "id": f"{level}_{lab}", "level": level, "member_ids": sorted(members),
            })
    return {"super_nodes": super_nodes}


def build_prior_affinity(ids, prev_clusters, n):
    """prev_clusters: {label: frozenset(prev_node_ids)}, from the previous
    snapshot's level-2 partition. Symmetric 0/1 sparse matrix over the
    CURRENT snapshot's ids, nonzero only between pairs that were together
    before AND both still exist now."""
    id_to_row = {nid: i for i, nid in enumerate(ids)}
    rows, cols = [], []
    for members in prev_clusters.values():
        idxs = [id_to_row[nid] for nid in members if nid in id_to_row]
        m = len(idxs)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    rows.append(idxs[a])
                    cols.append(idxs[b])
    if not rows:
        return sp.csr_matrix((n, n))
    data = np.ones(len(rows), dtype=float)
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def main():
    gammas = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else DEFAULT_GAMMAS

    data = load_tkh(DATA_PATH)
    snapshots = build_all_snapshots(data, cutoffs=SNAPSHOT_CUTOFFS)
    years = sorted(snapshots)

    # one-time, gamma-independent precompute per snapshot (same as alpha_sweep.py)
    per_year = {}
    embedding_cache = {}
    for year in years:
        snap = snapshots[year]
        A_struct, ids = build_structural_affinity(snap)
        missing = [nid for nid in ids if nid not in embedding_cache]
        if missing:
            texts = [snap.nodes[nid]["surface_form"] or "" for nid in missing]
            vecs = encode_semantic(texts, show_progress_bar=True)
            for nid, v in zip(missing, vecs):
                embedding_cache[nid] = v
        emb = np.stack([embedding_cache[nid] for nid in ids])
        A_sem = semantic_knn_graph(emb, k=KNN_K)
        A_task = combine_affinities(A_struct, A_sem, alpha=ALPHA)
        texts_ordered = [snap.nodes[nid]["surface_form"] or "" for nid in ids]
        X_tfidf = encode_lexical_tfidf(texts_ordered)
        id_to_row = {nid: i for i, nid in enumerate(ids)}
        per_year[year] = {"ids": ids, "A_task": A_task, "X_tfidf": X_tfidf, "id_to_row": id_to_row}
        log(f"precompute done for {year} (n={len(ids)})")

    results = {}
    for gamma in gammas:
        g_t0 = time.time()
        hier_by_year = {}
        coherence_by_year = {}
        prev_level2_clusters = None

        for year in years:
            d = per_year[year]
            ids = d["ids"]
            n = len(ids)
            if gamma > 0 and prev_level2_clusters is not None:
                A_prior = build_prior_affinity(ids, prev_level2_clusters, n)
                A_final = ((1 - gamma) * d["A_task"] + gamma * A_prior).tocsr()
                A_final.eliminate_zeros()
            else:
                A_final = d["A_task"]

            Z, forced = sparse_upgma(A_final, n)
            labels_by_level = {lvl: cut_to_k_clusters(Z, n, k) for lvl, k in enumerate(LEVEL_TARGETS)}
            hier = lightweight_hierarchy(ids, labels_by_level)
            hier_by_year[year] = hier
            coherence_by_year[year] = {
                lvl: coherence_vs_null(hier, lvl, d["X_tfidf"], d["id_to_row"], n_trials=30, seed=0)
                for lvl in range(len(LEVEL_TARGETS))
            }
            prev_level2_clusters = {sn["id"]: frozenset(sn["member_ids"])
                                     for sn in hier["super_nodes"] if sn["level"] == PRIOR_LEVEL}

        cross = cross_snapshot_stability(hier_by_year, len(LEVEL_TARGETS))
        coherence_summary = {
            year: {lvl: {k: v for k, v in row.items() if k != "per_cluster"}
                   for lvl, row in by_lvl.items()}
            for year, by_lvl in coherence_by_year.items()
        }
        results[str(gamma)] = {"gamma": gamma, "cross_snapshot": cross, "coherence_by_year": coherence_summary}
        log(f"gamma={gamma:.2f} done in {time.time()-g_t0:.1f}s, "
            f"cross-snapshot mean ARI by level: "
            f"{[round(cross['by_level'][l]['mean_ari'], 4) for l in range(len(LEVEL_TARGETS))]}")
        OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    log(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
