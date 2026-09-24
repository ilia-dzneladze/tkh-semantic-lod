"""Alpha sweep (DESIGN_NOTES.md sections 7 and 15): cluster every snapshot at
alpha = 0.05, 0.10, ..., 0.95 and score each with coherence against the
null and both stability measures. The structural and semantic graphs
don't depend on alpha, so they're built once per snapshot. Labels,
faithfulness and the extrinsic eval are left out, since they need glosses
written for one specific hierarchy.

Usage:
    python scripts/experiments/alpha_sweep.py                  # all 19 values
    python scripts/experiments/alpha_sweep.py 0.5              # one value, as a smoke test
    python scripts/experiments/alpha_sweep.py 0.05,0.5,0.95    # a subset
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, SNAPSHOT_CUTOFFS, DATA_PATH  # noqa: E402
from tkh.pipeline import LEVEL_TARGETS, KNN_K  # noqa: E402
from tkh.hypergraph import build_structural_affinity  # noqa: E402
from tkh.embeddings import encode_semantic, semantic_knn_graph, encode_lexical_tfidf  # noqa: E402
from tkh.cluster import combine_affinities, sparse_upgma, cut_to_k_clusters  # noqa: E402
from tkh.eval.coherence import coherence_vs_null  # noqa: E402
from tkh.eval.stability import perturbation_stability, cross_snapshot_stability  # noqa: E402

OUT_PATH = ROOT / "outputs" / "alpha_sweep.json"
DEFAULT_ALPHAS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def lightweight_hierarchy(ids, labels_by_level):
    """A hierarchy dict with only level, id and member_ids, which is all
    the coherence and stability functions read."""
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


def main():
    if len(sys.argv) > 1:
        alphas = [float(x) for x in sys.argv[1].split(",")]
    else:
        alphas = DEFAULT_ALPHAS

    data = load_tkh(DATA_PATH)
    snapshots = build_all_snapshots(data, cutoffs=SNAPSHOT_CUTOFFS)
    years = sorted(snapshots)

    # one-time, alpha-independent precompute per snapshot
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
        texts_ordered = [snap.nodes[nid]["surface_form"] or "" for nid in ids]
        X_tfidf = encode_lexical_tfidf(texts_ordered)
        id_to_row = {nid: i for i, nid in enumerate(ids)}
        per_year[year] = {
            "snap": snap, "A_struct": A_struct, "A_sem": A_sem, "ids": ids,
            "X_tfidf": X_tfidf, "id_to_row": id_to_row,
        }
        log(f"precompute done for {year} (n={len(ids)}, struct_nnz_pairs={A_struct.nnz // 2})")

    results = {}
    for alpha in alphas:
        a_t0 = time.time()
        hier_by_year = {}
        coherence_by_year = {}
        for year in years:
            d = per_year[year]
            A = combine_affinities(d["A_struct"], d["A_sem"], alpha=alpha)
            n = len(d["ids"])
            Z, forced = sparse_upgma(A, n)
            labels_by_level = {lvl: cut_to_k_clusters(Z, n, k) for lvl, k in enumerate(LEVEL_TARGETS)}
            hier = lightweight_hierarchy(d["ids"], labels_by_level)
            hier_by_year[year] = hier
            coherence_by_year[year] = {
                lvl: coherence_vs_null(hier, lvl, d["X_tfidf"], d["id_to_row"], n_trials=30, seed=0)
                for lvl in range(len(LEVEL_TARGETS))
            }

        cross = cross_snapshot_stability(hier_by_year, len(LEVEL_TARGETS))

        year2026 = years[-1]
        d2026 = per_year[year2026]
        pert = perturbation_stability(
            d2026["snap"], embedding_cache, LEVEL_TARGETS, alpha,
            original_hierarchy=hier_by_year[year2026], n_seeds=5, remove_frac=0.10)

        # drop the bulky per_cluster / per_seed detail before storing, keep summaries only
        coherence_summary = {
            year: {lvl: {k: v for k, v in row.items() if k != "per_cluster"}
                   for lvl, row in by_lvl.items()}
            for year, by_lvl in coherence_by_year.items()
        }

        results[str(alpha)] = {
            "alpha": alpha,
            "coherence_by_year": coherence_summary,
            "cross_snapshot": cross,
            "perturbation": pert,
        }
        log(f"alpha={alpha:.2f} done in {time.time()-a_t0:.1f}s")
        OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")  # write after every alpha, cheap insurance

    log(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
