"""Alpha ablation sweep (FIXES.md, iteration 1).

Sweeps alpha across the structural/semantic split in 5% steps: 0.05
through 0.95. Structural affinity, semantic embeddings, and the semantic
k-NN graph don't depend on alpha at all, so each is built once per
snapshot and reused for every value; only combine -> sparse_upgma -> cut
actually reruns per alpha.

Scores each value with coherence-vs-null and both stability measures,
reusing tkh.eval.coherence / tkh.eval.stability exactly as t6_evaluate.py
does, against a "lightweight" hierarchy dict that only carries the fields
those functions actually read (level, id, member_ids) -- T4 collapse, T3
persistent-id tracking, and T5 labels are all skipped here on purpose.
Labels were hand-written once, for the alpha=0.5 hierarchy; faithfulness
and the extrinsic eval both need gloss text, so this sweep doesn't touch
either of them, see FIXES.md for why.

Usage:
    python scripts\\alpha_sweep.py                  # full 19-value sweep
    python scripts\\alpha_sweep.py 0.5               # just alpha=0.5, for a smoke test
    python scripts\\alpha_sweep.py 0.05,0.5,0.95     # a specific subset
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, SNAPSHOT_CUTOFFS  # noqa: E402
from tkh.hypergraph import build_structural_affinity  # noqa: E402
from tkh.embeddings import encode_semantic, semantic_knn_graph, encode_lexical_tfidf  # noqa: E402
from tkh.cluster import combine_affinities, sparse_upgma, cut_to_k_clusters  # noqa: E402
from tkh.eval.coherence import coherence_vs_null  # noqa: E402
from tkh.eval.stability import perturbation_stability, cross_snapshot_stability  # noqa: E402

DATA_PATH = ROOT / "data" / "tkh_collection10.json"
OUT_PATH = ROOT / "outputs" / "alpha_sweep.json"
LEVEL_TARGETS = [12, 50, 200]
DEFAULT_ALPHAS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def lightweight_hierarchy(ids, labels_by_level):
    """super_nodes carrying only level/id/member_ids -- everything
    coherence_vs_null, labels_from_hierarchy and cross_snapshot_stability
    actually read. No T4 collapse, no persistent ids, no labels."""
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
        A_struct, ids, struct_stats = build_structural_affinity(snap, weighted=True)
        missing = [nid for nid in ids if nid not in embedding_cache]
        if missing:
            texts = [snap.nodes[nid]["surface_form"] or "" for nid in missing]
            vecs = encode_semantic(texts, show_progress_bar=True)
            for nid, v in zip(missing, vecs):
                embedding_cache[nid] = v
        emb = np.stack([embedding_cache[nid] for nid in ids])
        A_sem = semantic_knn_graph(emb, k=15)
        texts_ordered = [snap.nodes[nid]["surface_form"] or "" for nid in ids]
        X_tfidf = encode_lexical_tfidf(texts_ordered)
        id_to_row = {nid: i for i, nid in enumerate(ids)}
        per_year[year] = {
            "snap": snap, "A_struct": A_struct, "A_sem": A_sem, "ids": ids,
            "X_tfidf": X_tfidf, "id_to_row": id_to_row,
        }
        log(f"precompute done for {year} (n={len(ids)}, struct_nnz_pairs={struct_stats['n_nonzero_pairs']})")

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
