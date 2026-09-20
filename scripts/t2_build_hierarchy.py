"""T2 smoke test: build a laminar hierarchy for one snapshot end-to-end."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot  # noqa: E402
from tkh.hypergraph import build_structural_affinity  # noqa: E402
from tkh.embeddings import encode_semantic, semantic_knn_graph  # noqa: E402
from tkh.cluster import combine_affinities, sparse_upgma, cut_to_k_clusters  # noqa: E402

DATA_PATH = ROOT / "data" / "tkh_collection10.json"
LEVEL_TARGETS = [12, 50, 200]  # N_0..N_2, coarsest to finest super-node level
ALPHA = 0.5


def main(cutoff_year=2026):
    t0 = time.time()
    data = load_tkh(DATA_PATH)
    snap = build_snapshot(data, cutoff_year)
    print(f"[{time.time()-t0:.1f}s] snapshot built: "
          f"{len(snap.concept_ids)} concept nodes, {len(snap.hyperedges)} hyperedges")

    A_struct, ids, struct_stats = build_structural_affinity(snap, weighted=True)
    print(f"[{time.time()-t0:.1f}s] structural affinity: {struct_stats}")

    texts = [snap.nodes[nid]["surface_form"] or "" for nid in ids]
    emb = encode_semantic(texts, show_progress_bar=True)
    print(f"[{time.time()-t0:.1f}s] embeddings: {emb.shape}")

    A_sem = semantic_knn_graph(emb, k=15)
    print(f"[{time.time()-t0:.1f}s] semantic kNN graph: nnz={A_sem.nnz}")

    A_combined = combine_affinities(A_struct, A_sem, alpha=ALPHA)
    print(f"[{time.time()-t0:.1f}s] combined graph: nnz={A_combined.nnz}")

    n = len(ids)
    Z, forced = sparse_upgma(A_combined, n)
    print(f"[{time.time()-t0:.1f}s] UPGMA done: {len(Z)} merges, "
          f"{forced.sum()} forced ({100*forced.mean():.1f}%)")

    for k in LEVEL_TARGETS:
        labels = cut_to_k_clusters(Z, n, k)
        n_actual = len(set(labels))
        sizes = [int((labels == c).sum()) for c in set(labels)]
        print(f"  level target={k}: actual={n_actual} clusters, "
              f"size range [{min(sizes)}, {max(sizes)}], "
              f"median={sorted(sizes)[len(sizes)//2]}")

    print(f"[{time.time()-t0:.1f}s] done")


if __name__ == "__main__":
    main()
