"""How independent is the TF-IDF coherence check of the MPNet embedding
that drives clustering? Both read the same surface_form text, so this
measures how much they agree, per snapshot: the share of MPNet k-NN pairs
that share at least one TF-IDF term against the same share for random
pairs, the mean TF-IDF cosine of each, and the rank correlation of the two
similarities over random pairs. See DESIGN_NOTES.md section 5.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, SNAPSHOT_CUTOFFS, DATA_PATH  # noqa: E402
from tkh.embeddings import semantic_knn_graph  # noqa: E402
from tkh.pipeline import embed_concepts, KNN_K  # noqa: E402
from tkh.eval.coherence import fit_tfidf  # noqa: E402

N_RANDOM_PAIRS = 20000
OUT_PATH = ROOT / "outputs" / "signal_overlap.json"


def _tfidf_cos(X, rows, cols):
    return np.asarray(X[rows].multiply(X[cols]).sum(axis=1)).ravel()


def main():
    t0 = time.time()
    snapshots = build_all_snapshots(load_tkh(DATA_PATH), SNAPSHOT_CUTOFFS)
    cache, out = {}, {}
    for year in sorted(snapshots):
        snap = snapshots[year]
        ids, emb = embed_concepts(snap, cache)
        X, tf_ids, _ = fit_tfidf(snap)
        assert tf_ids == list(ids)

        knn = sp.triu(semantic_knn_graph(emb, k=KNN_K), k=1).tocoo()
        rng = np.random.default_rng(0)
        r = rng.integers(0, len(ids), N_RANDOM_PAIRS)
        c = rng.integers(0, len(ids), N_RANDOM_PAIRS)
        keep = r != c
        r, c = r[keep], c[keep]

        knn_tf = _tfidf_cos(X, knn.row, knn.col)
        rand_tf = _tfidf_cos(X, r, c)
        rand_mp = (emb[r] * emb[c]).sum(axis=1)
        out[year] = {
            "n_knn_pairs": int(knn.nnz), "n_random_pairs": int(len(r)),
            "knn_pairs_sharing_a_term": float((knn_tf > 0).mean()),
            "random_pairs_sharing_a_term": float((rand_tf > 0).mean()),
            "knn_mean_tfidf_cosine": float(knn_tf.mean()),
            "random_mean_tfidf_cosine": float(rand_tf.mean()),
            "spearman_mpnet_vs_tfidf_random_pairs": float(spearmanr(rand_mp, rand_tf).correlation),
        }
        o = out[year]
        print(f"[{time.time()-t0:5.1f}s] {year}: kNN pairs sharing a term {o['knn_pairs_sharing_a_term']:.2f} "
              f"vs random {o['random_pairs_sharing_a_term']:.2f}; mean TF-IDF cos {o['knn_mean_tfidf_cosine']:.3f} "
              f"vs {o['random_mean_tfidf_cosine']:.3f}; spearman {o['spearman_mpnet_vs_tfidf_random_pairs']:.2f}")

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
