"""How much of the combined affinity actually comes from the structural
term, per snapshot, at a few values of alpha.

alpha is the nominal weight in alpha*structural + (1-alpha)*semantic, but
the two normalized graphs sit at very different value ranges, so alpha is
not the share of the signal structure ends up carrying. This script
measures that share. See DESIGN_NOTES.md section 17.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, SNAPSHOT_CUTOFFS, DATA_PATH  # noqa: E402
from tkh.hypergraph import build_structural_affinity  # noqa: E402
from tkh.embeddings import semantic_knn_graph  # noqa: E402
from tkh.cluster import combine_affinities  # noqa: E402
from tkh.pipeline import embed_concepts, KNN_K  # noqa: E402

ALPHAS = [0.1, 0.3, 0.5, 0.7]
OUT_PATH = ROOT / "outputs" / "affinity_mass_share.json"


def main():
    t0 = time.time()
    data = load_tkh(DATA_PATH)
    snapshots = build_all_snapshots(data, cutoffs=SNAPSHOT_CUTOFFS)
    cache = {}
    out = {}

    for year in sorted(snapshots):
        snap = snapshots[year]
        ids, emb = embed_concepts(snap, cache)
        A_struct, struct_ids = build_structural_affinity(snap)
        assert struct_ids == list(ids)
        A_sem = semantic_knn_graph(emb, k=KNN_K)

        # alpha=1 / alpha=0 return exactly the two normalized graphs, so the
        # totals below are the normalized masses without reaching inside
        # combine_affinities for them
        norm_struct = combine_affinities(A_struct, A_sem, alpha=1.0)
        norm_sem = combine_affinities(A_struct, A_sem, alpha=0.0)
        struct_mass, sem_mass = float(norm_struct.sum()), float(norm_sem.sum())

        struct_pairs = set(zip(*norm_struct.nonzero()))
        sem_pairs = set(zip(*norm_sem.nonzero()))

        out[year] = {
            "n_concept_nodes": len(ids),
            "n_structural_pairs": len(struct_pairs) // 2,
            "n_semantic_pairs": len(sem_pairs) // 2,
            "n_structure_only_pairs": len(struct_pairs - sem_pairs) // 2,
            "n_both_pairs": len(struct_pairs & sem_pairs) // 2,
            "median_normalized_structural_weight": float(np.median(norm_struct.data)),
            "median_normalized_semantic_weight": float(np.median(norm_sem.data)),
            "structural_share_of_affinity_mass": {
                str(a): (a * struct_mass) / (a * struct_mass + (1 - a) * sem_mass)
                for a in ALPHAS
            },
        }
        r = out[year]
        print(f"[{time.time()-t0:5.1f}s] {year}: structural pairs {r['n_structural_pairs']}, "
              f"semantic pairs {r['n_semantic_pairs']}, structure-only {r['n_structure_only_pairs']}; "
              f"median weight struct {r['median_normalized_structural_weight']:.3f} vs "
              f"sem {r['median_normalized_semantic_weight']:.3f}; "
              f"mass share at alpha=0.3 = {r['structural_share_of_affinity_mass']['0.3']:.3f}")

    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
