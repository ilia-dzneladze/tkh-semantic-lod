"""Level-0 size skew (DESIGN_NOTES.md section 15): does the largest level-0
cluster move more under perturbation than its size alone predicts? Same
perturbation as the shipped stability number (10% of edges, 5 seeds,
base seed 1000), but each original cluster is followed on its own by its
best Jaccard match instead of one whole-partition ARI. Writes
outputs/level0_skew_check.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, DATA_PATH  # noqa: E402
from tkh.pipeline import ALPHA, KNN_K  # noqa: E402
from tkh.hypergraph import build_structural_affinity  # noqa: E402
from tkh.embeddings import encode_semantic, semantic_knn_graph  # noqa: E402
from tkh.cluster import combine_affinities, sparse_upgma, cut_to_k_clusters  # noqa: E402
from tkh.eval.stability import perturb_snapshot  # noqa: E402

OUT_DIR = ROOT / "outputs"
LEVEL0_K = 12
N_SEEDS = 5
BASE_SEED = 1000
REMOVE_FRAC = 0.10


def best_jaccard_match(orig_members, new_clusters):
    orig_set = set(orig_members)
    best = 0.0
    for members in new_clusters.values():
        s = set(members)
        inter = len(orig_set & s)
        if inter == 0:
            continue
        union = len(orig_set | s)
        j = inter / union
        if j > best:
            best = j
    return best


def main():
    t0 = time.time()
    data = load_tkh(DATA_PATH)
    snap = build_snapshot(data, 2026)
    hierarchy = json.loads((OUT_DIR / "snapshots" / "2026" / "hierarchy.json").read_text(encoding="utf-8"))

    level0 = [sn for sn in hierarchy["super_nodes"] if sn["level"] == 0]
    print(f"[{time.time()-t0:.1f}s] {len(level0)} level-0 clusters, "
          f"sizes {sorted((sn['member_count'] for sn in level0), reverse=True)}")

    A_struct0, ids = build_structural_affinity(snap)
    texts = [snap.nodes[nid]["surface_form"] or "" for nid in ids]
    vecs = encode_semantic(texts, show_progress_bar=True)
    embedding_cache = dict(zip(ids, vecs))
    emb = np.stack([embedding_cache[nid] for nid in ids])
    A_sem = semantic_knn_graph(emb, k=KNN_K)
    print(f"[{time.time()-t0:.1f}s] embeddings + structural affinity done")

    jaccards_by_cluster = {sn["id"]: [] for sn in level0}

    for s in range(N_SEEDS):
        pert = perturb_snapshot(snap, remove_frac=REMOVE_FRAC, seed=BASE_SEED + s)
        A_struct, pert_ids = build_structural_affinity(pert)
        assert pert_ids == ids, "perturbation must not change the node set"

        A_combined = combine_affinities(A_struct, A_sem, alpha=ALPHA)
        Z, forced = sparse_upgma(A_combined, len(ids))
        labels = cut_to_k_clusters(Z, len(ids), LEVEL0_K)

        new_clusters = {}
        for nid, lab in zip(ids, labels):
            new_clusters.setdefault(int(lab), []).append(nid)

        for sn in level0:
            j = best_jaccard_match(sn["member_ids"], new_clusters)
            jaccards_by_cluster[sn["id"]].append(j)
        print(f"[{time.time()-t0:.1f}s] seed {s} done "
              f"({len(new_clusters)} perturbed clusters, {int(forced.sum())} forced merges)")

    rows = []
    for sn in level0:
        js = jaccards_by_cluster[sn["id"]]
        rows.append({
            "id": sn["id"], "label": sn["label"], "size": sn["member_count"],
            "mean_jaccard": float(np.mean(js)), "jaccards": js,
        })
    rows.sort(key=lambda r: -r["size"])

    # fit stability ~ size on the 11 non-largest clusters, check where the
    # largest actually falls against that trend
    largest = rows[0]
    others = rows[1:]
    sizes = np.array([r["size"] for r in others], dtype=float)
    stabs = np.array([r["mean_jaccard"] for r in others], dtype=float)
    slope, intercept = np.polyfit(sizes, stabs, 1)
    predicted_for_largest = slope * largest["size"] + intercept
    residuals = stabs - (slope * sizes + intercept)
    resid_std = float(np.std(residuals, ddof=1))
    actual_residual = largest["mean_jaccard"] - predicted_for_largest

    pearson_r = float(np.corrcoef([r["size"] for r in rows], [r["mean_jaccard"] for r in rows])[0, 1])

    print()
    print(f"{'id':12s} {'size':>5s} {'mean_jaccard':>13s}")
    for r in rows:
        print(f"{r['id']:12s} {r['size']:5d} {r['mean_jaccard']:13.4f}")
    print()
    print(f"pearson r (size vs. mean_jaccard, all 12): {pearson_r:.3f}")
    print(f"trend fit on the 11 non-largest: stability = {slope:.6f}*size + {intercept:.4f}")
    print(f"largest cluster ({largest['id']}, size={largest['size']}): "
          f"actual={largest['mean_jaccard']:.4f}, predicted={predicted_for_largest:.4f}, "
          f"residual={actual_residual:.4f} ({actual_residual/resid_std:.2f} std devs of the trend's own residuals)")

    out = {
        "alpha": ALPHA, "n_seeds": N_SEEDS, "base_seed": BASE_SEED, "remove_frac": REMOVE_FRAC,
        "clusters": rows, "pearson_r_size_vs_stability": pearson_r,
        "trend_fit_excl_largest": {"slope": slope, "intercept": intercept, "resid_std": resid_std},
        "largest_cluster": {
            "id": largest["id"], "size": largest["size"], "actual_mean_jaccard": largest["mean_jaccard"],
            "predicted_by_trend": predicted_for_largest, "residual": actual_residual,
            "residual_in_std_devs": actual_residual / resid_std if resid_std else None,
        },
    }
    (OUT_DIR / "level0_skew_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[{time.time()-t0:.1f}s] wrote {OUT_DIR / 'level0_skew_check.json'}")


if __name__ == "__main__":
    main()
