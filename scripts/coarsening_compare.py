"""Single-dendrogram vs multilevel (T4-driven) coarsening, on label-free
metrics only. Decision rule written before running: DESIGN_NOTES.md
section 15. Writes outputs/coarsening_compare.json; changes nothing in the
shipped pipeline.

Also checks that the refactored dendrogram path reproduces the shipped
hierarchy.json member sets exactly before comparing anything.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, SNAPSHOT_CUTOFFS  # noqa: E402
from tkh.pipeline import (  # noqa: E402
    run_all_snapshots, build_hierarchy_json, embed_concepts, ALPHA, LEVEL_TARGETS)
from tkh.embeddings import encode_semantic  # noqa: E402
from tkh.eval.coherence import fit_tfidf, coherence_vs_null  # noqa: E402
from tkh.eval.stability import perturbation_stability, cross_snapshot_stability  # noqa: E402
from tkh.eval.extrinsic import (  # noqa: E402
    routing_pool_recall, load_type_a_questions, centroid_vectors)

OUT_DIR = ROOT / "outputs"
VARIANTS = ["dendrogram", "multilevel", "multilevel_sizenorm"]
ROUTING_BUDGETS = [(2, 2), (3, 3), (4, 4), (6, 6), (8, 8)]
N_BOOT = 2000
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def member_sets(h):
    return {(sn["level"], frozenset(sn["member_ids"])) for sn in h["super_nodes"]}


def size_stats(h, n_levels):
    out = {}
    for level in range(n_levels):
        sizes = sorted(sn["member_count"] for sn in h["super_nodes"] if sn["level"] == level)
        out[level] = {"n_clusters": len(sizes), "min": sizes[0], "median": sizes[len(sizes) // 2],
                      "max": sizes[-1], "max_share": sizes[-1] / sum(sizes)}
    return out


def main():
    data = load_tkh(ROOT / "data" / "tkh_collection10.json")
    snapshots = build_all_snapshots(data, cutoffs=SNAPSHOT_CUTOFFS)
    cache = {}
    for y in SNAPSHOT_CUTOFFS:
        embed_concepts(snapshots[y], cache)
    log("embeddings done")

    tfidf = {y: fit_tfidf(snapshots[y]) for y in SNAPSHOT_CUTOFFS}
    last = max(SNAPSHOT_CUTOFFS)
    questions = load_type_a_questions(ROOT, snapshots[last], encode_semantic)
    n_levels = len(LEVEL_TARGETS)

    results = {}
    for variant in VARIANTS:
        res = run_all_snapshots(snapshots, coarsening=variant, embedding_cache=cache)
        hier = {y: build_hierarchy_json(res, y, snapshots[y]) for y in SNAPSHOT_CUTOFFS}
        log(f"{variant}: built")

        if variant == "dendrogram":
            for y in SNAPSHOT_CUTOFFS:
                shipped = json.loads((OUT_DIR / "snapshots" / str(y) / "hierarchy.json").read_text(encoding="utf-8"))
                assert member_sets(shipped) == member_sets(hier[y]), f"{y}: refactor changed the shipped hierarchy"
            log("dendrogram variant reproduces shipped hierarchy.json exactly")

        coherence = {}
        for y in SNAPSHOT_CUTOFFS:
            X, _, id_to_row = tfidf[y]
            coherence[y] = {}
            for level in range(n_levels):
                r = coherence_vs_null(hier[y], level, X, id_to_row, n_trials=30, seed=0)
                r.pop("per_cluster")
                coherence[y][level] = r
        cross = cross_snapshot_stability(hier, n_levels)
        pert = perturbation_stability(snapshots[last], cache, LEVEL_TARGETS, ALPHA, hier[last],
                                      coarsening=variant)
        log(f"{variant}: coherence and stability done")

        rng = np.random.default_rng(0)
        vecs = centroid_vectors(hier[last], cache)
        routing = [routing_pool_recall(hier[last], snapshots[last], questions, vecs, b0, b1, rng,
                                       n_boot=N_BOOT)
                   for b0, b1 in ROUTING_BUDGETS]

        results[variant] = {
            "coherence": coherence,
            "cross_snapshot": cross["by_level"],
            "perturbation": pert["by_level"],
            "perturbation_forced_merges_by_seed": pert["n_forced_merges_by_seed"],
            "forced_merges_by_year": {y: res["per_year"][y]["forced_by_level"] for y in SNAPSHOT_CUTOFFS},
            "sizes_by_year": {y: size_stats(hier[y], n_levels) for y in SNAPSHOT_CUTOFFS},
            "label_free_routing": routing,
        }

    summary = {}
    for variant, r in results.items():
        summary[variant] = {}
        for level in range(n_levels):
            summary[variant][level] = {
                "mean_coherence": float(np.mean([r["coherence"][y][level]["observed_coherence"]
                                                 for y in SNAPSHOT_CUTOFFS])),
                "mean_null": float(np.mean([r["coherence"][y][level]["null_mean"] for y in SNAPSHOT_CUTOFFS])),
                "perturbation_ari": r["perturbation"][level]["mean_ari"],
                "perturbation_ci95": r["perturbation"][level]["ci95"],
                "cross_snapshot_ari": r["cross_snapshot"][level]["mean_ari"],
                "max_share_2026": r["sizes_by_year"][last][level]["max_share"],
            }

    d = summary["dendrogram"]
    checks, passes = {}, {}
    for variant in VARIANTS[1:]:
        m = summary[variant]
        checks[variant] = {level: {
            "coherence_within_10pct": m[level]["mean_coherence"] >= 0.9 * d[level]["mean_coherence"],
            "perturbation_not_below_ci": m[level]["perturbation_ari"] >= d[level]["perturbation_ci95"][0],
            "cross_snapshot_within_0.05": m[level]["cross_snapshot_ari"] >= d[level]["cross_snapshot_ari"] - 0.05,
        } for level in (0, 1)}
        passes[variant] = all(all(c.values()) for c in checks[variant].values())

    out = {"alpha": ALPHA, "level_targets": LEVEL_TARGETS, "summary": summary,
           "decision_checks": checks, "passes_rule": passes,
           "variants": results}
    (OUT_DIR / "coarsening_compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for level in range(n_levels):
        for variant in VARIANTS:
            s = summary[variant][level]
            print(f"L{level} {variant:10s} coh={s['mean_coherence']:.4f} (null {s['mean_null']:.4f}) "
                  f"pert={s['perturbation_ari']:.3f} {np.round(s['perturbation_ci95'], 3).tolist()} "
                  f"cross={s['cross_snapshot_ari']:.3f} max_share={s['max_share_2026']:.2f}")
    for variant in VARIANTS:
        for r in results[variant]["label_free_routing"]:
            print(f"routing {variant:10s} ({r['b0']},{r['b1']}): gt_in_pool={r['gt_in_pool_rate']:.3f} "
                  f"pool={r['mean_pool_fraction']:.3f} lift={r['lift_over_chance']:.3f} "
                  f"CI{np.round(r['lift_ci95'], 3).tolist()}")
    for variant in VARIANTS[1:]:
        print(f"forced merges ({variant}):", results[variant]["forced_merges_by_year"])
        print(f"decision checks ({variant}):", checks[variant])
        print(f"passes rule ({variant}):", passes[variant])
    log(f"wrote {OUT_DIR / 'coarsening_compare.json'}")


if __name__ == "__main__":
    main()
