"""Structure-side coherence with held-out hyperedges, traded off against
TF-IDF coherence across alpha. Decision rule written before running:
DESIGN_NOTES.md section 15. Writes outputs/structural_holdout.json and
copies the summary into outputs/metrics.json under "structural_holdout".
Changes nothing in the shipped pipeline.
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, DATA_PATH  # noqa: E402
from tkh.pipeline import build_levels, embed_concepts, KNN_K, LEVEL_TARGETS, ALPHA  # noqa: E402
from tkh.embeddings import semantic_knn_graph  # noqa: E402
from tkh.eval.coherence import fit_tfidf, coherence_vs_null, heldout_edge_cohesion  # noqa: E402

YEAR = 2026
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
SCHEMES = ["edge", "paper"]
N_SEEDS = 5
HOLDOUT_FRAC = 0.20
NULL_TRIALS = 10
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def mean_ci(values):
    a = np.asarray(values, dtype=float)
    m = float(a.mean())
    if len(a) < 2 or a.std(ddof=1) == 0:
        return m, [m, m]
    lo, hi = stats.t.interval(0.95, len(a) - 1, loc=m, scale=a.std(ddof=1) / np.sqrt(len(a)))
    return m, [float(lo), float(hi)]


def split(snap, scheme, seed):
    """(training snapshot, held-out edges as concept-member lists)."""
    rng = np.random.default_rng(seed)
    edges = snap.hyperedges
    qualifying = [i for i, e in enumerate(edges)
                  if sum(m in snap.concept_ids for m in e["members"]) >= 2]
    if scheme == "edge":
        held = set(rng.choice(qualifying, size=round(HOLDOUT_FRAC * len(qualifying)), replace=False).tolist())
    else:
        papers = sorted({e["provenance"]["article_id"] for e in edges})
        chosen = set(rng.choice(papers, size=round(HOLDOUT_FRAC * len(papers)), replace=False).tolist())
        held = {i for i, e in enumerate(edges) if e["provenance"]["article_id"] in chosen}
    train = dataclasses.replace(snap, hyperedges=[e for i, e in enumerate(edges) if i not in held])
    held_members = [[m for m in edges[i]["members"] if m in snap.concept_ids]
                    for i in sorted(held) if i in set(qualifying)]
    return train, held_members


def main():
    snap = build_snapshot(load_tkh(DATA_PATH), YEAR)
    cache = {}
    ids, emb = embed_concepts(snap, cache)
    A_sem = semantic_knn_graph(emb, k=KNN_K)
    X, _, id_to_row = fit_tfidf(snap)
    log("embeddings and TF-IDF ready")

    runs = []
    for scheme in SCHEMES:
        for s in range(N_SEEDS):
            train, held = split(snap, scheme, seed=3000 + s)
            for alpha in ALPHAS:
                built = build_levels(train, ids, emb, alpha=alpha, level_targets=LEVEL_TARGETS,
                                     coarsening="dendrogram", A_sem=A_sem)
                for level, labels in built["labels_by_level"].items():
                    obs, exp, lift = heldout_edge_cohesion(labels, ids, held)
                    clusters = {}
                    for nid, lab in zip(ids, labels):
                        clusters.setdefault(int(lab), []).append(nid)
                    h = {"super_nodes": [{"id": str(k), "level": level, "member_ids": v}
                                         for k, v in clusters.items()]}
                    coh = coherence_vs_null(h, level, X, id_to_row, n_trials=NULL_TRIALS, seed=s)
                    runs.append({
                        "scheme": scheme, "seed": s, "alpha": alpha, "level": level,
                        "n_heldout_edges": len(held), "heldout_observed": obs,
                        "heldout_expected": exp, "heldout_lift": lift,
                        "tfidf_observed": coh["observed_coherence"], "tfidf_null": coh["null_mean"],
                        "tfidf_ratio": coh["observed_coherence"] / coh["null_mean"],
                        "forced_merges": built["forced_by_level"][level],
                        "max_cluster_share": max(len(v) for v in clusters.values()) / len(ids),
                    })
            log(f"{scheme} seed {s} done ({len(held)} held-out edges)")

    def pick(scheme, alpha, level, key):
        return [r[key] for r in sorted(runs, key=lambda r: r["seed"])
                if r["scheme"] == scheme and r["alpha"] == alpha and r["level"] == level]

    summary = {}
    for scheme in SCHEMES:
        summary[scheme] = {}
        for level in range(len(LEVEL_TARGETS)):
            rows = []
            for alpha in ALPHAS:
                lift_m, lift_ci = mean_ci(pick(scheme, alpha, level, "heldout_lift"))
                coh_m, coh_ci = mean_ci(pick(scheme, alpha, level, "tfidf_ratio"))
                rows.append({"alpha": alpha, "heldout_lift": lift_m, "heldout_lift_ci95": lift_ci,
                             "tfidf_ratio": coh_m, "tfidf_ratio_ci95": coh_ci,
                             "forced_merges_mean": float(np.mean(pick(scheme, alpha, level, "forced_merges"))),
                             "max_cluster_share_mean": float(np.mean(pick(scheme, alpha, level, "max_cluster_share")))})
            diff = np.array(pick(scheme, ALPHA, level, "heldout_lift")) - np.array(pick(scheme, 0.0, level, "heldout_lift"))
            d_m, d_ci = mean_ci(diff)
            dominated_by = []
            for alpha in ALPHAS:
                if alpha == ALPHA:
                    continue
                dl = mean_ci(np.array(pick(scheme, alpha, level, "heldout_lift")) - np.array(pick(scheme, ALPHA, level, "heldout_lift")))[1]
                dc = mean_ci(np.array(pick(scheme, alpha, level, "tfidf_ratio")) - np.array(pick(scheme, ALPHA, level, "tfidf_ratio")))[1]
                if dl[0] > 0 and dc[0] > 0:
                    dominated_by.append(alpha)
            summary[scheme][level] = {"by_alpha": rows,
                                      "shipped_minus_alpha0_lift": {"mean": d_m, "ci95": d_ci},
                                      "shipped_dominated_by": dominated_by}

    structure_earns = {level: all(summary[s][level]["shipped_minus_alpha0_lift"]["ci95"][0] > 0 for s in SCHEMES)
                       for level in range(len(LEVEL_TARGETS))}
    out = {"year": YEAR, "alphas": ALPHAS, "schemes": SCHEMES, "n_seeds": N_SEEDS,
           "holdout_frac": HOLDOUT_FRAC, "shipped_alpha": ALPHA, "summary": summary,
           "structure_earns_its_place_by_level": structure_earns, "runs": runs}
    (ROOT / "outputs" / "structural_holdout.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    metrics_path = ROOT / "outputs" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["structural_holdout"] = {k: v for k, v in out.items() if k != "runs"}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for scheme in SCHEMES:
        for level in range(len(LEVEL_TARGETS)):
            s = summary[scheme][level]
            print(f"{scheme:5s} L{level}: " + "  ".join(
                f"a={r['alpha']}: lift {r['heldout_lift']:.2f} tfidf x{r['tfidf_ratio']:.2f}" for r in s["by_alpha"]))
            print(f"         shipped-alpha0 lift {s['shipped_minus_alpha0_lift']['mean']:+.2f} "
                  f"CI{np.round(s['shipped_minus_alpha0_lift']['ci95'], 2).tolist()}  dominated by {s['shipped_dominated_by']}")
    print("forced merges at alpha=1.0:", [summary['edge'][l]["by_alpha"][-1]["forced_merges_mean"] for l in range(3)])
    print("structure earns its place:", structure_earns)
    log("wrote outputs/structural_holdout.json and metrics.json")


if __name__ == "__main__":
    main()
