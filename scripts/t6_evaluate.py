"""T6: coherence vs null model, stability (perturbation + cross-snapshot),
label faithfulness (NLI over-claim rate), extrinsic drill-down vs flat
baseline. Writes outputs/metrics.json.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, SNAPSHOT_CUTOFFS  # noqa: E402
from tkh.embeddings import encode_semantic  # noqa: E402
from tkh.eval.coherence import fit_tfidf, coherence_vs_null  # noqa: E402
from tkh.eval.stability import perturbation_stability, cross_snapshot_stability  # noqa: E402
from tkh.eval.faithfulness import check_hierarchy_faithfulness  # noqa: E402
from tkh.eval.extrinsic import (  # noqa: E402
    METHOD_LIKE_TYPES, match_ground_truth_methods, flat_baseline,
    hierarchy_drilldown, score_retrieval, build_retrieval_texts,
    build_level1_ancestor_map,
)

DATA_PATH = ROOT / "data" / "tkh_collection10.json"
OUT_DIR = ROOT / "outputs"
LEVEL_TARGETS = [12, 50, 200]
EXTRINSIC_K = 20
# At alpha=0.5 (original run): b0=3,b1=3 cut recall roughly in half vs
# flat, b0=5,b1=5 matched flat exactly at ~16% of the candidate pool.
# After the alpha ablation moved the default to 0.3 (DESIGN_NOTES.md section
# 7), the clustering changed enough that b0=5,b1=5 no longer matches flat
# (0.027 vs flat's 0.032 recall) -- re-swept via t6_patch_extrinsic.py:
# b0=8,b1=8 is the narrowest value that matches flat's recall and
# precision exactly again, at ~25% of the candidate pool (774 vs 3104).
# Wider than that doesn't help further, same saturation pattern as before.
# See DESIGN_NOTES.md section 14.
DRILL_B0, DRILL_B1 = 8, 8
# Matching flat was the ceiling for pool restriction alone (drill-down can
# only ever rank a subset of flat's pool with flat's own scoring function,
# so it can tie flat, never beat it). Blending in each candidate's level-1
# ancestor label+gloss score gives the ranker real information flat
# doesn't have. Swept beta 0.0-1.0 (DESIGN_NOTES.md section 14): beta=0.6 beats
# flat's recall and precision outright at the same candidate count as
# DRILL_B0/B1 above, not just matches it. See DESIGN_NOTES.md section 14.
DRILL_BETA = 0.6


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def run_coherence(snapshots, hierarchies):
    out = {}
    for year, snap in snapshots.items():
        X, ids, id_to_row = fit_tfidf(snap)
        out[year] = {
            level: coherence_vs_null(hierarchies[year], level, X, id_to_row, n_trials=30, seed=0)
            for level in range(len(LEVEL_TARGETS))
        }
        log(f"coherence {year} done")
    return out


def run_stability(snapshots, hierarchies):
    hierarchies_common = {y: hierarchies[y] for y in hierarchies}
    cross = cross_snapshot_stability(hierarchies_common, len(LEVEL_TARGETS))
    log("cross-snapshot stability done")

    year = max(snapshots)  # perturbation run on the largest/latest snapshot
    snap = snapshots[year]
    ids = sorted(snap.concept_ids)
    texts = [snap.nodes[nid]["surface_form"] or "" for nid in ids]
    vecs = encode_semantic(texts, show_progress_bar=True)
    cache = dict(zip(ids, vecs))
    log(f"perturbation embeddings ({year}) done")

    pert = perturbation_stability(snap, cache, LEVEL_TARGETS, alpha=0.3,
                                   original_hierarchy=hierarchies[year],
                                   n_seeds=5, remove_frac=0.10)
    log("perturbation stability done")
    return {"cross_snapshot": cross, "perturbation": {"year": year, **pert}}


def run_faithfulness(snapshots, hierarchies):
    out = {}
    for year, snap in snapshots.items():
        out[year] = check_hierarchy_faithfulness(hierarchies[year], snap, levels=(0, 1), seed=0)
        r = out[year]
        log(f"faithfulness {year} done: checked {r['n_checked']}/{r['n_labelled']} "
            f"(contradiction real={r['real_contradiction_rate']:.3f} control={r['control_contradiction_rate']:.3f}; "
            f"not-entailed real={r['real_not_entailed_rate']:.3f} control={r['control_not_entailed_rate']:.3f})")
    return out


def load_questions():
    with open(ROOT / "data" / "questions.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def run_extrinsic(snap, hierarchy):
    questions = load_questions()
    ground_truth = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))

    method_ids = sorted(nid for nid in snap.concept_ids if snap.nodes[nid]["type"] in METHOD_LIKE_TYPES)
    texts, thin_ids = build_retrieval_texts(snap, method_ids, thin_threshold=20, max_context_terms=10)
    node_emb_matrix = encode_semantic(texts, batch_size=32, max_seq_length=64, show_progress_bar=True)
    node_emb_by_id = dict(zip(method_ids, node_emb_matrix))
    log(f"extrinsic candidate-pool embeddings done (n={len(method_ids)}, n_enriched={len(thin_ids)})")

    lg_nodes = [sn for sn in hierarchy["super_nodes"] if sn["level"] in (0, 1) and sn.get("gloss")]
    lg_ids = [sn["id"] for sn in lg_nodes]
    lg_texts = [f"{sn['label']}. {sn['gloss']}" for sn in lg_nodes]
    lg_matrix = encode_semantic(lg_texts, show_progress_bar=False)
    node_to_level1 = build_level1_ancestor_map(hierarchy)

    results = []
    match_coverage = []
    for q in questions:
        qid = q["question_id"]
        gt = ground_truth.get(qid)
        if gt is None or gt.get("type") != "A":
            continue
        expected = gt["expected_methods"]
        matched = match_ground_truth_methods(snap, expected)
        gt_ids = sorted({nid for m in matched.values() for nid in m["node_ids"]})
        match_coverage.append({
            "question_id": qid, "n_expected": len(expected),
            "n_matched_terms": sum(1 for m in matched.values() if m["match_type"] != "none"),
            "match_types": {k: v["match_type"] for k, v in matched.items()},
        })
        if not gt_ids:
            continue

        qvec = encode_semantic([q["question"]])[0]
        flat_retrieved, flat_n = flat_baseline(qvec, method_ids, node_emb_matrix, k=EXTRINSIC_K)
        drill_retrieved, drill_n, drill_detail = hierarchy_drilldown(
            qvec, hierarchy, lg_ids, lg_matrix, node_emb_by_id,
            b0=DRILL_B0, b1=DRILL_B1, k=EXTRINSIC_K,
            beta=DRILL_BETA, node_to_level1=node_to_level1)

        results.append({
            "question_id": qid, "n_gt_node_ids": len(gt_ids),
            "flat": {**score_retrieval(flat_retrieved, gt_ids, EXTRINSIC_K), "n_candidates_scored": flat_n},
            "drilldown": {**score_retrieval(drill_retrieved, gt_ids, EXTRINSIC_K),
                          "n_candidates_scored": drill_n, **drill_detail},
        })
    log("extrinsic eval done")

    def avg(rows, path):
        vals = [r[path[0]][path[1]] for r in rows if r[path[0]].get(path[1]) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_questions_scored": len(results),
        "k": EXTRINSIC_K, "branching": {"b0": DRILL_B0, "b1": DRILL_B1, "beta": DRILL_BETA},
        "flat_mean_recall": avg(results, ("flat", "recall_at_k")),
        "flat_mean_precision": avg(results, ("flat", "precision_at_k")),
        "flat_mean_candidates_scored": avg(results, ("flat", "n_candidates_scored")),
        "drilldown_mean_recall": avg(results, ("drilldown", "recall_at_k")),
        "drilldown_mean_precision": avg(results, ("drilldown", "precision_at_k")),
        "drilldown_mean_candidates_scored": avg(results, ("drilldown", "n_candidates_scored")),
    }
    return {"summary": summary, "per_question": results, "ground_truth_match_coverage": match_coverage}


SECTIONS = ("coherence", "stability", "faithfulness", "extrinsic")


def main():
    """Usage: t6_evaluate.py [section ...]. With no arguments, runs every
    section and writes a fresh metrics.json. With section names, reruns only
    those and updates them in the existing metrics.json, leaving other keys."""
    global T0
    T0 = time.time()
    only = sys.argv[1:] or list(SECTIONS)
    unknown = [s for s in only if s not in SECTIONS]
    if unknown:
        sys.exit(f"unknown section(s) {unknown}, expected any of {SECTIONS}")

    data = load_tkh(DATA_PATH)
    snapshots = {y: build_snapshot(data, y) for y in SNAPSHOT_CUTOFFS}
    hierarchies = {y: json.loads((OUT_DIR / "snapshots" / str(y) / "hierarchy.json").read_text(encoding="utf-8"))
                   for y in SNAPSHOT_CUTOFFS}
    log("loaded snapshots and hierarchies")

    metrics_path = OUT_DIR / "metrics.json"
    metrics = {}
    if sys.argv[1:] and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "coherence" in only:
        metrics["coherence"] = run_coherence(snapshots, hierarchies)
    if "stability" in only:
        metrics["stability"] = run_stability(snapshots, hierarchies)
    if "faithfulness" in only:
        metrics["faithfulness"] = run_faithfulness(snapshots, hierarchies)
    if "extrinsic" in only:
        metrics["extrinsic"] = run_extrinsic(snapshots[2026], hierarchies[2026])

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"wrote {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
