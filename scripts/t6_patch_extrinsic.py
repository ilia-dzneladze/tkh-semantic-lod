"""Recompute the extrinsic section of metrics.json without re-running the
expensive coherence/stability/faithfulness stages:

- the shipped drill-down setting vs flat (settings picked in-sample),
- leave-one-out drill-down vs flat with a paired bootstrap CI and
  sign-flip test (the headline number, DESIGN_NOTES.md section 15),
- routing: ground truth kept in the routed pool vs a same-size random pool,
  for label and centroid routing at every budget,
- the branching-factor and beta sweeps as supporting detail.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot  # noqa: E402
from tkh.embeddings import encode_semantic  # noqa: E402
from tkh.pipeline import embed_concepts  # noqa: E402
from tkh.eval.extrinsic import (  # noqa: E402
    METHOD_LIKE_TYPES, match_ground_truth_methods, flat_baseline,
    hierarchy_drilldown, score_retrieval, build_retrieval_texts,
    build_level1_ancestor_map, leave_one_out_select, paired_comparison,
    routing_pool_recall, centroid_vectors,
)

DATA_PATH = ROOT / "data" / "tkh_collection10.json"
OUT_DIR = ROOT / "outputs"
K = 20
FINAL_B0, FINAL_B1 = 8, 8
FINAL_BETA = 0.6
SWEEP = [(3, 3), (5, 5), (8, 8), (5, 10), (8, 15)]
BETA_SWEEP = [round(0.1 * i, 1) for i in range(11)]


def main():
    t0 = time.time()
    data = load_tkh(DATA_PATH)
    snap = build_snapshot(data, 2026)
    hierarchy = json.loads((OUT_DIR / "snapshots" / "2026" / "hierarchy.json").read_text(encoding="utf-8"))

    method_ids = sorted(nid for nid in snap.concept_ids if snap.nodes[nid]["type"] in METHOD_LIKE_TYPES)
    texts, thin_ids = build_retrieval_texts(snap, method_ids, thin_threshold=20, max_context_terms=10)
    node_emb_matrix = encode_semantic(texts, batch_size=32, max_seq_length=64, show_progress_bar=True)
    node_emb_by_id = dict(zip(method_ids, node_emb_matrix))
    print(f"[{time.time()-t0:.1f}s] candidate pool embedded: n={len(method_ids)}")

    lg_nodes = [sn for sn in hierarchy["super_nodes"] if sn["level"] in (0, 1) and sn.get("gloss")]
    lg_ids = [sn["id"] for sn in lg_nodes]
    lg_texts = [f"{sn['label']}. {sn['gloss']}" for sn in lg_nodes]
    lg_matrix = encode_semantic(lg_texts, show_progress_bar=False)
    node_to_level1 = build_level1_ancestor_map(hierarchy)

    with open(ROOT / "data" / "questions.csv", encoding="utf-8") as f:
        questions = list(csv.DictReader(f, delimiter=";"))
    ground_truth = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))

    per_question_ctx = []
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
        per_question_ctx.append((qid, gt_ids, qvec))

    results = []
    for qid, gt_ids, qvec in per_question_ctx:
        flat_retrieved, flat_n = flat_baseline(qvec, method_ids, node_emb_matrix, k=K)
        drill_retrieved, drill_n, drill_detail = hierarchy_drilldown(
            qvec, hierarchy, lg_ids, lg_matrix, node_emb_by_id,
            b0=FINAL_B0, b1=FINAL_B1, k=K, beta=FINAL_BETA, node_to_level1=node_to_level1)
        results.append({
            "question_id": qid, "n_gt_node_ids": len(gt_ids),
            "flat": {**score_retrieval(flat_retrieved, gt_ids, K), "n_candidates_scored": flat_n},
            "drilldown": {**score_retrieval(drill_retrieved, gt_ids, K),
                          "n_candidates_scored": drill_n, **drill_detail},
        })
    print(f"[{time.time()-t0:.1f}s] final-branching eval done "
          f"(b0={FINAL_B0}, b1={FINAL_B1}, beta={FINAL_BETA})")

    def avg(rows, path):
        vals = [r[path[0]][path[1]] for r in rows if r[path[0]].get(path[1]) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_questions_scored": len(results),
        "k": K, "branching": {"b0": FINAL_B0, "b1": FINAL_B1, "beta": FINAL_BETA},
        "flat_mean_recall": avg(results, ("flat", "recall_at_k")),
        "flat_mean_precision": avg(results, ("flat", "precision_at_k")),
        "flat_mean_candidates_scored": avg(results, ("flat", "n_candidates_scored")),
        "drilldown_mean_recall": avg(results, ("drilldown", "recall_at_k")),
        "drilldown_mean_precision": avg(results, ("drilldown", "precision_at_k")),
        "drilldown_mean_candidates_scored": avg(results, ("drilldown", "n_candidates_scored")),
    }

    # branching sweep, reusing the already-computed embeddings (beta=0, pure
    # node-cosine ranking, matches how this sweep was originally run)
    sweep_rows = []
    for b0, b1 in SWEEP:
        recalls, precisions, scored = [], [], []
        for qid, gt_ids, qvec in per_question_ctx:
            retrieved, n_scored, _ = hierarchy_drilldown(
                qvec, hierarchy, lg_ids, lg_matrix, node_emb_by_id, b0=b0, b1=b1, k=K)
            s = score_retrieval(retrieved, gt_ids, K)
            recalls.append(s["recall_at_k"])
            precisions.append(s["precision_at_k"])
            scored.append(n_scored)
        sweep_rows.append({
            "b0": b0, "b1": b1,
            "mean_recall": float(np.mean(recalls)), "mean_precision": float(np.mean(precisions)),
            "mean_candidates_scored": float(np.mean(scored)),
        })
    print(f"[{time.time()-t0:.1f}s] branching sweep done")

    # rerank beta sweep at the shipped (FINAL_B0, FINAL_B1) pool, DESIGN_NOTES.md
    # section 15: does blending in the level-1 ancestor's label+gloss
    # score let drill-down beat flat, not just match it?
    beta_sweep_rows = []
    for beta in BETA_SWEEP:
        recalls, precisions, scored = [], [], []
        for qid, gt_ids, qvec in per_question_ctx:
            retrieved, n_scored, _ = hierarchy_drilldown(
                qvec, hierarchy, lg_ids, lg_matrix, node_emb_by_id,
                b0=FINAL_B0, b1=FINAL_B1, k=K, beta=beta, node_to_level1=node_to_level1)
            s = score_retrieval(retrieved, gt_ids, K)
            recalls.append(s["recall_at_k"])
            precisions.append(s["precision_at_k"])
            scored.append(n_scored)
        beta_sweep_rows.append({
            "beta": beta,
            "mean_recall": float(np.mean(recalls)), "mean_precision": float(np.mean(precisions)),
            "mean_candidates_scored": float(np.mean(scored)),
        })
    print(f"[{time.time()-t0:.1f}s] rerank beta sweep done")

    # leave-one-out over the full branching x beta grid (DESIGN_NOTES.md section 15)
    flat_recall = np.array([r["flat"]["recall_at_k"] for r in results])
    grid_recall, grid_cost = {}, {}
    for b0, b1 in SWEEP:
        for beta in BETA_SWEEP:
            recalls, scored = [], []
            for qid, gt_ids, qvec in per_question_ctx:
                retrieved, n_scored, _ = hierarchy_drilldown(
                    qvec, hierarchy, lg_ids, lg_matrix, node_emb_by_id,
                    b0=b0, b1=b1, k=K, beta=beta, node_to_level1=node_to_level1)
                recalls.append(score_retrieval(retrieved, gt_ids, K)["recall_at_k"])
                scored.append(n_scored)
            grid_recall[(b0, b1, beta)] = np.array(recalls)
            grid_cost[(b0, b1, beta)] = (float(np.mean(scored)), beta)
    loo_recall, loo_chosen = leave_one_out_select(grid_recall, grid_cost)
    rng = np.random.default_rng(0)
    shipped = grid_recall[(FINAL_B0, FINAL_B1, FINAL_BETA)]
    comparison = {
        "leave_one_out_vs_flat": {
            **paired_comparison(loo_recall, flat_recall, rng),
            "loo_mean_recall": float(loo_recall.mean()), "flat_mean_recall": float(flat_recall.mean()),
            "chosen_config_per_question": [
                {"question_id": q[0], "b0": c[0], "b1": c[1], "beta": c[2], "held_out_recall": float(s)}
                for q, c, s in zip(per_question_ctx, loo_chosen, loo_recall)],
        },
        "shipped_in_sample_vs_flat": paired_comparison(shipped, flat_recall, rng),
    }
    print(f"[{time.time()-t0:.1f}s] leave-one-out: drill {loo_recall.mean():.4f} vs flat {flat_recall.mean():.4f}, "
          f"diff CI {np.round(comparison['leave_one_out_vs_flat']['ci95'], 4).tolist()}, "
          f"p={comparison['leave_one_out_vs_flat']['p_sign_flip']:.3f}")

    # routing: label vs centroid, every budget (the main extrinsic metric)
    routing_questions = [(qid, qvec, gt_ids) for qid, gt_ids, qvec in per_question_ctx]
    label_vecs = dict(zip(lg_ids, lg_matrix))
    cache = {}
    embed_concepts(snap, cache)
    cent_vecs = centroid_vectors(hierarchy, cache)
    routing = {}
    for name, vecs in (("label", label_vecs), ("centroid", cent_vecs)):
        rrng = np.random.default_rng(0)
        routing[name] = [routing_pool_recall(hierarchy, snap, routing_questions, vecs, b0, b1, rrng)
                         for b0, b1 in [(2, 2), (3, 3), (4, 4), (6, 6), (8, 8)]]
    print(f"[{time.time()-t0:.1f}s] routing done")

    metrics_path = OUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["extrinsic"] = {
        "summary": summary, "per_question": results,
        "comparison_vs_flat": comparison, "routing": routing,
        "ground_truth_match_coverage": match_coverage,
        "branching_factor_sweep": sweep_rows,
        "rerank_beta_sweep": beta_sweep_rows,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[{time.time()-t0:.1f}s] wrote {metrics_path}")


if __name__ == "__main__":
    main()
