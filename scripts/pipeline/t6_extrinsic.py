"""T6 extrinsic evaluation on the 2026 snapshot, written into metrics.json
under "extrinsic":

- drill-down at the shipped setting against flat retrieval, per question
  (the setting was picked in-sample),
- leave-one-out drill-down against flat, with a paired bootstrap CI and a
  sign-flip test (the headline number, DESIGN_NOTES.md section 15),
- routing: how much ground truth survives in the routed pool against a
  random pool of the same size, for label and centroid routing,
- the branching and beta sweeps behind the shipped setting.

Every drill-down setting in the grid (branching x beta) is scored once;
the sweeps and the shipped row are read from that grid.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, load_hierarchy, update_metrics, DATA_PATH  # noqa: E402
from tkh.embeddings import encode_semantic  # noqa: E402
from tkh.labeling import unlabelled_super_nodes  # noqa: E402
from tkh.pipeline import embed_concepts  # noqa: E402
from tkh.eval.extrinsic import (  # noqa: E402
    METHOD_LIKE_TYPES, match_ground_truth_methods, flat_baseline,
    hierarchy_drilldown, score_retrieval, build_retrieval_texts,
    build_level1_ancestor_map, leave_one_out_select, paired_comparison,
    routing_pool_recall, centroid_vectors,
)

YEAR = 2026
K = 20
SHIPPED = (8, 8, 0.6)  # b0, b1, beta: DESIGN_NOTES.md section 14
BRANCHING = [(3, 3), (5, 5), (8, 8), (5, 10), (8, 15)]
BETAS = [round(0.1 * i, 1) for i in range(11)]
ROUTING_BUDGETS = [(2, 2), (3, 3), (4, 4), (6, 6), (8, 8)]


def load_questions(snap):
    """[(question id, ground-truth node ids)] for the type-A questions with
    at least one matched node, plus the match coverage of every type-A
    question."""
    with open(ROOT / "data" / "questions.csv", encoding="utf-8") as f:
        questions = list(csv.DictReader(f, delimiter=";"))
    ground_truth = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))
    scored, coverage = [], []
    for q in questions:
        gt = ground_truth.get(q["question_id"])
        if gt is None or gt.get("type") != "A":
            continue
        matched = match_ground_truth_methods(snap, gt["expected_methods"])
        coverage.append({
            "question_id": q["question_id"], "n_expected": len(gt["expected_methods"]),
            "n_matched_terms": sum(1 for m in matched.values() if m["match_type"] != "none"),
            "match_types": {k: v["match_type"] for k, v in matched.items()},
        })
        gt_ids = sorted({nid for m in matched.values() for nid in m["node_ids"]})
        if gt_ids:
            scored.append((q, gt_ids))
    return scored, coverage


def mean_of(rows, key):
    return float(np.mean([r[key] for r in rows]))


def main():
    t0 = time.time()
    snap = build_snapshot(load_tkh(DATA_PATH), YEAR)
    hierarchy = load_hierarchy(YEAR)
    if unlabelled_super_nodes(hierarchy):
        sys.exit("hierarchy.json has unlabelled level-0/1 super-nodes; run t5_apply_labels.py first")

    method_ids = sorted(nid for nid in snap.concept_ids if snap.nodes[nid]["type"] in METHOD_LIKE_TYPES)
    texts, _ = build_retrieval_texts(snap, method_ids, thin_threshold=20, max_context_terms=10)
    node_emb_matrix = encode_semantic(texts, batch_size=32, max_seq_length=64, show_progress_bar=True)
    node_emb_by_id = dict(zip(method_ids, node_emb_matrix))
    print(f"[{time.time()-t0:.1f}s] candidate pool embedded: n={len(method_ids)}")

    lg_nodes = [sn for sn in hierarchy["super_nodes"] if sn["level"] in (0, 1)]
    lg_ids = [sn["id"] for sn in lg_nodes]
    lg_matrix = encode_semantic([f"{sn['label']}. {sn['gloss']}" for sn in lg_nodes], show_progress_bar=False)
    node_to_level1 = build_level1_ancestor_map(hierarchy)

    scored_questions, coverage = load_questions(snap)
    questions = [(q["question_id"], gt_ids, encode_semantic([q["question"]])[0]) for q, gt_ids in scored_questions]

    flat_rows = []
    for _, gt_ids, qvec in questions:
        retrieved, n_scored = flat_baseline(qvec, method_ids, node_emb_matrix, k=K)
        flat_rows.append({**score_retrieval(retrieved, gt_ids, K), "n_candidates_scored": n_scored})

    # one row per question for every drill-down setting
    grid = {}
    for b0, b1 in BRANCHING:
        for beta in BETAS:
            rows = []
            for _, gt_ids, qvec in questions:
                retrieved, n_scored, detail = hierarchy_drilldown(
                    qvec, hierarchy, lg_ids, lg_matrix, node_emb_by_id,
                    b0=b0, b1=b1, k=K, beta=beta, node_to_level1=node_to_level1)
                rows.append({**score_retrieval(retrieved, gt_ids, K), "n_candidates_scored": n_scored, **detail})
            grid[(b0, b1, beta)] = rows
    print(f"[{time.time()-t0:.1f}s] {len(grid)} drill-down settings scored")

    shipped = grid[SHIPPED]
    per_question = [{"question_id": qid, "n_gt_node_ids": len(gt_ids), "flat": flat, "drilldown": drill}
                    for (qid, gt_ids, _), flat, drill in zip(questions, flat_rows, shipped)]
    summary = {
        "n_questions_scored": len(questions),
        "k": K, "branching": dict(zip(("b0", "b1", "beta"), SHIPPED)),
        "flat_mean_recall": mean_of(flat_rows, "recall_at_k"),
        "flat_mean_precision": mean_of(flat_rows, "precision_at_k"),
        "flat_mean_candidates_scored": mean_of(flat_rows, "n_candidates_scored"),
        "drilldown_mean_recall": mean_of(shipped, "recall_at_k"),
        "drilldown_mean_precision": mean_of(shipped, "precision_at_k"),
        "drilldown_mean_candidates_scored": mean_of(shipped, "n_candidates_scored"),
    }

    def sweep_row(rows, **setting):
        return {**setting, "mean_recall": mean_of(rows, "recall_at_k"),
                "mean_precision": mean_of(rows, "precision_at_k"),
                "mean_candidates_scored": mean_of(rows, "n_candidates_scored")}

    branching_sweep = [sweep_row(grid[(b0, b1, 0.0)], b0=b0, b1=b1) for b0, b1 in BRANCHING]
    beta_sweep = [sweep_row(grid[(SHIPPED[0], SHIPPED[1], beta)], beta=beta) for beta in BETAS]

    # leave-one-out over the whole grid (DESIGN_NOTES.md section 15)
    flat_recall = np.array([r["recall_at_k"] for r in flat_rows])
    grid_recall = {c: np.array([r["recall_at_k"] for r in rows]) for c, rows in grid.items()}
    grid_cost = {c: (mean_of(rows, "n_candidates_scored"), c[2]) for c, rows in grid.items()}
    loo_recall, loo_chosen = leave_one_out_select(grid_recall, grid_cost)
    rng = np.random.default_rng(0)
    comparison = {
        "leave_one_out_vs_flat": {
            **paired_comparison(loo_recall, flat_recall, rng),
            "loo_mean_recall": float(loo_recall.mean()), "flat_mean_recall": float(flat_recall.mean()),
            "chosen_config_per_question": [
                {"question_id": q[0], "b0": c[0], "b1": c[1], "beta": c[2], "held_out_recall": float(s)}
                for q, c, s in zip(questions, loo_chosen, loo_recall)],
        },
        "shipped_in_sample_vs_flat": paired_comparison(grid_recall[SHIPPED], flat_recall, rng),
    }
    loo = comparison["leave_one_out_vs_flat"]
    print(f"[{time.time()-t0:.1f}s] leave-one-out: drill {loo_recall.mean():.4f} vs flat {flat_recall.mean():.4f}, "
          f"diff CI {np.round(loo['ci95'], 4).tolist()}, p={loo['p_sign_flip']:.3f}")

    # routing, by label+gloss and by member centroid, at every budget
    routing_questions = [(qid, qvec, gt_ids) for qid, gt_ids, qvec in questions]
    cache = {}
    embed_concepts(snap, cache)
    routing = {}
    for name, vecs in (("label", dict(zip(lg_ids, lg_matrix))), ("centroid", centroid_vectors(hierarchy, cache))):
        rng = np.random.default_rng(0)
        routing[name] = [routing_pool_recall(hierarchy, snap, routing_questions, vecs, b0, b1, rng)
                         for b0, b1 in ROUTING_BUDGETS]

    update_metrics("extrinsic", {
        "summary": summary, "per_question": per_question,
        "comparison_vs_flat": comparison, "routing": routing,
        "ground_truth_match_coverage": coverage,
        "branching_factor_sweep": branching_sweep,
        "rerank_beta_sweep": beta_sweep,
    })
    print(f"[{time.time()-t0:.1f}s] wrote outputs/metrics.json [extrinsic]")


if __name__ == "__main__":
    main()
