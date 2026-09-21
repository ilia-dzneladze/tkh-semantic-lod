"""FIXES.md iteration 7: does blending a candidate's own cosine score with
its level-1 ancestor's label+gloss cosine score let drill-down retrieval
beat flat baseline (not just tie it)? Reuses the same embeddings/questions
as t6_patch_extrinsic.py. Writes outputs/rerank_sweep.json.
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
from tkh.eval.extrinsic import (  # noqa: E402
    METHOD_LIKE_TYPES, match_ground_truth_methods, flat_baseline,
    score_retrieval, build_retrieval_texts,
)

DATA_PATH = ROOT / "data" / "tkh_collection10.json"
OUT_DIR = ROOT / "outputs"
K = 20
BETAS = [round(0.1 * i, 1) for i in range(11)]
SHIPPED_B0, SHIPPED_B1 = 8, 8


def _cos(matrix, vec):
    return matrix @ vec


def build_ancestor_maps(hierarchy):
    """node_id -> level1 ancestor supernode id, via the level-2 supernode
    it belongs to (level 2 has no label of its own, T5 only covers 0-1)."""
    level1_parent_of_level2 = {sn["id"]: sn["parent_id"]
                                for sn in hierarchy["super_nodes"] if sn["level"] == 2}
    node_to_level1 = {}
    for sn in hierarchy["super_nodes"]:
        if sn["level"] != 2:
            continue
        l1 = level1_parent_of_level2[sn["id"]]
        for nid in sn["member_ids"]:
            node_to_level1[nid] = l1
    return node_to_level1


def restricted_pool(hierarchy, lg_score, b0, b1):
    level0 = [sn for sn in hierarchy["super_nodes"] if sn["level"] == 0]
    level0_sorted = sorted(level0, key=lambda sn: -lg_score.get(sn["id"], -1))
    selected0 = {sn["id"] for sn in level0_sorted[:b0]}

    level1 = [sn for sn in hierarchy["super_nodes"]
              if sn["level"] == 1 and sn["parent_id"] in selected0]
    level1_sorted = sorted(level1, key=lambda sn: -lg_score.get(sn["id"], -1))
    selected1 = {sn["id"] for sn in level1_sorted[:b1]}

    level2 = [sn for sn in hierarchy["super_nodes"]
              if sn["level"] == 2 and sn["parent_id"] in selected1]
    return sorted({nid for sn in level2 for nid in sn["member_ids"]})


def rerank(pool_ids, node_emb_by_id, node_to_level1, lg_score, qvec, beta, k):
    pool_ids = [nid for nid in pool_ids if nid in node_emb_by_id]
    if not pool_ids:
        return [], 0
    node_matrix = np.stack([node_emb_by_id[nid] for nid in pool_ids])
    node_scores = _cos(node_matrix, qvec)
    ancestor_scores = np.array([lg_score.get(node_to_level1.get(nid), 0.0) for nid in pool_ids])
    final = (1 - beta) * node_scores + beta * ancestor_scores
    order = np.argsort(-final)[:k]
    return [pool_ids[i] for i in order], len(pool_ids)


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
    node_to_level1 = build_ancestor_maps(hierarchy)
    print(f"[{time.time()-t0:.1f}s] label+gloss embedded: n={len(lg_ids)}, "
          f"{len(node_to_level1)} nodes have a level-1 ancestor")

    with open(ROOT / "data" / "questions.csv", encoding="utf-8") as f:
        questions = list(csv.DictReader(f, delimiter=";"))
    ground_truth = json.loads((ROOT / "data" / "ground_truth.json").read_text(encoding="utf-8"))

    per_question = []
    for q in questions:
        qid = q["question_id"]
        gt = ground_truth.get(qid)
        if gt is None or gt.get("type") != "A":
            continue
        expected = gt["expected_methods"]
        matched = match_ground_truth_methods(snap, expected)
        gt_ids = sorted({nid for m in matched.values() for nid in m["node_ids"]})
        if not gt_ids:
            continue
        qvec = encode_semantic([q["question"]])[0]
        per_question.append((qid, gt_ids, qvec))
    print(f"[{time.time()-t0:.1f}s] {len(per_question)} scoreable questions")

    flat_rows = []
    for qid, gt_ids, qvec in per_question:
        retrieved, n = flat_baseline(qvec, method_ids, node_emb_matrix, k=K)
        flat_rows.append(score_retrieval(retrieved, gt_ids, K) | {"n_candidates_scored": n})
    flat_summary = {
        "mean_recall": float(np.mean([r["recall_at_k"] for r in flat_rows])),
        "mean_precision": float(np.mean([r["precision_at_k"] for r in flat_rows])),
        "mean_candidates_scored": float(np.mean([r["n_candidates_scored"] for r in flat_rows])),
    }
    print(f"[{time.time()-t0:.1f}s] flat baseline: {flat_summary}")

    results = {"full_pool": [], "restricted_pool_8_8": []}
    for beta in BETAS:
        for condition in ("full_pool", "restricted_pool_8_8"):
            recalls, precisions, scored = [], [], []
            for qid, gt_ids, qvec in per_question:
                lg_sims = _cos(lg_matrix, qvec)
                lg_score = dict(zip(lg_ids, lg_sims))
                if condition == "full_pool":
                    pool = method_ids
                else:
                    pool = restricted_pool(hierarchy, lg_score, SHIPPED_B0, SHIPPED_B1)
                retrieved, n = rerank(pool, node_emb_by_id, node_to_level1, lg_score, qvec, beta, K)
                s = score_retrieval(retrieved, gt_ids, K)
                recalls.append(s["recall_at_k"])
                precisions.append(s["precision_at_k"])
                scored.append(n)
            row = {
                "beta": beta, "mean_recall": float(np.mean(recalls)),
                "mean_precision": float(np.mean(precisions)),
                "mean_candidates_scored": float(np.mean(scored)),
                "beats_flat_recall": float(np.mean(recalls)) > flat_summary["mean_recall"],
            }
            results[condition].append(row)
        print(f"[{time.time()-t0:.1f}s] beta={beta} done")

    out = {"flat_summary": flat_summary, "shipped_drilldown_b0_b1": [SHIPPED_B0, SHIPPED_B1], **results}
    (OUT_DIR / "rerank_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[{time.time()-t0:.1f}s] wrote {OUT_DIR / 'rerank_sweep.json'}")


if __name__ == "__main__":
    main()
