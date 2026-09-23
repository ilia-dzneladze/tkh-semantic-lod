"""T6 extrinsic task: coarse-to-fine drill-down vs. a flat baseline,
scored against questions.csv / ground_truth.json's expected_methods.

Both start from the same per-node cosine similarity to the question. Flat
ranks by that alone. Drill-down routes through level-0 and level-1
label+gloss text to a candidate pool, and ranks it with each candidate's
level-1 ancestor score blended in (`beta` in `hierarchy_drilldown`).
Level 2 has no labels, so the pool itself is ranked on node embeddings.

Candidates are embedded with their hyperedge neighbourhood added when the
surface form is short (`build_retrieval_texts`), not by the bare surface
form used for clustering. See DESIGN_NOTES.md section 14.
"""
from collections import Counter
import numpy as np

METHOD_LIKE_TYPES = {"method", "technique", "cited_work", "dataset", "component"}


def _context_counts(snap, node_ids):
    """{node_id: Counter(neighbor surface_form -> co-occurrence count)}
    over node_ids' hyperedge memberships."""
    counts = {nid: Counter() for nid in node_ids}
    node_id_set = set(node_ids)
    for e in snap.hyperedges:
        members = e.get("members", [])
        present = [m for m in members if m in node_id_set]
        if not present:
            continue
        member_forms = [(m, snap.nodes[m].get("surface_form"))
                         for m in members if m in snap.nodes]
        for nid in present:
            for m, form in member_forms:
                if m != nid and form:
                    counts[nid][form] += 1
    return counts


def build_retrieval_texts(snap, node_ids, thin_threshold=20, max_context_terms=10):
    """Bare surface_form for nodes whose surface_form is already a
    descriptive phrase (>thin_threshold chars); surface_form enriched
    with its top co-occurring hyperedge neighbors for short/acronym-like
    ones. Enriching every node is correct but far too slow on CPU (each
    enriched text is long enough to hit the model's max sequence length,
    and padding within a batch means a handful of long texts drag the
    whole batch down): benchmarked at ~87s per 100 texts with 40 context
    terms on all 3104 method-like nodes, versus ~13s per 100 on just the
    short ones with 10 context terms. Only short names actually need the
    help, see the module docstring."""
    thin_ids = [nid for nid in node_ids
                if len(snap.nodes[nid].get("surface_form") or "") <= thin_threshold]
    thin_set = set(thin_ids)
    counts = _context_counts(snap, thin_ids)

    texts = []
    for nid in node_ids:
        own = snap.nodes[nid].get("surface_form") or ""
        if nid in thin_set:
            top_context = [t for t, _ in counts[nid].most_common(max_context_terms)]
            texts.append(own + (". " + ". ".join(top_context) if top_context else ""))
        else:
            texts.append(own)
    return texts, thin_ids


def build_surface_index(snap, types=METHOD_LIKE_TYPES):
    index = {}
    for nid, node in snap.nodes.items():
        if node.get("type") in types:
            key = (node.get("surface_form") or "").strip().lower()
            if key:
                index.setdefault(key, []).append(nid)
    return index


def match_ground_truth_methods(snap, expected_methods, min_substring_len=4):
    """Ground-truth method name -> the corpus nodes that count as that
    method. Exact surface-form match first, then substring in either
    direction. Both sides of a substring match must be at least
    min_substring_len characters. See DESIGN_NOTES.md section 16."""
    index = build_surface_index(snap)
    matched = {}
    for m in expected_methods:
        key = m.strip().lower()
        if key in index:
            matched[m] = {"node_ids": index[key], "match_type": "exact"}
            continue
        candidates = []
        if len(key) >= min_substring_len:
            for sf, ids in index.items():
                if len(sf) < min_substring_len:
                    continue
                if key in sf or sf in key:
                    candidates.extend(ids)
        matched[m] = {"node_ids": sorted(set(candidates)),
                      "match_type": "substring" if candidates else "none"}
    return matched


def _cos_matrix(emb_matrix, query_vec):
    # embeddings are L2-normalized (encode_semantic uses normalize_embeddings=True)
    return emb_matrix @ query_vec


def flat_baseline(question_vec, node_ids, node_emb_matrix, k=20):
    sims = _cos_matrix(node_emb_matrix, question_vec)
    order = np.argsort(-sims)[:k]
    retrieved = [node_ids[i] for i in order]
    return retrieved, len(node_ids)


def build_level1_ancestor_map(hierarchy):
    """node_id -> its level-1 ancestor's super-node id, via the level-2
    group it belongs to (level 2 has no label/gloss of its own)."""
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


def hierarchy_drilldown(question_vec, hierarchy, label_gloss_ids, label_gloss_matrix,
                          node_emb_by_id, b0=3, b1=3, k=20, beta=0.0, node_to_level1=None):
    lg_sims = _cos_matrix(label_gloss_matrix, question_vec)
    lg_score = dict(zip(label_gloss_ids, lg_sims))

    level0 = [sn for sn in hierarchy["super_nodes"] if sn["level"] == 0]
    level0_sorted = sorted(level0, key=lambda sn: -lg_score.get(sn["id"], -1))
    selected0_ids = {sn["id"] for sn in level0_sorted[:b0]}

    level1 = [sn for sn in hierarchy["super_nodes"]
              if sn["level"] == 1 and sn["parent_id"] in selected0_ids]
    level1_sorted = sorted(level1, key=lambda sn: -lg_score.get(sn["id"], -1))
    selected1_ids = {sn["id"] for sn in level1_sorted[:b1]}

    level2 = [sn for sn in hierarchy["super_nodes"]
              if sn["level"] == 2 and sn["parent_id"] in selected1_ids]

    candidate_pool = sorted({nid for sn in level2 for nid in sn["member_ids"]
                              if nid in node_emb_by_id})
    if not candidate_pool:
        return [], 0, {"n_level0_scored": len(level0), "n_level1_scored": len(level1),
                        "n_level2_groups_selected": len(level2)}

    cand_matrix = np.stack([node_emb_by_id[nid] for nid in candidate_pool])
    node_sims = _cos_matrix(cand_matrix, question_vec)
    if beta:
        ancestor_sims = np.array([lg_score.get((node_to_level1 or {}).get(nid), 0.0)
                                   for nid in candidate_pool])
        sims = (1 - beta) * node_sims + beta * ancestor_sims
    else:
        sims = node_sims
    order = np.argsort(-sims)[:k]
    retrieved = [candidate_pool[i] for i in order]

    return retrieved, len(candidate_pool), {
        "n_level0_scored": len(level0), "n_level1_scored": len(level1),
        "n_level2_groups_selected": len(level2),
    }


def score_retrieval(retrieved, gt_node_ids, k):
    gt = set(gt_node_ids)
    if not gt:
        return None
    hit = len(set(retrieved) & gt)
    return {
        "recall_at_k": hit / len(gt),
        "precision_at_k": hit / k if k else None,
        "n_gt": len(gt), "n_hit": hit,
    }


def run_extrinsic_eval(snap, hierarchy, questions, ground_truth, node_ids, node_emb_matrix,
                        node_emb_by_id, label_gloss_ids, label_gloss_matrix,
                        encode_fn, b0=3, b1=3, k=20):
    results = []
    for q in questions:
        qid = q["question_id"]
        gt = ground_truth.get(qid)
        if gt is None or gt.get("type") != "A":
            continue
        expected = gt["expected_methods"]
        matched = match_ground_truth_methods(snap, expected)
        gt_node_ids = sorted({nid for m in matched.values() for nid in m["node_ids"]})
        n_matched_terms = sum(1 for m in matched.values() if m["match_type"] != "none")

        qvec = encode_fn([q["question"]])[0]

        flat_retrieved, flat_n_scored = flat_baseline(qvec, node_ids, node_emb_matrix, k=k)
        drill_retrieved, drill_n_scored, drill_detail = hierarchy_drilldown(
            qvec, hierarchy, label_gloss_ids, label_gloss_matrix, node_emb_by_id,
            b0=b0, b1=b1, k=k)

        results.append({
            "question_id": qid, "question": q["question"],
            "n_expected_methods": len(expected),
            "n_expected_methods_matched": n_matched_terms,
            "n_gt_node_ids": len(gt_node_ids),
            "flat": {**(score_retrieval(flat_retrieved, gt_node_ids, k) or {}),
                     "n_candidates_scored": flat_n_scored},
            "drilldown": {**(score_retrieval(drill_retrieved, gt_node_ids, k) or {}),
                          "n_candidates_scored": drill_n_scored, **drill_detail},
        })
    return results


def summarize_extrinsic(results):
    def avg(key_path):
        vals = []
        for r in results:
            d = r
            for k in key_path:
                d = d.get(k, {}) if isinstance(d, dict) else {}
            if isinstance(d, (int, float)):
                vals.append(d)
        return float(np.mean(vals)) if vals else None

    return {
        "n_questions": len(results),
        "flat_mean_recall_at_k": avg(["flat", "recall_at_k"]),
        "flat_mean_precision_at_k": avg(["flat", "precision_at_k"]),
        "flat_mean_candidates_scored": avg(["flat", "n_candidates_scored"]),
        "drilldown_mean_recall_at_k": avg(["drilldown", "recall_at_k"]),
        "drilldown_mean_precision_at_k": avg(["drilldown", "precision_at_k"]),
        "drilldown_mean_candidates_scored": avg(["drilldown", "n_candidates_scored"]),
    }


def routing_pool_recall(hierarchy, snap, questions, sn_vectors, b0, b1, rng, n_boot=2000):
    """How much of each question's ground truth survives coarse-to-fine
    routing, against a random pool of the same size.

    questions: [(question_id, unit question vector, ground-truth node ids)].
    sn_vectors: {super_node_id: unit vector} for level 0 and 1 super-nodes,
    e.g. label+gloss embeddings or member centroids. Route to the top b0
    level-0 super-nodes, then the top b1 of their level-1 children; the pool
    is the method-like nodes under those. Chance = the pool's share of all
    method-like nodes. Returns the pooled ground-truth-in-pool rate, the mean
    pool fraction, and a bootstrap CI (over questions) on their difference.
    See DESIGN_NOTES.md section 14."""
    method_ids = {n for n in snap.concept_ids if snap.nodes[n]["type"] in METHOD_LIKE_TYPES}
    l0 = [sn for sn in hierarchy["super_nodes"] if sn["level"] == 0]
    l1 = [sn for sn in hierarchy["super_nodes"] if sn["level"] == 1]
    hits, totals, fracs = [], [], []
    for _, qv, gt_ids in questions:
        sel0 = {sn["id"] for sn in sorted(l0, key=lambda s: -(sn_vectors[s["id"]] @ qv))[:b0]}
        kids = [sn for sn in l1 if sn["parent_id"] in sel0]
        sel1 = sorted(kids, key=lambda s: -(sn_vectors[s["id"]] @ qv))[:b1]
        pool = {n for sn in sel1 for n in sn["member_ids"] if n in method_ids}
        hits.append(sum(1 for n in gt_ids if n in pool))
        totals.append(len(gt_ids))
        fracs.append(len(pool) / len(method_ids))
    hits, totals, fracs = map(np.array, (hits, totals, fracs))
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(hits), len(hits))
        boot.append(hits[i].sum() / totals[i].sum() - fracs[i].mean())
    rate = hits.sum() / totals.sum()
    return {"b0": b0, "b1": b1, "gt_in_pool_rate": float(rate),
            "mean_pool_fraction": float(fracs.mean()),
            "lift_over_chance": float(rate - fracs.mean()),
            "lift_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def load_type_a_questions(root, snap, encode_fn):
    """[(question_id, question vector, ground-truth node ids)] for the
    type-A questions whose expected methods match at least one node."""
    import csv
    import json
    with open(root / "data" / "questions.csv", encoding="utf-8") as f:
        questions = list(csv.DictReader(f, delimiter=";"))
    gt = json.loads((root / "data" / "ground_truth.json").read_text(encoding="utf-8"))
    out = []
    for q in questions:
        g = gt.get(q["question_id"])
        if not g or g.get("type") != "A":
            continue
        ids = sorted({n for m in match_ground_truth_methods(snap, g["expected_methods"]).values()
                      for n in m["node_ids"]})
        if ids:
            out.append((q["question_id"], encode_fn([q["question"]])[0], ids))
    return out


def centroid_vectors(hierarchy, emb_by_id, levels=(0, 1)):
    out = {}
    for sn in hierarchy["super_nodes"]:
        if sn["level"] in levels:
            v = np.mean([emb_by_id[n] for n in sn["member_ids"]], axis=0)
            out[sn["id"]] = v / np.linalg.norm(v)
    return out


def leave_one_out_select(scores_by_config, cost_by_config):
    """For each question, pick the config with the best mean score on the
    OTHER questions and score it on this one. scores_by_config: {config:
    per-question score array}. Ties go to lower cost_by_config[config]
    (compared as tuples). Returns (held-out scores, chosen config per
    question). See DESIGN_NOTES.md section 15."""
    configs = list(scores_by_config)
    n = len(scores_by_config[configs[0]])
    held_out, chosen = [], []
    for i in range(n):
        others = np.arange(n) != i
        best = min(configs, key=lambda c: (-scores_by_config[c][others].mean(), cost_by_config[c]))
        held_out.append(float(scores_by_config[best][i]))
        chosen.append(best)
    return np.array(held_out), chosen


def paired_comparison(a, b, rng, n_boot=10000):
    """Per-question paired comparison of a against b: mean difference,
    bootstrap 95% CI over questions, two-sided sign-flip p-value, and how
    many questions each side wins."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    flips = rng.choice([-1.0, 1.0], size=(n_boot, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return {
        "mean_diff": float(d.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_sign_flip": float((np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n_boot + 1)),
        "n_a_better": int((d > 0).sum()), "n_b_better": int((d < 0).sum()),
        "n_tied": int((d == 0).sum()), "n": int(len(d)),
    }
