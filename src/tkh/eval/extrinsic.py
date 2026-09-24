"""T6 extrinsic task: find each question's expected methods
(questions.csv, ground_truth.json) by drilling down the hierarchy, against
a flat ranking of every node.

Both use each node's cosine similarity to the question. Flat ranks by that
alone. Drill-down picks level-0 and then level-1 super-nodes by their
label and gloss, pools the nodes under them, and ranks the pool with the
node's level-1 ancestor score blended in (beta). Level 2 has no labels,
so it's only used for membership.

Short surface forms are embedded with their hyperedge neighbours added
(build_retrieval_texts). DESIGN_NOTES.md section 14.
"""
import csv
import json
from collections import Counter

import numpy as np

METHOD_LIKE_TYPES = {"method", "technique", "cited_work", "dataset", "component"}


def _context_counts(snap, node_ids):
    """{node id: Counter of the surface forms it shares a hyperedge with}."""
    counts = {nid: Counter() for nid in node_ids}
    for e in snap.hyperedges:
        present = [m for m in e["members"] if m in counts]
        if not present:
            continue
        forms = [(m, snap.nodes[m]["surface_form"]) for m in e["members"]]
        for nid in present:
            for m, form in forms:
                if m != nid and form:
                    counts[nid][form] += 1
    return counts


def build_retrieval_texts(snap, node_ids, thin_threshold=20, max_context_terms=10):
    """(texts, enriched ids): the surface form as is when it's longer than
    thin_threshold characters, otherwise followed by its most frequent
    hyperedge neighbours. Why only the short ones: DESIGN_NOTES.md
    section 14."""
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
    """{lowercased surface form: node ids} for nodes of `types`."""
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


# embeddings are unit length, so a dot product is the cosine similarity


def flat_baseline(question_vec, node_ids, node_emb_matrix, k=20):
    """(top-k node ids by cosine to the question, number of nodes scored)."""
    order = np.argsort(-(node_emb_matrix @ question_vec))[:k]
    return [node_ids[i] for i in order], len(node_ids)


def build_level1_ancestor_map(hierarchy):
    """{node id: its level-1 super-node}, read off the level-2 parents."""
    return {nid: sn["parent_id"] for sn in hierarchy["super_nodes"] if sn["level"] == 2
            for nid in sn["member_ids"]}


def hierarchy_drilldown(question_vec, hierarchy, label_gloss_ids, label_gloss_matrix,
                        node_emb_by_id, b0=3, b1=3, k=20, beta=0.0, node_to_level1=None):
    """Keep the b0 level-0 super-nodes whose label and gloss are closest to
    the question, then the b1 closest of their level-1 children, and rank
    the nodes under those by cosine, blended with the node's level-1
    ancestor score by weight beta. Returns (top-k ids, pool size, counts
    of what was scored at each level)."""
    lg_score = dict(zip(label_gloss_ids, label_gloss_matrix @ question_vec))

    level0 = [sn for sn in hierarchy["super_nodes"] if sn["level"] == 0]
    level0_sorted = sorted(level0, key=lambda sn: -lg_score.get(sn["id"], -1))
    selected0_ids = {sn["id"] for sn in level0_sorted[:b0]}

    level1 = [sn for sn in hierarchy["super_nodes"]
              if sn["level"] == 1 and sn["parent_id"] in selected0_ids]
    level1_sorted = sorted(level1, key=lambda sn: -lg_score.get(sn["id"], -1))
    selected1_ids = {sn["id"] for sn in level1_sorted[:b1]}

    level2 = [sn for sn in hierarchy["super_nodes"]
              if sn["level"] == 2 and sn["parent_id"] in selected1_ids]

    detail = {"n_level0_scored": len(level0), "n_level1_scored": len(level1),
              "n_level2_groups_selected": len(level2)}
    candidate_pool = sorted({nid for sn in level2 for nid in sn["member_ids"]
                             if nid in node_emb_by_id})
    if not candidate_pool:
        return [], 0, detail

    sims = np.stack([node_emb_by_id[nid] for nid in candidate_pool]) @ question_vec
    if beta:
        ancestor_sims = np.array([lg_score.get((node_to_level1 or {}).get(nid), 0.0)
                                  for nid in candidate_pool])
        sims = (1 - beta) * sims + beta * ancestor_sims
    order = np.argsort(-sims)[:k]
    return [candidate_pool[i] for i in order], len(candidate_pool), detail


def score_retrieval(retrieved, gt_node_ids, k):
    """Recall and precision at k against the ground-truth node ids."""
    gt = set(gt_node_ids)
    if not gt:
        return None
    hit = len(set(retrieved) & gt)
    return {
        "recall_at_k": hit / len(gt),
        "precision_at_k": hit / k if k else None,
        "n_gt": len(gt), "n_hit": hit,
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
    """{super-node id: unit-length mean of its members' embeddings}."""
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
