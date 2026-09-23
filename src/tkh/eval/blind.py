"""Blind judgements that don't use the clustering's own signal.

Intruder test (coherence): show five members of a super-node plus one
node from a different level-0 branch, shuffled, and ask a rater which one
doesn't belong. Null items do the same with five random nodes, so a rater
who can find the intruder by surface cues (type, length) shows up as
beating chance on the nulls. Chance is 1/6.

Gloss rating (faithfulness): show a gloss and a sample of members the
labeller never saw, and ask whether the gloss is accurate, vague or wrong.
Control items pair the gloss with another cluster's members. The NLI
judge is run on exactly the same members, so the two can be compared.

Items carry no ids that reveal their kind; the key lives in a separate
file. See DESIGN_NOTES.md sections 21 and 22.
"""
from collections import Counter

import numpy as np

from tkh.eval.stats import rate_with_ci, binomial_p_greater, cohen_kappa_ci

GLOSS_RATINGS = ("accurate", "vague", "wrong")
NLI_TO_RATING = {"entailment": "accurate", "neutral": "vague", "contradiction": "wrong"}


def _form(snap, nid):
    return (snap.nodes[nid].get("surface_form") or "").strip()


def level0_branch(hierarchy):
    """node id -> id of its level-0 super-node."""
    return {nid: sn["id"] for sn in hierarchy["super_nodes"] if sn["level"] == 0
            for nid in sn["member_ids"]}


def _pick_intruder(snap, shown, exclude, rng, candidates_by_type):
    """A node of the same type as one of the shown members, outside
    `exclude`, whose surface form doesn't repeat one already shown. Shown
    members are tried as the type anchor in random order, so a type with
    no candidates outside `exclude` doesn't lose the item."""
    shown_forms = {_form(snap, n).lower() for n in shown}
    for i in rng.permutation(len(shown)):
        pool = [n for n in candidates_by_type[snap.nodes[shown[i]]["type"]]
                if n not in exclude and _form(snap, n).lower() not in shown_forms and _form(snap, n)]
        if pool:
            return pool[rng.integers(len(pool))]
    return None


def _sample_distinct_forms(snap, ids, k, rng):
    """k ids from ids with pairwise-distinct, non-empty surface forms."""
    order = rng.permutation(len(ids))
    picked, forms = [], set()
    for i in order:
        f = _form(snap, ids[i]).lower()
        if f and f not in forms:
            picked.append(ids[i])
            forms.add(f)
        if len(picked) == k:
            return picked
    return None


def make_intruder_items(hierarchy, snap, n_per_level, n_null, rng, n_shown=5):
    """Returns (items, key). items: [{item_id, options}], key: {item_id: {...}}.
    n_per_level: {level: how many super-nodes to sample at that level}."""
    branch = level0_branch(hierarchy)
    concept = sorted(snap.concept_ids)
    by_type = {}
    for nid in concept:
        by_type.setdefault(snap.nodes[nid]["type"], []).append(nid)

    raw = []
    for level, n_items in sorted(n_per_level.items()):
        sns = [sn for sn in hierarchy["super_nodes"] if sn["level"] == level
               and len(sn["member_ids"]) >= n_shown]
        chosen = [sns[i] for i in rng.choice(len(sns), size=min(n_items, len(sns)), replace=False)]
        for sn in chosen:
            shown = _sample_distinct_forms(snap, sn["member_ids"], n_shown, rng)
            if shown is None:
                continue
            own_branch = {n for n in concept if branch.get(n) == branch.get(shown[0])}
            intruder = _pick_intruder(snap, shown, own_branch, rng, by_type)
            if intruder is None:
                continue
            raw.append({"kind": "real", "level": level, "super_node_id": sn["id"],
                        "shown": shown, "intruder": intruder})
    for _ in range(n_null):
        shown = _sample_distinct_forms(snap, concept, n_shown, rng)
        intruder = _pick_intruder(snap, shown, set(shown), rng, by_type)
        raw.append({"kind": "null", "level": None, "super_node_id": None,
                    "shown": shown, "intruder": intruder})

    items, key = [], {}
    for j, i in enumerate(rng.permutation(len(raw))):
        r = raw[i]
        nodes = r["shown"] + [r["intruder"]]
        perm = rng.permutation(len(nodes))
        nodes = [nodes[p] for p in perm]
        item_id = f"I{j + 1:03d}"
        items.append({"item_id": item_id, "options": [_form(snap, n) for n in nodes]})
        key[item_id] = {"kind": r["kind"], "level": r["level"], "super_node_id": r["super_node_id"],
                        "intruder_index": nodes.index(r["intruder"]), "option_node_ids": nodes}
    return items, key


def score_intruder(key, ratings, n_options=6):
    """ratings: {item_id: chosen option index}. Detection rate per kind and
    per level, Wilson CIs, and a one-sided binomial test against chance."""
    chance = 1 / n_options
    rows = []
    for item_id, k in key.items():
        if item_id in ratings:
            rows.append((k["kind"], k["level"], int(ratings[item_id]) == k["intruder_index"]))

    def summarise(sel):
        hits = sum(1 for r in sel if r[2])
        return {**rate_with_ci(hits, len(sel)), "p_vs_chance": binomial_p_greater(hits, len(sel), chance)}

    out = {"chance": chance, "n_rated": len(rows), "n_items": len(key),
           "real": summarise([r for r in rows if r[0] == "real"]),
           "null": summarise([r for r in rows if r[0] == "null"]),
           "real_by_level": {}}
    for level in sorted({r[1] for r in rows if r[0] == "real"}):
        out["real_by_level"][str(level)] = summarise([r for r in rows if r[0] == "real" and r[1] == level])
    return out


def score_gloss_ratings(key, ratings, rng):
    """ratings: {item_id: 'accurate'|'vague'|'wrong'}. Rater distribution
    for real and control items, over-claim rate (share rated wrong) with
    Wilson CIs, and agreement with the NLI judge on the same items."""
    rows = [(k["kind"], ratings[i], k["nli_label"]) for i, k in key.items() if i in ratings]
    bad = [r for r in rows if r[1] not in GLOSS_RATINGS]
    if bad:
        raise ValueError(f"unknown ratings: {sorted({r[1] for r in bad})}")

    out = {"n_rated": len(rows), "n_items": len(key)}
    for kind in ("real", "control"):
        sel = [r for r in rows if r[0] == kind]
        c = Counter(r[1] for r in sel)
        out[kind] = {rating: rate_with_ci(c[rating], len(sel)) for rating in GLOSS_RATINGS}
    out["over_claim_rate_real"] = out["real"]["wrong"]

    rater3 = [r[1] for r in rows]
    nli3 = [NLI_TO_RATING[r[2]] for r in rows]
    out["agreement_3way"] = cohen_kappa_ci(rater3, nli3, rng)
    out["agreement_wrong_vs_contradiction"] = cohen_kappa_ci(
        [r[1] == "wrong" for r in rows], [r[2] == "contradiction" for r in rows], rng)
    out["confusion_rater_by_nli"] = {
        rating: dict(Counter(r[2] for r in rows if r[1] == rating)) for rating in GLOSS_RATINGS}
    real_not_entailed = [r for r in rows if r[0] == "real" and r[2] != "entailment"]
    out["real_nli_not_entailed_rated_accurate"] = rate_with_ci(
        sum(1 for r in real_not_entailed if r[1] == "accurate"), len(real_not_entailed))
    return out


def make_gloss_items(hierarchies_by_year, snaps, n_real, n_control, rng,
                     levels=(0, 1), max_members=15, min_held_out=5):
    """Returns (items, key). n_real / n_control: per snapshot. Members come
    from the super-node's held-out set (never shown to the labeller); a
    control pairs a gloss with another checkable super-node's held-out
    members from the same snapshot. key[item_id]["premise"] is the exact
    NLI premise for those members, so the caller can run the NLI judge on
    identical inputs."""
    from tkh.eval.faithfulness import held_out_member_ids, premise_member_forms

    raw = []
    for year in sorted(hierarchies_by_year):
        h, snap = hierarchies_by_year[year], snaps[year]
        targets = [sn for sn in h["super_nodes"] if sn["level"] in levels and sn.get("gloss")]
        held = {sn["id"]: held_out_member_ids(sn["member_ids"]) for sn in targets}
        checkable = [sn for sn in targets if len(held[sn["id"]]) >= min_held_out]
        real_idx = rng.choice(len(checkable), size=min(n_real, len(checkable)), replace=False)
        for i in real_idx:
            sn = checkable[i]
            raw.append({"kind": "real", "year": year, "sn": sn, "src": sn})
        ctrl_idx = rng.choice(len(checkable), size=min(n_control, len(checkable)), replace=False)
        for i in ctrl_idx:
            sn = checkable[i]
            others = [s for s in checkable if s["id"] != sn["id"]]
            raw.append({"kind": "control", "year": year, "sn": sn, "src": others[rng.integers(len(others))]})
        for r in raw:
            if r["year"] == year:
                r["forms"] = premise_member_forms(snap, held[r["src"]["id"]], max_members, rng)

    items, key = [], {}
    for j, i in enumerate(rng.permutation(len(raw))):
        r = raw[i]
        item_id = f"G{j + 1:03d}"
        items.append({"item_id": item_id, "gloss": r["sn"]["gloss"], "members": r["forms"]})
        key[item_id] = {"kind": r["kind"], "year": r["year"], "level": r["sn"]["level"],
                        "super_node_id": r["sn"]["id"], "member_source_id": r["src"]["id"],
                        "premise": "The cluster includes " + ", ".join(r["forms"]) + "."}
    return items, key
