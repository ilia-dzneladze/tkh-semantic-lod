"""T6 label faithfulness: over-claim rate via NLI entailment.

Bare short surface_form phrases don't give the NLI model enough to work
with (an off-the-shelf NLI model returns "neutral" on almost everything
when the premise is a single noun phrase), so a sample of a cluster's
members is aggregated into one natural-sentence premise.

The premise is built only from HELD-OUT members, ones the labeller was
not shown (see labeling.labeller_sample_ids), so the gloss is graded
against evidence it wasn't written from. Clusters with too few held-out
members are skipped and counted, not graded against the labeller's input.

Over-claim is reported two ways: contradiction rate (gloss conflicts with
the members) and not-entailed rate (members don't support the gloss). A
negative control checks each gloss against a random OTHER cluster.
See DESIGN_NOTES.md section 12.
"""
import numpy as np

from tkh.labeling import labeller_sample_ids, LABELLER_SAMPLING
from tkh.eval.stats import rate_with_ci
from tkh import model_outputs

_model = None
_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_MODEL_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"  # DESIGN_NOTES.md section 19


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        try:
            # a pinned commit that's already cached needs no network
            _model = CrossEncoder(_MODEL_NAME, revision=_MODEL_REVISION, local_files_only=True)
        except OSError:
            _model = CrossEncoder(_MODEL_NAME, revision=_MODEL_REVISION)
    return _model


def nli_labels(pairs):
    """NLI label ("contradiction" / "entailment" / "neutral") for each
    [premise, hypothesis] pair. Goes through model_outputs, so it can be
    recorded or replayed without loading the model (DESIGN_NOTES.md
    section 23)."""
    pairs = [list(p) for p in pairs]
    payload = {"model": f"{_MODEL_NAME}@{_MODEL_REVISION}", "pairs": pairs}
    scores = model_outputs.cached_array(
        "nli", payload, lambda: _get_model().predict(pairs, show_progress_bar=False))
    id2label = model_outputs.cached_json(
        "nli_id2label", lambda: {str(k): v for k, v in _get_model().config.id2label.items()})
    return [id2label[str(int(i))] for i in np.asarray(scores).argmax(axis=1)]


def held_out_member_ids(member_ids, sampling=LABELLER_SAMPLING):
    shown = set(labeller_sample_ids(member_ids, sampling=sampling))
    return [nid for nid in member_ids if nid not in shown]


def premise_member_forms(snap, member_ids, max_members, rng):
    """Surface forms of a seeded random sample of up to max_members of
    member_ids, in id order. Shared by the NLI premise and the blind
    rating packet so both judge exactly the same members."""
    ids = list(member_ids)
    if len(ids) > max_members:
        ids = [ids[i] for i in sorted(rng.choice(len(ids), size=max_members, replace=False))]
    forms = []
    for nid in ids:
        node = snap.nodes.get(nid)
        if node is not None and node.get("surface_form"):
            forms.append(node["surface_form"])
    return forms


def build_premise(snap, member_ids, max_members, rng):
    """Premise from a seeded random sample of up to max_members of member_ids."""
    forms = premise_member_forms(snap, member_ids, max_members, rng)
    if not forms:
        return None
    return "The cluster includes " + ", ".join(forms) + "."


def check_hierarchy_faithfulness(hierarchy, snap, levels=(0, 1), max_members=15,
                                 min_held_out=5, seed=0, sampling=LABELLER_SAMPLING):
    """sampling must match how the labels being checked were produced
    (see labeling.labeller_sample_ids)."""
    rng = np.random.default_rng(seed)

    targets = [sn for sn in hierarchy["super_nodes"]
               if sn["level"] in levels and sn.get("gloss")]
    held_out = {sn["id"]: held_out_member_ids(sn["member_ids"], sampling) for sn in targets}
    checkable = [sn for sn in targets if len(held_out[sn["id"]]) >= min_held_out]
    skipped = [sn["id"] for sn in targets if len(held_out[sn["id"]]) < min_held_out]

    pairs = []
    meta = []
    for sn in checkable:
        premise = build_premise(snap, held_out[sn["id"]], max_members, rng)
        if premise is None:
            continue
        pairs.append([premise, sn["gloss"]])
        meta.append({"super_node_id": sn["id"], "level": sn["level"], "kind": "real"})

        other = [s for s in checkable if s["id"] != sn["id"]]
        if other:
            control_sn = other[rng.integers(len(other))]
            control_premise = build_premise(snap, held_out[control_sn["id"]], max_members, rng)
            if control_premise is not None:
                pairs.append([control_premise, sn["gloss"]])
                meta.append({"super_node_id": sn["id"], "level": sn["level"], "kind": "control",
                             "control_source": control_sn["id"]})

    base = {"labeller_sampling": sampling, "n_labelled": len(targets), "n_skipped_too_few_held_out": len(skipped),
            "skipped_super_node_ids": skipped, "min_held_out": min_held_out,
            "max_premise_members": max_members}
    if not pairs:
        return {**base, "n_checked": 0}

    for m, label in zip(meta, nli_labels(pairs)):
        m["nli_label"] = label

    real = [m for m in meta if m["kind"] == "real"]
    control = [m for m in meta if m["kind"] == "control"]

    def rate(items, pred):
        return sum(1 for m in items if pred(m["nli_label"])) / len(items) if items else None

    def ci(items, pred):
        return rate_with_ci(sum(1 for m in items if pred(m["nli_label"])), len(items))["ci95"]

    def dist(items):
        out = {}
        for m in items:
            out[m["nli_label"]] = out.get(m["nli_label"], 0) + 1
        return out

    return {
        **base,
        "n_checked": len(real),
        "real_contradiction_rate": rate(real, lambda l: l == "contradiction"),
        "real_not_entailed_rate": rate(real, lambda l: l != "entailment"),
        "real_entailment_rate": rate(real, lambda l: l == "entailment"),
        "control_contradiction_rate": rate(control, lambda l: l == "contradiction"),
        "control_not_entailed_rate": rate(control, lambda l: l != "entailment"),
        "control_entailment_rate": rate(control, lambda l: l == "entailment"),
        "real_contradiction_ci95": ci(real, lambda l: l == "contradiction"),
        "real_not_entailed_ci95": ci(real, lambda l: l != "entailment"),
        "control_contradiction_ci95": ci(control, lambda l: l == "contradiction"),
        "control_not_entailed_ci95": ci(control, lambda l: l != "entailment"),
        "real_label_distribution": dist(real),
        "control_label_distribution": dist(control),
        "contradicted_super_node_ids": [m["super_node_id"] for m in real if m["nli_label"] == "contradiction"],
        "per_entry": meta,
    }


def load_article_titles(path):
    """collection10_articles.csv -> {article id: title}."""
    import csv
    with open(path, encoding="utf-8") as f:
        return {int(r["id"]): r["title"] for r in csv.DictReader(f)}


def provenance_premise(snap, member_ids, titles, max_titles=5):
    """Premise built from the titles of the papers the members were
    extracted from (node provenance), not from the members' own text. The
    most frequent source papers first, ties broken by article id."""
    counts = {}
    for nid in member_ids:
        for aid in (snap.nodes.get(nid, {}).get("provenance") or {}).get("articles", []):
            if aid in titles:
                counts[aid] = counts.get(aid, 0) + 1
    if not counts:
        return None
    top = sorted(counts, key=lambda a: (-counts[a], a))[:max_titles]
    return "These concepts were extracted from the papers: " + "; ".join(titles[a] for a in top) + "."


def check_provenance_faithfulness(hierarchy, snap, titles, levels=(0, 1), min_held_out=5,
                                  max_titles=5, seed=0, sampling=LABELLER_SAMPLING):
    """Each gloss against the titles of its held-out members' source
    papers, with the same random-other-cluster control as the member
    check. Titles were never shown to the labeller and never used for
    clustering, so this grades the gloss against evidence independent of
    the surface forms. See DESIGN_NOTES.md section 22."""
    rng = np.random.default_rng(seed)
    targets = [sn for sn in hierarchy["super_nodes"] if sn["level"] in levels and sn.get("gloss")]
    held_out = {sn["id"]: held_out_member_ids(sn["member_ids"], sampling) for sn in targets}
    checkable = [sn for sn in targets if len(held_out[sn["id"]]) >= min_held_out]

    pairs, meta = [], []
    for sn in checkable:
        premise = provenance_premise(snap, held_out[sn["id"]], titles, max_titles)
        if premise is None:
            continue
        pairs.append([premise, sn["gloss"]])
        meta.append({"super_node_id": sn["id"], "kind": "real"})
        other = [s for s in checkable if s["id"] != sn["id"]]
        control_sn = other[rng.integers(len(other))]
        control = provenance_premise(snap, held_out[control_sn["id"]], titles, max_titles)
        if control is not None and control != premise:
            pairs.append([control, sn["gloss"]])
            meta.append({"super_node_id": sn["id"], "kind": "control", "control_source": control_sn["id"]})
    if not pairs:
        return {"n_checked": 0}
    for m, label in zip(meta, nli_labels(pairs)):
        m["nli_label"] = label
    out = {"n_checked": sum(1 for m in meta if m["kind"] == "real"), "max_titles": max_titles}
    for kind in ("real", "control"):
        items = [m for m in meta if m["kind"] == kind]
        for name, test in (("contradiction", lambda l: l == "contradiction"),
                           ("not_entailed", lambda l: l != "entailment")):
            out[f"{kind}_{name}"] = rate_with_ci(sum(1 for m in items if test(m["nli_label"])), len(items))
    out["per_entry"] = meta
    return out
