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

_model = None
_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def held_out_member_ids(member_ids, sampling=LABELLER_SAMPLING):
    shown = set(labeller_sample_ids(member_ids, sampling=sampling))
    return [nid for nid in member_ids if nid not in shown]


def build_premise(snap, member_ids, max_members, rng):
    """Premise from a seeded random sample of up to max_members of member_ids."""
    ids = list(member_ids)
    if len(ids) > max_members:
        ids = [ids[i] for i in sorted(rng.choice(len(ids), size=max_members, replace=False))]
    forms = []
    for nid in ids:
        node = snap.nodes.get(nid)
        if node is not None and node.get("surface_form"):
            forms.append(node["surface_form"])
    if not forms:
        return None
    return "The cluster includes " + ", ".join(forms) + "."


def check_hierarchy_faithfulness(hierarchy, snap, levels=(0, 1), max_members=15,
                                 min_held_out=5, seed=0, sampling=LABELLER_SAMPLING):
    """sampling must match how the labels being checked were produced
    (see labeling.labeller_sample_ids)."""
    model = _get_model()
    rng = np.random.default_rng(seed)
    id2label = model.config.id2label

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

    scores = model.predict(pairs, show_progress_bar=False)
    pred_idx = scores.argmax(axis=1)
    for m, i in zip(meta, pred_idx):
        m["nli_label"] = id2label[i]

    real = [m for m in meta if m["kind"] == "real"]
    control = [m for m in meta if m["kind"] == "control"]

    def rate(items, pred):
        return sum(1 for m in items if pred(m["nli_label"])) / len(items) if items else None

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
        "real_label_distribution": dist(real),
        "control_label_distribution": dist(control),
        "contradicted_super_node_ids": [m["super_node_id"] for m in real if m["nli_label"] == "contradiction"],
        "per_entry": meta,
    }
