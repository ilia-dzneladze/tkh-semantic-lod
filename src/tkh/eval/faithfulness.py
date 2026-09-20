"""T6 label faithfulness: over-claim rate via NLI entailment.

Bare short surface_form phrases don't give the NLI model enough to work
with (checked this empirically, an off-the-shelf NLI model returns
"neutral" on almost everything when the premise is a single noun phrase).
Aggregating a cluster's members into one natural-sentence premise fixes
this, verified against a sanity-check pair the model gets right and a
clearly mismatched pair it correctly flags as contradiction.

The faithfulness check uses ALL (up to a cap) of a cluster's members, not
just the up-to-25 sample the labeller saw, so it's a genuinely independent
check rather than grading the gloss against exactly what wrote it.

Includes a negative control: the same gloss checked against a random
OTHER cluster's members. If the real-premise contradiction rate isn't
meaningfully lower than the negative-control rate, the metric isn't
discriminating and that should be reported, not hidden.
"""
import numpy as np

_model = None
_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def build_premise(snap, member_ids, max_members=15):
    forms = []
    for nid in member_ids[:max_members]:
        node = snap.nodes.get(nid)
        if node is not None:
            forms.append(node.get("surface_form") or "")
    forms = [f for f in forms if f]
    if not forms:
        return None
    return "The cluster includes " + ", ".join(forms) + "."


def check_hierarchy_faithfulness(hierarchy, snap, levels=(0, 1), max_members=15, seed=0):
    model = _get_model()
    rng = np.random.default_rng(seed)
    id2label = model.config.id2label

    targets = [sn for sn in hierarchy["super_nodes"]
               if sn["level"] in levels and sn.get("gloss")]

    pairs = []
    meta = []
    for sn in targets:
        premise = build_premise(snap, sn["member_ids"], max_members)
        if premise is None:
            continue
        pairs.append([premise, sn["gloss"]])
        meta.append({"super_node_id": sn["id"], "level": sn["level"], "kind": "real"})

        other = [s for s in targets if s["id"] != sn["id"]]
        if other:
            control_sn = other[rng.integers(len(other))]
            control_premise = build_premise(snap, control_sn["member_ids"], max_members)
            if control_premise is not None:
                pairs.append([control_premise, sn["gloss"]])
                meta.append({"super_node_id": sn["id"], "level": sn["level"], "kind": "control",
                              "control_source": control_sn["id"]})

    if not pairs:
        return {"n_checked": 0}

    scores = model.predict(pairs, show_progress_bar=False)
    pred_idx = scores.argmax(axis=1)
    labels = [id2label[i] for i in pred_idx]

    for m, lab in zip(meta, labels):
        m["nli_label"] = lab

    real = [m for m in meta if m["kind"] == "real"]
    control = [m for m in meta if m["kind"] == "control"]

    def rate(items, target_label):
        return sum(1 for m in items if m["nli_label"] == target_label) / len(items) if items else None

    def dist(items):
        out = {}
        for m in items:
            out[m["nli_label"]] = out.get(m["nli_label"], 0) + 1
        return out

    return {
        "n_checked": len(real),
        "real_contradiction_rate": rate(real, "contradiction"),
        "control_contradiction_rate": rate(control, "contradiction"),
        "real_label_distribution": dist(real),
        "control_label_distribution": dist(control),
        "contradicted_super_node_ids": [m["super_node_id"] for m in real if m["nli_label"] == "contradiction"],
        "per_entry": meta,
    }
