"""T5: label + one-sentence gloss per super-node at levels 0-1.

This code makes no LLM API call. The labels were written by Claude Code
sub-agents reading the dumped prompt files. Why, and how temporal honesty
(P6) still holds: see DESIGN_NOTES.md section 12 and AI_USAGE.md.
"""
import json
import zlib

import numpy as np

LABEL_PROMPT_TEMPLATE = """\
You are labelling ONE cluster ("super-node") from a hierarchy built over a \
temporal knowledge hypergraph of ML/materials-science literature, as it \
looked at the {year} snapshot.

Below are {sample_n} member surface forms, a random sample of the \
{total_n} total members, and a breakdown of the node types of ALL \
members. Write:
  1. A short label (<=6 words) naming the shared scientific concept.
  2. A one-sentence gloss (<=25 words) describing what unites the members.

Rules:
- Assert NOTHING the member list does not support. If members are diverse \
or you are unsure, say so plainly in the gloss rather than overclaiming a \
crisp topic.
- Do not reference anything outside this member list (no outside knowledge \
about methods that AREN'T listed here, even if you know they exist).
- This snapshot is dated {year}: do not describe members using terms/status \
that wouldn't have been known by then.

Node types of all {total_n} members: {type_line}

Sampled members (type: surface_form):
{member_lines}
"""


LABELLER_SAMPLE_N = 25
LABELLER_SAMPLING = "random"  # "first" reproduces the v1 labels' input


def labeller_sample_ids(member_ids, n=LABELLER_SAMPLE_N, sampling=LABELLER_SAMPLING):
    """The member ids the labeller is shown. The faithfulness check
    excludes exactly these, so this must stay the single definition of the
    labeller's input, and a pure function of member_ids.

    "random": n ids drawn with a seed derived from the member list itself.
    "first": the first n sorted ids, which is type-biased because ids are
    type-prefixed (used for the v1 labels). See DESIGN_NOTES.md section 12."""
    ids = list(member_ids)
    if len(ids) <= n:
        return ids
    if sampling == "first":
        return ids[:n]
    if sampling != "random":
        raise ValueError(f"unknown sampling {sampling!r}")
    seed = zlib.crc32("\n".join(ids).encode("utf-8"))
    picked = np.random.default_rng(seed).choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(picked)]


def write_labeling_input(hierarchy, snap, year, out_path, levels=(0, 1)):
    """member_ids are raw concept node ids at every level (see
    pipeline.build_hierarchy_json), so the labelling prompt can read them
    directly -- no separate raw-member resolution needed."""
    entries = []
    for sn in hierarchy["super_nodes"]:
        if sn["level"] not in levels:
            continue
        raw_ids = sn["member_ids"]
        type_counts = {}
        sample_lines = []
        for nid in raw_ids:
            node = snap.nodes.get(nid)
            if node is None:
                continue
            t = node.get("type")
            type_counts[t] = type_counts.get(t, 0) + 1
        sample_ids = labeller_sample_ids(raw_ids)
        for nid in sample_ids:
            node = snap.nodes.get(nid)
            if node is None:
                continue
            sample_lines.append(f"  - {node.get('type')}: {node.get('surface_form')}")

        prompt = LABEL_PROMPT_TEMPLATE.format(
            year=year, sample_n=len(sample_ids), total_n=len(raw_ids),
            type_line=", ".join(f"{t} {c}" for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1])),
            member_lines="\n".join(sample_lines),
        )
        entries.append({
            "super_node_id": sn["id"], "level": sn["level"], "total_members": len(raw_ids),
            "type_breakdown": type_counts, "prompt": prompt,
            "label": None, "gloss": None,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    return entries


def apply_labels_to_hierarchy(hierarchy, labeled_entries_path):
    with open(labeled_entries_path, encoding="utf-8") as f:
        entries = json.load(f)
    by_id = {e["super_node_id"]: e for e in entries}
    n_applied = 0
    for sn in hierarchy["super_nodes"]:
        e = by_id.get(sn["id"])
        if e and e.get("label"):
            sn["label"] = e["label"]
            sn["gloss"] = e["gloss"]
            n_applied += 1
    return hierarchy, n_applied
