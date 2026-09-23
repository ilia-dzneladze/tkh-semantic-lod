"""T5: label + one-sentence gloss per super-node at levels 0-1.

This code makes no LLM API call. The labels were written by Claude Code
sub-agents reading the dumped prompt files. Why, and how temporal honesty
(P6) still holds: see DESIGN_NOTES.md section 12 and AI_USAGE.md.
"""
import json
import zlib
from pathlib import Path

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


TEMPLATE_FIELDS = {"year", "sample_n", "total_n", "type_line", "member_lines"}
TEMPLATE_FILE = "prompt_template.txt"  # inside a label-set directory


def validate_template(text):
    """A labelling prompt template: str.format placeholders drawn from
    TEMPLATE_FIELDS, and it must show the members ({member_lines}).
    Literal braces must be doubled. Raises ValueError otherwise."""
    import string
    try:
        fields = {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}
    except ValueError as e:
        raise ValueError(f"template isn't a valid format string ({e}); double any literal braces") from None
    unknown = fields - TEMPLATE_FIELDS
    if unknown:
        raise ValueError(f"unknown placeholders {sorted(unknown)}; allowed: {sorted(TEMPLATE_FIELDS)}")
    if "member_lines" not in fields:
        raise ValueError("the template must include {member_lines}, or the labeller never sees the cluster")
    return text


def label_set_template(set_dir):
    """The template a label set was written from: its prompt_template.txt
    if it has one, else the default (the shipped sets have none)."""
    path = Path(set_dir) / TEMPLATE_FILE
    return validate_template(path.read_text(encoding="utf-8")) if path.exists() else LABEL_PROMPT_TEMPLATE


def build_label_prompt(sn, snap, year, template=LABEL_PROMPT_TEMPLATE):
    """The exact prompt the labeller sees for super-node sn. Also used as
    the fingerprint that ties a written label to the members it was
    written from (apply_labels_to_hierarchy)."""
    raw_ids = sn["member_ids"]
    type_counts = {}
    for nid in raw_ids:
        node = snap.nodes.get(nid)
        if node is None:
            continue
        t = node.get("type")
        type_counts[t] = type_counts.get(t, 0) + 1
    sample_ids = labeller_sample_ids(raw_ids)
    sample_lines = []
    for nid in sample_ids:
        node = snap.nodes.get(nid)
        if node is None:
            continue
        sample_lines.append(f"  - {node.get('type')}: {node.get('surface_form')}")
    prompt = template.format(
        year=year, sample_n=len(sample_ids), total_n=len(raw_ids),
        type_line=", ".join(f"{t} {c}" for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1])),
        member_lines="\n".join(sample_lines),
    )
    return prompt, type_counts


def write_labeling_input(hierarchy, snap, year, out_path, levels=(0, 1), template=LABEL_PROMPT_TEMPLATE):
    """member_ids are raw concept node ids at every level (see
    pipeline.build_hierarchy_json), so the labelling prompt can read them
    directly -- no separate raw-member resolution needed."""
    entries = []
    for sn in hierarchy["super_nodes"]:
        if sn["level"] not in levels:
            continue
        prompt, type_counts = build_label_prompt(sn, snap, year, template)
        entries.append({
            "super_node_id": sn["id"], "level": sn["level"], "total_members": len(sn["member_ids"]),
            "type_breakdown": type_counts, "prompt": prompt,
            "label": None, "gloss": None,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    return entries


def apply_labels_to_hierarchy(hierarchy, labeled_entries_path, snap=None, year=None,
                              template=LABEL_PROMPT_TEMPLATE):
    """Copy label and gloss onto the super-nodes they were written for.

    An entry only applies if its super-node still has the same member
    count and, when snap and year are given, the prompt rebuilt from the
    current members (with the template the set was made with) is
    identical to the one the label was written from.
    Anything else is stale (same persistent id, different cluster) and is
    skipped. See DESIGN_NOTES.md section 19.
    Returns (hierarchy, n_applied, stale_super_node_ids)."""
    with open(labeled_entries_path, encoding="utf-8") as f:
        entries = json.load(f)
    by_id = {e["super_node_id"]: e for e in entries}
    n_applied = 0
    stale = []
    for sn in hierarchy["super_nodes"]:
        e = by_id.get(sn["id"])
        if not e or not e.get("label"):
            continue
        fresh = e.get("total_members") == len(sn["member_ids"])
        if fresh and snap is not None:
            fresh = build_label_prompt(sn, snap, year, template)[0] == e.get("prompt")
        if not fresh:
            stale.append(sn["id"])
            continue
        sn["label"] = e["label"]
        sn["gloss"] = e["gloss"]
        n_applied += 1
    return hierarchy, n_applied, stale


def unlabelled_super_nodes(hierarchy, levels=(0, 1)):
    """Ids of super-nodes at `levels` with no gloss, e.g. straight after
    run_pipeline.py, which writes every label as null."""
    return [sn["id"] for sn in hierarchy["super_nodes"]
            if sn["level"] in levels and not sn.get("gloss")]


REQUEST_HEADER = """\
You are labelling {n} clusters ("super-nodes") from a hierarchy over a
knowledge graph of ML / materials-science literature, snapshot {year}.
Each item below has an id and its own instructions and members. Follow
each item's instructions.

Reply with ONLY a JSON array, one object per item, covering all {n} items:
[{{"super_node_id": "...", "label": "...", "gloss": "..."}}, ...]
"""


def write_labeller_request(entries, year, out_path):
    """One markdown file holding every prompt for a snapshot plus the reply
    format, ready to give to any LLM or human labeller. Save the reply as
    labels_<year>.json and import it with scripts/t5_import_labels.py."""
    parts = [REQUEST_HEADER.format(n=len(entries), year=year)]
    for e in entries:
        parts.append(f"=== item {e['super_node_id']} ===\n{e['prompt']}")
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")
