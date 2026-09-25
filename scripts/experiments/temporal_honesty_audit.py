"""Does any level-0/1 label or gloss name something the corpus first saw
after its snapshot? The rule and how to read the result:
DESIGN_NOTES.md section 25.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, load_hierarchy, CONCEPT_TYPES, DATA_PATH, OUTPUTS  # noqa: E402
from tkh.eval.stats import rate_with_ci  # noqa: E402

YEARS = [2020, 2022, 2024]
MIN_FORM_LEN = 4  # same floor as the ground-truth matcher, DESIGN_NOTES.md section 16
OUT_PATH = OUTPUTS / "temporal_honesty_audit.json"
SAMPLE_LINE = re.compile(r"^\s+- (\w+): (.+)$")

# the flags read as real leaks, as (year, super-node id, matched form).
# Every flag not supported by a seen input member was read by hand; the rest are
# field vocabulary ("materials", "accuracy", ...) or names the corpus had
# seen earlier under another surface form (MPNN, ACE, LightGBM)
REAL_LEAKS = {
    (2020, "L1_S00003", "deepmd-kit"),
    (2020, "L1_S00017", "oc20"),
    (2020, "L1_S00048", "equiformerv2"),
    (2022, "L1_S00017", "oc20"),
    (2022, "L1_S00017", "oc22"),
    (2022, "L1_S00009", "togo database"),
    (2022, "L1_S00009", "csp blind test"),
}


def _norm(form):
    return " ".join((form or "").split()).lower()


def _mentions(text, form):
    if form not in text.lower():  # cheap prefilter before the word-boundary check
        return False
    return re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", text, re.IGNORECASE) is not None


def sampled_members(prompt, member_nodes):
    """The member nodes whose (type, surface form) appear in the prompt's
    sample. A form shared by several members returns all of them."""
    by_key = defaultdict(list)
    for n in member_nodes:
        by_key[(n["type"], _norm(n["surface_form"]))].append(n)
    found = []
    in_sample = False
    for line in prompt.splitlines():
        if line.startswith("Sampled members"):
            in_sample = True
            continue
        m = SAMPLE_LINE.match(line) if in_sample else None
        if m:
            found += by_key.get((m.group(1), _norm(m.group(2))), [])
    return found


def main():
    data = load_tkh(DATA_PATH)
    nodes = {n["id"]: n for n in data["nodes"]}
    earliest = {}  # normalised surface form -> earliest first_seen_year in the export
    for n in data["nodes"]:
        if n["type"] in CONCEPT_TYPES and n.get("first_seen_year") is not None:
            f = _norm(n["surface_form"])
            earliest[f] = min(earliest.get(f, 9999), n["first_seen_year"])

    out = {"per_year": {}, "flags": []}
    exposed = set()  # (year, super-node id) whose labeller input held a future-seen node
    for year in YEARS:
        labels = json.loads((OUTPUTS / "snapshots" / str(year) / "labeling_output.json")
                            .read_text(encoding="utf-8"))
        members_of = {s["id"]: s["member_ids"] for s in load_hierarchy(year)["super_nodes"]}
        future_forms = {f for f, y in earliest.items() if y > year and len(f) >= MIN_FORM_LEN}
        n_exposed = n_flagged = 0
        for lab in labels:
            text = f"{lab['label']} || {lab['gloss']}"
            sample = sampled_members(lab["prompt"], [nodes[i] for i in members_of[lab["super_node_id"]]])
            future_in_input = {_norm(n["surface_form"]) for n in sample
                               if (n.get("first_seen_year") or 0) > year}
            seen_in_input = [_norm(n["surface_form"]) for n in sample
                             if (n.get("first_seen_year") or 0) <= year]
            if future_in_input:
                n_exposed += 1
                exposed.add((year, lab["super_node_id"]))
            hits = {}
            for f in future_in_input:
                if len(f) >= MIN_FORM_LEN and _mentions(text, f):
                    hits[f] = "labeller input"
            for f in future_forms:
                if f not in hits and _mentions(text, f):
                    hits[f] = "whole export"
            n_flagged += bool(hits)
            for f, source in sorted(hits.items()):
                # a sampled member seen by t that contains the term means the
                # corpus had seen it by t, just not as a node of its own
                supported = any(_mentions(s, f) for s in seen_in_input)
                out["flags"].append({
                    "year": year, "super_node_id": lab["super_node_id"], "level": lab["level"],
                    "label": lab["label"], "gloss": lab["gloss"], "matched_form": f,
                    "earliest_first_seen": earliest[f], "source": source,
                    "supported_by_seen_input": supported,
                    "real_leak": (year, lab["super_node_id"], f) in REAL_LEAKS,
                })
        out["per_year"][year] = {"n_labels": len(labels), "n_inputs_with_future_node": n_exposed,
                                 "n_labels_flagged": n_flagged}

    real = {(f["year"], f["super_node_id"]) for f in out["flags"] if f["real_leak"]}
    assert len({(f["year"], f["super_node_id"], f["matched_form"]) for f in out["flags"]
                if f["real_leak"]}) == len(REAL_LEAKS), "a listed leak no longer matches a flag"
    n_labels = sum(v["n_labels"] for v in out["per_year"].values())
    out["summary"] = {"n_labels": n_labels,
                      "n_inputs_with_future_node": sum(v["n_inputs_with_future_node"]
                                                       for v in out["per_year"].values()),
                      "n_labels_flagged": sum(v["n_labels_flagged"] for v in out["per_year"].values()),
                      "n_flags": len(out["flags"]),
                      "n_flags_supported_by_seen_input": sum(f["supported_by_seen_input"]
                                                             for f in out["flags"]),
                      "n_labels_with_real_leak": len(real),
                      "n_leaking_labels_with_future_node_in_input":
                          sum(1 for key in real if key in exposed),
                      "leak_rate_over_labels": rate_with_ci(len(real), n_labels),
                      "leak_rate_over_exposed_inputs": rate_with_ci(len(real & exposed), len(exposed))}
    for year, v in out["per_year"].items():
        print(year, v)
    for f in out["flags"]:
        if not f["real_leak"]:
            continue
        print(f"  {f['year']} {f['super_node_id']} [{f['source']}] '{f['matched_form']}' "
              f"(first seen {f['earliest_first_seen']}) -> {f['real_leak']}\n"
              f"      {f['label']} | {f['gloss']}")
    print(out["summary"])
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
