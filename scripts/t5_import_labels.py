"""T5 step 2: merge a labeller's reply into labeling_output.json.

Usage: t5_import_labels.py DIR, where DIR holds labels_<year>.json files,
each a JSON array of {"super_node_id", "label", "gloss"} as returned by the
labeller for that year's labeling_input.json. Checks that the ids match the
input exactly and that the word limits hold, then writes
outputs/snapshots/<year>/labeling_output.json. Run t5_apply_labels.py
afterwards to merge the labels into hierarchy.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import SNAPSHOT_CUTOFFS  # noqa: E402

MAX_LABEL_WORDS, MAX_GLOSS_WORDS = 6, 25


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: t5_import_labels.py DIR")
    src_dir = Path(sys.argv[1])
    problems = []
    for year in SNAPSHOT_CUTOFFS:
        snap_dir = ROOT / "outputs" / "snapshots" / str(year)
        inputs = json.loads((snap_dir / "labeling_input.json").read_text(encoding="utf-8"))
        reply = json.loads((src_dir / f"labels_{year}.json").read_text(encoding="utf-8"))
        by_id = {r["super_node_id"]: r for r in reply}
        want = [e["super_node_id"] for e in inputs]
        if len(by_id) != len(reply) or set(by_id) != set(want):
            problems.append(f"{year}: ids differ (missing {sorted(set(want) - set(by_id))}, "
                            f"extra {sorted(set(by_id) - set(want))}, duplicates {len(reply) - len(by_id)})")
            continue
        for e in inputs:
            r = by_id[e["super_node_id"]]
            label, gloss = (r.get("label") or "").strip(), (r.get("gloss") or "").strip()
            if not label or not gloss:
                problems.append(f"{year} {e['super_node_id']}: empty label or gloss")
            if len(label.split()) > MAX_LABEL_WORDS or len(gloss.split()) > MAX_GLOSS_WORDS:
                problems.append(f"{year} {e['super_node_id']}: over word limit "
                                f"({len(label.split())}/{len(gloss.split())} words)")
            e["label"], e["gloss"] = label, gloss
        (snap_dir / "labeling_output.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
        print(f"{year}: imported {len(inputs)} labels")
    if problems:
        print("\n".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
