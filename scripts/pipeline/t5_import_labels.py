"""T5 step 2: check a labeller's replies and write them into a label set.

Usage: t5_import_labels.py REPLY_FOLDER [--set DIR]

REPLY_FOLDER holds labels_<year>.json for each snapshot, each the
labeller's reply to that year's prompts: a JSON array of
{"super_node_id", "label", "gloss"}, as pure JSON or wrapped in a code
fence. Every prompt must get exactly one answer and the word limits must
hold (label <= 6 words, gloss <= 25). All or nothing: if any year has a
problem, no file is written. Writes DIR/<year>/labeling_output.json (DIR
defaults to the shipped set, outputs/snapshots). Then apply the set with
t5_apply_labels.py --labels DIR. See DESIGN_NOTES.md section 24.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import SNAPSHOT_CUTOFFS, OUTPUTS  # noqa: E402
from tkh.replies import extract_json  # noqa: E402

MAX_LABEL_WORDS, MAX_GLOSS_WORDS = 6, 25


def check_year(inputs, reply, year):
    """Fill the inputs with the reply's labels; return (entries, problems)."""
    problems = []
    by_id = {r.get("super_node_id"): r for r in reply if isinstance(r, dict)}
    want = [e["super_node_id"] for e in inputs]
    if len(by_id) != len(reply) or set(by_id) != set(want):
        return None, [f"{year}: ids differ (missing {sorted(set(want) - set(by_id))[:5]}, "
                      f"extra {sorted(set(by_id) - set(want))[:5]}, duplicates {len(reply) - len(by_id)})"]
    for e in inputs:
        r = by_id[e["super_node_id"]]
        label, gloss = (r.get("label") or "").strip(), (r.get("gloss") or "").strip()
        if not label or not gloss:
            problems.append(f"{year} {e['super_node_id']}: empty label or gloss")
        if len(label.split()) > MAX_LABEL_WORDS or len(gloss.split()) > MAX_GLOSS_WORDS:
            problems.append(f"{year} {e['super_node_id']}: over word limit "
                            f"({len(label.split())}/{len(gloss.split())} words)")
        e["label"], e["gloss"] = label, gloss
    return inputs, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replies", type=Path, help="folder with labels_<year>.json replies")
    ap.add_argument("--set", type=Path, default=OUTPUTS / "snapshots",
                    help="label set to write into (default: the shipped set)")
    args = ap.parse_args()

    problems, ready = [], {}
    for year in SNAPSHOT_CUTOFFS:
        inputs = json.loads((args.set / str(year) / "labeling_input.json").read_text(encoding="utf-8"))
        try:
            reply = extract_json((args.replies / f"labels_{year}.json").read_text(encoding="utf-8"), list)
        except (OSError, ValueError) as e:
            problems.append(f"{year}: {e}")
            continue
        entries, year_problems = check_year(inputs, reply, year)
        problems += year_problems
        if entries is not None:
            ready[year] = entries
    if problems:
        sys.exit("nothing written:\n" + "\n".join(problems))
    for year, entries in ready.items():
        (args.set / str(year) / "labeling_output.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print(f"{year}: imported {len(entries)} labels")


if __name__ == "__main__":
    main()
