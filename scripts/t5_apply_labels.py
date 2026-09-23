"""T5 step 3: put a label set's labels onto each snapshot's hierarchy.json.

Usage: t5_apply_labels.py [--labels DIR]

DIR is a label set: DIR/<year>/labeling_output.json, plus
prompt_template.txt if it was written from a custom template. The default
is the shipped set in outputs/snapshots.

run_pipeline.py writes every label as null, so this has to run after it
and before anything that reads labels (t6_evaluate.py, blind_eval.py).
A label is only applied if the prompt rebuilt from the cluster's current
members, with the template the set was made with, matches the prompt the
label was written from; anything else is reported as stale and the script
exits non-zero. See DESIGN_NOTES.md sections 19 and 24.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, SNAPSHOT_CUTOFFS  # noqa: E402
from tkh.labeling import apply_labels_to_hierarchy, label_set_template  # noqa: E402

OUT_DIR = ROOT / "outputs"
DATA_PATH = ROOT / "data" / "tkh_collection10.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=OUT_DIR / "snapshots",
                    help="label set directory (default: the shipped set, outputs/snapshots)")
    args = ap.parse_args()
    try:
        template = label_set_template(args.labels)
    except ValueError as e:
        sys.exit(f"{args.labels}: bad prompt_template.txt: {e}")

    data = load_tkh(DATA_PATH)
    problems = []
    for year in SNAPSHOT_CUTOFFS:
        hpath = OUT_DIR / "snapshots" / str(year) / "hierarchy.json"
        labeled_path = args.labels / str(year) / "labeling_output.json"

        if not labeled_path.exists():
            problems.append(f"{year}: no {labeled_path}")
            continue

        hierarchy = json.loads(hpath.read_text(encoding="utf-8"))
        hierarchy, n_applied, stale = apply_labels_to_hierarchy(
            hierarchy, labeled_path, snap=build_snapshot(data, year), year=year, template=template)
        hpath.write_text(json.dumps(hierarchy, indent=2), encoding="utf-8")

        n_expected = sum(1 for sn in hierarchy["super_nodes"] if sn["level"] in (0, 1))
        print(f"{year}: applied {n_applied}/{n_expected} labels from {labeled_path}")
        if stale:
            problems.append(f"{year}: {len(stale)} labels written for a different cluster "
                            f"than the one now under that id, not applied: {stale[:5]}")

    if problems:
        print("\n".join(problems))
        sys.exit("labels not fully applied: relabel, or point --labels at a set made for these clusters")


if __name__ == "__main__":
    main()
