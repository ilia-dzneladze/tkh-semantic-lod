"""T5 step 2: merge labeling_output.json (label+gloss filled in) back into
each snapshot's hierarchy.json.

run_pipeline.py writes every label as null, so this has to run after it
before anything that reads labels (t6_evaluate.py, label_routing.py).
A label is only applied if the cluster it was written for is still the
same cluster; anything else is reported as stale and the script exits
non-zero. See DESIGN_NOTES.md section 19.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, SNAPSHOT_CUTOFFS  # noqa: E402
from tkh.labeling import apply_labels_to_hierarchy  # noqa: E402

OUT_DIR = ROOT / "outputs"
DATA_PATH = ROOT / "data" / "tkh_collection10.json"


def main():
    data = load_tkh(DATA_PATH)
    problems = []
    for year in SNAPSHOT_CUTOFFS:
        snap_dir = OUT_DIR / "snapshots" / str(year)
        hpath = snap_dir / "hierarchy.json"
        labeled_path = snap_dir / "labeling_output.json"

        if not labeled_path.exists():
            print(f"{year}: no labeling_output.json yet, skipping")
            continue

        hierarchy = json.loads(hpath.read_text(encoding="utf-8"))
        hierarchy, n_applied, stale = apply_labels_to_hierarchy(
            hierarchy, labeled_path, snap=build_snapshot(data, year), year=year)
        hpath.write_text(json.dumps(hierarchy, indent=2), encoding="utf-8")

        n_expected = sum(1 for sn in hierarchy["super_nodes"] if sn["level"] in (0, 1))
        print(f"{year}: applied {n_applied}/{n_expected} labels into {hpath}")
        if stale:
            problems.append(f"{year}: {len(stale)} labels written for a different cluster "
                            f"than the one now under that id, not applied: {stale[:5]}")

    if problems:
        print("\n".join(problems))
        sys.exit("stale labels: the clustering changed since labelling, relabel before evaluating")


if __name__ == "__main__":
    main()
