"""T5 step 2: merge labeling_output.json (label+gloss filled in) back into
each snapshot's hierarchy.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import SNAPSHOT_CUTOFFS  # noqa: E402
from tkh.labeling import apply_labels_to_hierarchy  # noqa: E402

OUT_DIR = ROOT / "outputs"


def main():
    for year in SNAPSHOT_CUTOFFS:
        snap_dir = OUT_DIR / "snapshots" / str(year)
        hpath = snap_dir / "hierarchy.json"
        labeled_path = snap_dir / "labeling_output.json"

        if not labeled_path.exists():
            print(f"{year}: no labeling_output.json yet, skipping")
            continue

        hierarchy = json.loads(hpath.read_text(encoding="utf-8"))
        hierarchy, n_applied = apply_labels_to_hierarchy(hierarchy, labeled_path)
        hpath.write_text(json.dumps(hierarchy, indent=2), encoding="utf-8")

        n_expected = sum(1 for sn in hierarchy["super_nodes"] if sn["level"] in (0, 1))
        print(f"{year}: applied {n_applied}/{n_expected} labels into {hpath}")


if __name__ == "__main__":
    main()
