"""T5 step 1: dump the labeling prompt input (levels 0-1) for every snapshot.
No LLM call, no automation here, this just writes what a labeller would see.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, SNAPSHOT_CUTOFFS  # noqa: E402
from tkh.labeling import write_labeling_input  # noqa: E402

DATA_PATH = ROOT / "data" / "tkh_collection10.json"
OUT_DIR = ROOT / "outputs"


def main():
    data = load_tkh(DATA_PATH)
    snapshots = build_all_snapshots(data, cutoffs=SNAPSHOT_CUTOFFS)

    for year in sorted(snapshots):
        snap = snapshots[year]
        hpath = OUT_DIR / "snapshots" / str(year) / "hierarchy.json"
        hierarchy = json.loads(hpath.read_text(encoding="utf-8"))

        out_path = OUT_DIR / "snapshots" / str(year) / "labeling_input.json"
        entries = write_labeling_input(hierarchy, snap, year, out_path, levels=(0, 1))
        n0 = sum(1 for e in entries if e["level"] == 0)
        n1 = sum(1 for e in entries if e["level"] == 1)
        print(f"{year}: wrote {len(entries)} entries ({n0} level-0, {n1} level-1) -> {out_path}")


if __name__ == "__main__":
    main()
