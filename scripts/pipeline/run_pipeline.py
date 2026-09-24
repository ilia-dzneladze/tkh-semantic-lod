"""T2-T4: build the hierarchy for every snapshot, track it across
snapshots, and write outputs/snapshots/<year>/hierarchy.json and
outputs/temporal_events.json. Labels (T5) are left null for
t5_apply_labels.py, and metrics (T6) come later.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, DATA_PATH, OUTPUTS  # noqa: E402
from tkh.pipeline import run_all_snapshots, build_hierarchy_json  # noqa: E402


def main():
    t0 = time.time()
    data = load_tkh(DATA_PATH)
    snapshots = build_all_snapshots(data)
    print(f"[{time.time()-t0:.1f}s] built {len(snapshots)} snapshots")

    result = run_all_snapshots(snapshots)
    print(f"[{time.time()-t0:.1f}s] clustering + temporal tracking done")

    for level_idx, events in result["events_by_level"].items():
        print(f"  level {level_idx} events: {dict(Counter(e['type'] for e in events))}")

    for year in result["years"]:
        h = build_hierarchy_json(result, year, snapshots[year])
        snap_dir = OUTPUTS / "snapshots" / str(year)
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "hierarchy.json").write_text(json.dumps(h, indent=2), encoding="utf-8")
        print(f"[{time.time()-t0:.1f}s] wrote {snap_dir / 'hierarchy.json'} "
              f"({len(h['super_nodes'])} super-nodes)")

    all_events = [{**e, "level": level_idx}
                  for level_idx, events in result["events_by_level"].items() for e in events]
    (OUTPUTS / "temporal_events.json").write_text(json.dumps(all_events, indent=2), encoding="utf-8")
    print(f"[{time.time()-t0:.1f}s] wrote {OUTPUTS / 'temporal_events.json'} "
          f"({len(all_events)} events)")

    print(f"[{time.time()-t0:.1f}s] done")


if __name__ == "__main__":
    main()
