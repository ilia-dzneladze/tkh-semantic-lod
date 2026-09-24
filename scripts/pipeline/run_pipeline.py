"""End-to-end: load TKH export -> snapshots (T1) -> multi-level laminar
hierarchy per snapshot with temporal tracking (T2/T3/T4) -> hierarchy.json
per snapshot + temporal_events.json. Labels/glosses (T5) and metrics (T6)
are filled in by later scripts.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, SNAPSHOT_CUTOFFS, DATA_PATH  # noqa: E402
from tkh.pipeline import run_all_snapshots, build_hierarchy_json  # noqa: E402

OUT_DIR = ROOT / "outputs"


def main():
    t0 = time.time()
    data = load_tkh(DATA_PATH)
    snapshots = build_all_snapshots(data, cutoffs=SNAPSHOT_CUTOFFS)
    print(f"[{time.time()-t0:.1f}s] built {len(snapshots)} snapshots")

    result = run_all_snapshots(snapshots)
    print(f"[{time.time()-t0:.1f}s] clustering + temporal tracking done")

    for level_idx, events in result["events_by_level"].items():
        from collections import Counter
        counts = Counter(e["type"] for e in events)
        print(f"  level {level_idx} events: {dict(counts)}")

    for year in result["years"]:
        h = build_hierarchy_json(result, year, snapshots[year])
        snap_dir = OUT_DIR / "snapshots" / str(year)
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / "hierarchy.json").write_text(json.dumps(h, indent=2), encoding="utf-8")
        print(f"[{time.time()-t0:.1f}s] wrote {snap_dir / 'hierarchy.json'} "
              f"({len(h['super_nodes'])} super-nodes)")

    all_events = []
    for level_idx, events in result["events_by_level"].items():
        for e in events:
            e2 = dict(e)
            e2["level"] = level_idx
            all_events.append(e2)
    (OUT_DIR / "temporal_events.json").write_text(json.dumps(all_events, indent=2), encoding="utf-8")
    print(f"[{time.time()-t0:.1f}s] wrote {OUT_DIR / 'temporal_events.json'} "
          f"({len(all_events)} events)")

    print(f"[{time.time()-t0:.1f}s] done")


if __name__ == "__main__":
    main()
