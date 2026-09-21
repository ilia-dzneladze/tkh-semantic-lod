"""FIXES.md iteration 3: how much do event counts (birth/death/merge/split/
grow/shrink/stable) move as STABLE_JACCARD, MATCH_THRESHOLD, and
SIZE_CHANGE_RATIO vary? T3 only rematches the already-computed per-snapshot
clusterings, it doesn't re-embed or re-cluster, so this sweep just reloads
each snapshot's hierarchy.json and reruns classify_events with different
threshold globals -- no pipeline rebuild needed. Writes
outputs/temporal_threshold_sweep.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tkh.temporal as temporal  # noqa: E402
from tkh.io import SNAPSHOT_CUTOFFS  # noqa: E402

OUT_DIR = ROOT / "outputs"
N_LEVELS = 3
SHIPPED = {"MATCH_THRESHOLD": 0.15, "STABLE_JACCARD": 0.5, "SIZE_CHANGE_RATIO": 0.2}


def load_clusters_by_level_and_year():
    out = {}
    for level in range(N_LEVELS):
        out[level] = {}
        for year in SNAPSHOT_CUTOFFS:
            h = json.loads((OUT_DIR / "snapshots" / str(year) / "hierarchy.json").read_text(encoding="utf-8"))
            out[level][year] = {
                sn["id"]: frozenset(sn["member_ids"])
                for sn in h["super_nodes"] if sn["level"] == level
            }
    return out


def run_transitions(clusters_by_year, prefix):
    """Mirrors track_across_snapshots, minus the first-year birth bookkeeping
    (those aren't affected by any threshold, excluded from the sweep so the
    counts only reflect what classify_events actually decides)."""
    years = sorted(clusters_by_year)
    next_id_counter = [0]
    prev_clusters = clusters_by_year[years[0]]
    prev_persistent = {}
    for lab in prev_clusters:
        next_id_counter[0] += 1
        prev_persistent[lab] = f"{prefix}S{next_id_counter[0]:05d}"

    events_by_transition = []
    for year in years[1:]:
        curr_clusters = clusters_by_year[year]
        curr_persistent, events = temporal.classify_events(
            prev_clusters, curr_clusters, prev_persistent, next_id_counter, prefix=prefix)
        events_by_transition.append((year, events))
        prev_clusters, prev_persistent = curr_clusters, curr_persistent
    return events_by_transition


def event_counts(clusters_by_level):
    total = Counter()
    per_level = {}
    for level in range(N_LEVELS):
        level_counter = Counter()
        for year, events in run_transitions(clusters_by_level[level], prefix=f"L{level}_"):
            for e in events:
                level_counter[e["type"]] += 1
        per_level[level] = dict(level_counter)
        total.update(level_counter)
    return dict(total), per_level


def set_thresholds(**overrides):
    for k, v in SHIPPED.items():
        setattr(temporal, k, overrides.get(k, v))


def main():
    clusters_by_level = load_clusters_by_level_and_year()

    set_thresholds()
    shipped_total, shipped_per_level = event_counts(clusters_by_level)
    print("shipped thresholds, total:", shipped_total)
    for lvl, c in shipped_per_level.items():
        print(f"  level {lvl}: {c}")

    grids = {
        "MATCH_THRESHOLD": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "STABLE_JACCARD": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "SIZE_CHANGE_RATIO": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    }

    results = {}
    for param, values in grids.items():
        rows = []
        for v in values:
            set_thresholds(**{param: v})
            total, per_level = event_counts(clusters_by_level)
            rows.append({"value": v, "total": total, "per_level": per_level})
        results[param] = rows
        print()
        print(param)
        for r in rows:
            print(f"  {r['value']:.2f}: {r['total']}")

    set_thresholds()  # restore shipped values

    out = {"shipped": SHIPPED, "shipped_total": shipped_total,
           "shipped_per_level": shipped_per_level, "sweeps": results}
    (OUT_DIR / "temporal_threshold_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'temporal_threshold_sweep.json'}")


if __name__ == "__main__":
    main()
