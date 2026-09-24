"""T1: slice the TKH export into snapshots and describe each one.
Writes outputs/t1_snapshot_stats.json. Also prints how much of the pair
weight comes from hyperedges with more than 10 concept members, with and
without the 1/(k-1) weighting (DESIGN_NOTES.md sections 3 and 4)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, describe_snapshot, DATA_PATH, OUTPUTS  # noqa: E402
from tkh.hypergraph import high_arity_weight_share  # noqa: E402

OUT_PATH = OUTPUTS / "t1_snapshot_stats.json"


def main():
    data = load_tkh(DATA_PATH)
    snapshots = build_all_snapshots(data)

    report = {}
    prev_ids = None
    for t in sorted(snapshots):
        snap = snapshots[t]
        stats = describe_snapshot(snap)
        if prev_ids is not None:
            cur_ids = set(snap.nodes)
            stats["new_nodes_since_prev"] = len(cur_ids - prev_ids)
            stats["node_growth_pct"] = round(
                100 * len(cur_ids - prev_ids) / max(len(prev_ids), 1), 1)
        prev_ids = set(snap.nodes)
        report[str(t)] = stats

        print(f"--- snapshot <= {t} ---")
        print(f"  nodes: {stats['n_nodes']} (concept-eligible: {stats['n_concept_nodes']})")
        print(f"  hyperedges: {stats['n_hyperedges']}  "
              f"(arity range {stats['min_arity']}-{stats['max_arity']})")
        print(f"  node types: {stats['node_type_distribution']}")
        if stats["n_quality_notes"]:
            print(f"  quality notes: {stats['n_quality_notes']} "
                  f"(sample: {stats['quality_notes_sample'][:2]})")
        print(f"  nodes first seen after the cutoff, kept anyway: "
              f"{stats['n_nodes_first_seen_after_cutoff']} "
              f"({stats['n_concept_nodes_first_seen_after_cutoff']} concept nodes)")
        unweighted, weighted = high_arity_weight_share(snap)
        print(f"  pair weight from hyperedges with >10 concept members: "
              f"{unweighted:.0%} unweighted, {weighted:.0%} with 1/(k-1)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
