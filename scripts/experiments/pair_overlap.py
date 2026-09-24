"""How often concept pairs recur across hyperedges and across papers, per
snapshot. Context: DESIGN_NOTES.md section 13 (why the structure term
generalises weakly to unseen papers). Writes outputs/pair_overlap.json.
Changes nothing in the shipped pipeline.
"""
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, DATA_PATH  # noqa: E402

YEARS = [2020, 2022, 2024, 2026]


def pair_stats(snap):
    n_edges = defaultdict(int)
    papers = defaultdict(set)
    weight = defaultdict(float)
    for e in snap.hyperedges:
        mem = sorted({m for m in e["members"] if m in snap.concept_ids})
        if len(mem) < 2:
            continue
        for pair in combinations(mem, 2):
            n_edges[pair] += 1
            papers[pair].add(e["provenance"]["article_id"])
            weight[pair] += 1 / (len(mem) - 1)  # same weighting as build_structural_affinity
    total_w = sum(weight.values())
    multi_edge = [p for p in n_edges if n_edges[p] > 1]
    multi_paper = [p for p in papers if len(papers[p]) > 1]
    return {
        "n_pairs": len(n_edges),
        "share_pairs_in_multiple_edges": len(multi_edge) / len(n_edges),
        "share_weight_in_multiple_edges": sum(weight[p] for p in multi_edge) / total_w,
        "share_pairs_in_multiple_papers": len(multi_paper) / len(n_edges),
        "share_weight_in_multiple_papers": sum(weight[p] for p in multi_paper) / total_w,
    }


def main():
    data = load_tkh(DATA_PATH)
    out = {str(y): pair_stats(build_snapshot(data, y)) for y in YEARS}
    (ROOT / "outputs" / "pair_overlap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for y, s in out.items():
        print(f"{y}: {s['n_pairs']} pairs, in >1 edge {s['share_pairs_in_multiple_edges']:.1%} "
              f"(weight {s['share_weight_in_multiple_edges']:.1%}), in >1 paper "
              f"{s['share_pairs_in_multiple_papers']:.1%} (weight {s['share_weight_in_multiple_papers']:.1%})")


if __name__ == "__main__":
    main()
