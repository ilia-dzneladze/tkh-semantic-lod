"""Load the TKH export and slice it into temporal snapshots (T1)."""
import json
from collections import Counter
from dataclasses import dataclass, field

# see DESIGN_NOTES.md section 1: which node types get clustered
CONCEPT_TYPES = {
    "method", "technique", "task", "problem", "dataset", "metric",
    "component", "cited_work", "future_topic", "claim",
}
CONTEXT_TYPES = {"article", "author"}

SNAPSHOT_CUTOFFS = [2020, 2022, 2024, 2026]


def load_tkh(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Snapshot:
    cutoff_year: int
    nodes: dict = field(default_factory=dict)       # id -> node dict
    hyperedges: list = field(default_factory=list)   # list of edge dicts
    concept_ids: set = field(default_factory=set)    # nodes eligible for clustering
    quality_notes: list = field(default_factory=list)


def _node_present_year(node):
    # first_seen_year, not origin_year. See DESIGN_NOTES.md section 2.
    return node.get("first_seen_year")


def build_snapshot(data, cutoff_year):
    nodes_by_id = {n["id"]: n for n in data["nodes"]}
    snap = Snapshot(cutoff_year=cutoff_year)

    included_edges = []
    referenced_ids = set()
    for e in data["hyperedges"]:
        ey = e.get("year")
        if ey is None or ey > cutoff_year:
            continue
        members = e.get("members", [])
        # data-quality check: does every member exist in the node table?
        missing = [m for m in members if m not in nodes_by_id]
        if missing:
            snap.quality_notes.append(
                f"edge {e['id']} references missing node ids: {missing}")
            continue
        included_edges.append(e)
        referenced_ids.update(members)

    for nid in sorted(referenced_ids):  # sorted so quality_notes order is reproducible
        node = nodes_by_id[nid]
        fsy = _node_present_year(node)
        if fsy is not None and fsy > cutoff_year:
            snap.quality_notes.append(
                f"node {nid} (first_seen_year={fsy}) referenced by an edge "
                f"with year<={cutoff_year}: inconsistent provenance, kept anyway")
        snap.nodes[nid] = node
        if node.get("type") in CONCEPT_TYPES:
            snap.concept_ids.add(nid)

    snap.hyperedges = included_edges
    return snap


def build_all_snapshots(data, cutoffs=SNAPSHOT_CUTOFFS):
    return {t: build_snapshot(data, t) for t in cutoffs}


def describe_snapshot(snap):
    type_dist = Counter(n.get("type") for n in snap.nodes.values())
    arity_dist = Counter(len(e.get("members", [])) for e in snap.hyperedges)
    rel_dist = Counter(e.get("relation_type") for e in snap.hyperedges)
    node_years = [n.get("year") for n in snap.nodes.values() if n.get("year") is not None]
    return {
        "cutoff_year": snap.cutoff_year,
        "n_nodes": len(snap.nodes),
        "n_concept_nodes": len(snap.concept_ids),
        "n_hyperedges": len(snap.hyperedges),
        "node_type_distribution": dict(type_dist),
        "relation_type_distribution": dict(rel_dist),
        "hyperedge_arity_distribution": dict(sorted(arity_dist.items())),
        "max_arity": max(arity_dist) if arity_dist else 0,
        "min_arity": min(arity_dist) if arity_dist else 0,
        "node_year_span": [min(node_years), max(node_years)] if node_years else None,
        "n_quality_notes": len(snap.quality_notes),
        "quality_notes_sample": snap.quality_notes[:10],
    }
