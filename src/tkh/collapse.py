"""T4: the hyperedge collapse rule.

Map each hyperedge's members to their super-nodes, then:
  all in one super-node   -> internal: dropped, counted per super-node
  exactly two super-nodes -> a pairwise edge
  three or more           -> stays one hyperedge, not exploded into pairs
Why, and what it loses: DESIGN_NOTES.md section 9.

Articles and authors aren't clustered, so they pass through as singleton
super-nodes under their own id.
"""
from collections import Counter, defaultdict

from tkh.hypergraph import clique_expansion


def collapse_hyperedge(edge, mapping):
    members = edge.get("members", [])
    counts = Counter(mapping[m] for m in members)
    distinct = list(counts)
    if len(distinct) <= 1:
        return {
            "kind": "internal",
            "supernode": distinct[0] if distinct else None,
            "original_arity": len(members),
            "collapsed_member_count": counts[distinct[0]] if distinct else 0,
        }
    return {
        "kind": "pairwise" if len(distinct) == 2 else "hyperedge",
        "members": sorted(distinct),
        "arity": len(distinct),
        "original_arity": len(members),
        "member_counts": dict(counts),
        "relation_type": edge.get("relation_type"),
        "source_edge_id": edge.get("id"),
    }


def build_coarse_hyperedges(hyperedges, mapping, snap):
    """Collapse every edge. Edges that land on the same set of super-nodes
    merge into one coarse edge with a count (weight). Returns (coarse
    edges, internal-edge stats per super-node, summary counts)."""
    full_mapping = {nid: mapping.get(nid, nid) for nid in snap.nodes}
    merged = defaultdict(lambda: {"weight": 0, "source_edge_ids": [], "relation_type_counts": Counter()})
    internal_stats = defaultdict(lambda: {"edge_count": 0, "member_count": 0})

    for e in hyperedges:
        result = collapse_hyperedge(e, full_mapping)
        if result["kind"] == "internal":
            if result["supernode"] is not None:
                s = internal_stats[result["supernode"]]
                s["edge_count"] += 1
                s["member_count"] += result["collapsed_member_count"]
        else:
            m = merged[tuple(result["members"])]
            m["weight"] += 1
            m["source_edge_ids"].append(result["source_edge_id"])
            m["relation_type_counts"][result["relation_type"]] += 1

    coarse_edges = [{"members": list(key), "arity": len(key), "weight": m["weight"],
                     "source_edge_ids": m["source_edge_ids"],
                     "relation_type_counts": dict(m["relation_type_counts"])}
                    for key, m in merged.items()]
    summary = {
        "n_coarse_edges": len(coarse_edges),
        "n_pairwise": sum(1 for c in coarse_edges if c["arity"] == 2),
        "n_genuine_hyperedges": sum(1 for c in coarse_edges if c["arity"] > 2),
        "n_internal_edges_dropped": sum(s["edge_count"] for s in internal_stats.values()),
        "n_original_edges": len(hyperedges),
    }
    return coarse_edges, dict(internal_stats), summary


def coarse_structural_affinity(coarse_edges, super_ids):
    """Structural affinity between super-nodes from the collapsed edges: the
    same weighted clique expansion as at the fine level, over each coarse
    edge's super-node members (context singletons dropped), scaled by the
    edge's weight. Rows follow super_ids. DESIGN_NOTES.md section 9."""
    idx = {s: i for i, s in enumerate(super_ids)}
    groups = [[idx[m] for m in c["members"] if m in idx] for c in coarse_edges]
    return clique_expansion(groups, len(super_ids), weights=[c["weight"] for c in coarse_edges])
