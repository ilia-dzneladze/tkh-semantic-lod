"""T4: hyper-edge collapse rule.

m = n (all endpoints in one super-node) -> internal, dropped from the
coarse graph, counted as an internal-density stat instead.
Spans exactly 2 super-nodes -> pairwise edge.
Spans 3+ super-nodes -> stays a genuine hyperedge, not exploded into a
clique of pairwise edges.

Full rationale and the clique-blowup numbers: DESIGN_NOTES.md section 9.

Context nodes (article/author) pass through as singleton "super-nodes"
identified by their own id, so collapse_hyperedge has one code path.
"""
from collections import Counter, defaultdict


def _full_mapping(mapping, snap):
    """Extend a concept-node cluster mapping with identity entries for
    context nodes (article/author), so every member id in a hyperedge has
    a supernode to resolve to."""
    full = dict(mapping)
    for nid in snap.nodes:
        if nid not in full:
            full[nid] = nid
    return full


def collapse_hyperedge(edge, mapping):
    members = edge.get("members", [])
    targets = [mapping[m] for m in members]
    counts = Counter(targets)
    distinct = list(counts.keys())

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
    """Apply collapse_hyperedge to every edge, aggregate duplicates (same
    super-node-set from different original edges) into one weighted coarse
    edge, and separately track per-super-node internal-edge statistics."""
    full_mapping = _full_mapping(mapping, snap)

    agg = defaultdict(lambda: {
        "weight": 0, "source_edge_ids": [], "relation_type_counts": Counter(),
        "total_original_arity": 0,
    })
    internal_stats = defaultdict(lambda: {"edge_count": 0, "member_count": 0})

    for e in hyperedges:
        result = collapse_hyperedge(e, full_mapping)
        if result["kind"] == "internal":
            if result["supernode"] is None:
                continue
            s = internal_stats[result["supernode"]]
            s["edge_count"] += 1
            s["member_count"] += result["collapsed_member_count"]
        else:
            key = tuple(result["members"])
            a = agg[key]
            a["weight"] += 1
            a["source_edge_ids"].append(result["source_edge_id"])
            a["relation_type_counts"][result["relation_type"]] += 1
            a["total_original_arity"] += result["original_arity"]

    coarse_edges = []
    for key, info in agg.items():
        coarse_edges.append({
            "members": list(key),
            "arity": len(key),
            "weight": info["weight"],
            "source_edge_ids": info["source_edge_ids"],
            "relation_type_counts": dict(info["relation_type_counts"]),
        })

    n_pairwise = sum(1 for c in coarse_edges if c["arity"] == 2)
    n_hyper = sum(1 for c in coarse_edges if c["arity"] > 2)
    n_internal_edges = sum(s["edge_count"] for s in internal_stats.values())
    summary = {
        "n_coarse_edges": len(coarse_edges),
        "n_pairwise": n_pairwise,
        "n_genuine_hyperedges": n_hyper,
        "n_internal_edges_dropped": n_internal_edges,
        "n_original_edges": len(hyperedges),
    }
    return coarse_edges, dict(internal_stats), summary


def clique_explosion_comparison(coarse_edges):
    """Pairwise edges a clique expansion would add vs. the 1 coarse
    hyperedge kept per group. See DESIGN_NOTES.md section 9."""
    total_clique_pairs = 0
    total_native_edges = 0
    for c in coarse_edges:
        if c["arity"] > 2:
            k = c["arity"]
            total_clique_pairs += k * (k - 1) // 2
            total_native_edges += 1
    return {
        "genuine_hyperedges_kept": total_native_edges,
        "pairwise_edges_clique_alternative_would_add": total_clique_pairs,
        "blowup_factor": (
            total_clique_pairs / total_native_edges if total_native_edges else 0.0
        ),
    }
