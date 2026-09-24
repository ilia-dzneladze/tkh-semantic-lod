import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.collapse import collapse_hyperedge, build_coarse_hyperedges


def _snap(node_ids):
    return SimpleNamespace(nodes={nid: {} for nid in node_ids})


def test_fully_internal_m_equals_n():
    edge = {"id": "e1", "members": ["a", "b", "c"], "relation_type": "solves"}
    mapping = {"a": "S1", "b": "S1", "c": "S1"}
    r = collapse_hyperedge(edge, mapping)
    assert r["kind"] == "internal"
    assert r["supernode"] == "S1"
    assert r["collapsed_member_count"] == 3
    assert r["original_arity"] == 3


def test_spans_exactly_two_is_pairwise():
    edge = {"id": "e2", "members": ["a", "b", "c"], "relation_type": "evaluated_on"}
    mapping = {"a": "S1", "b": "S1", "c": "S2"}
    r = collapse_hyperedge(edge, mapping)
    assert r["kind"] == "pairwise"
    assert r["arity"] == 2
    assert sorted(r["members"]) == ["S1", "S2"]
    assert r["member_counts"] == {"S1": 2, "S2": 1}


def test_spans_three_or_more_is_genuine_hyperedge_not_clique():
    edge = {"id": "e3", "members": ["a", "b", "c", "d"], "relation_type": "addresses"}
    mapping = {"a": "S1", "b": "S2", "c": "S3", "d": "S3"}
    r = collapse_hyperedge(edge, mapping)
    assert r["kind"] == "hyperedge"
    assert r["arity"] == 3  # 3 distinct supernodes, NOT exploded into C(3,2)=3 pairwise edges
    assert sorted(r["members"]) == ["S1", "S2", "S3"]


def test_context_nodes_pass_through_as_singletons():
    node_ids = ["a", "art_1"]
    snap = _snap(node_ids)
    edges = [{"id": "e4", "members": ["a", "art_1"], "relation_type": "presents"}]
    mapping = {"a": "S1"}  # art_1 not in mapping -> should become identity
    coarse, internal, summary = build_coarse_hyperedges(edges, mapping, snap)
    assert len(coarse) == 1
    assert sorted(coarse[0]["members"]) == ["S1", "art_1"]


def test_duplicate_collapses_aggregate_with_weight():
    node_ids = ["a", "b", "c", "d"]
    snap = _snap(node_ids)
    edges = [
        {"id": "e5", "members": ["a", "c"], "relation_type": "cites"},
        {"id": "e6", "members": ["b", "d"], "relation_type": "extends"},
    ]
    mapping = {"a": "S1", "b": "S1", "c": "S2", "d": "S2"}
    coarse, internal, summary = build_coarse_hyperedges(edges, mapping, snap)
    assert len(coarse) == 1
    assert coarse[0]["weight"] == 2
    assert set(coarse[0]["relation_type_counts"]) == {"cites", "extends"}
    assert summary["n_original_edges"] == 2
    assert summary["n_coarse_edges"] == 1


def test_no_original_members_are_silently_dropped():
    # every member of every edge must be resolvable via mapping/identity
    node_ids = ["a", "b", "art_1", "auth_1"]
    snap = _snap(node_ids)
    edges = [{"id": "e7", "members": ["a", "b", "art_1", "auth_1"], "relation_type": "authored_by"}]
    mapping = {"a": "S1", "b": "S2"}
    coarse, internal, summary = build_coarse_hyperedges(edges, mapping, snap)
    assert coarse[0]["arity"] == 4  # S1, S2, art_1, auth_1 all distinct
    total_members_seen = sum(coarse[0]["members"].count(m) for m in set(coarse[0]["members"]))
    assert total_members_seen == 4

