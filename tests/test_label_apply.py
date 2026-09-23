import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.io import Snapshot  # noqa: E402
from tkh.labeling import (  # noqa: E402
    write_labeling_input, apply_labels_to_hierarchy, unlabelled_super_nodes)


def _snap():
    snap = Snapshot(cutoff_year=2026)
    forms = {f"meth_{i:02d}": f"method {i}" for i in range(8)}
    snap.nodes = {nid: {"id": nid, "type": "method", "surface_form": sf} for nid, sf in forms.items()}
    snap.concept_ids = set(forms)
    return snap


def _hierarchy():
    return {"super_nodes": [
        {"id": "L0_S00001", "level": 0, "parent_id": None,
         "member_ids": [f"meth_{i:02d}" for i in range(8)], "label": None, "gloss": None},
        {"id": "L1_S00001", "level": 1, "parent_id": "L0_S00001",
         "member_ids": ["meth_00", "meth_01", "meth_02", "meth_03"], "label": None, "gloss": None},
        {"id": "L1_S00002", "level": 1, "parent_id": "L0_S00001",
         "member_ids": ["meth_04", "meth_05", "meth_06", "meth_07"], "label": None, "gloss": None},
    ]}


def _labelled_file(tmp_path, hierarchy, snap):
    path = tmp_path / "labeling_output.json"
    entries = write_labeling_input(hierarchy, snap, 2026, path)
    for e in entries:
        e["label"], e["gloss"] = f"label {e['super_node_id']}", f"gloss {e['super_node_id']}"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_labels_apply_to_the_clusters_they_were_written_for(tmp_path):
    snap, h = _snap(), _hierarchy()
    path = _labelled_file(tmp_path, h, snap)
    assert len(unlabelled_super_nodes(h)) == 3
    h, n, stale = apply_labels_to_hierarchy(h, path, snap=snap, year=2026)
    assert n == 3 and stale == [] and unlabelled_super_nodes(h) == []


def test_same_size_different_members_is_stale(tmp_path):
    # same ids, same sizes, members swapped between the two level-1 groups:
    # only the prompt fingerprint can tell
    snap, h = _snap(), _hierarchy()
    path = _labelled_file(tmp_path, h, snap)
    h2 = _hierarchy()
    h2["super_nodes"][1]["member_ids"], h2["super_nodes"][2]["member_ids"] = (
        h2["super_nodes"][2]["member_ids"], h2["super_nodes"][1]["member_ids"])
    h2, n, stale = apply_labels_to_hierarchy(h2, path, snap=snap, year=2026)
    assert sorted(stale) == ["L1_S00001", "L1_S00002"] and n == 1
    assert h2["super_nodes"][1]["gloss"] is None


def test_size_change_is_stale_even_without_snap(tmp_path):
    snap, h = _snap(), _hierarchy()
    path = _labelled_file(tmp_path, h, snap)
    h2 = _hierarchy()
    h2["super_nodes"][1]["member_ids"] = h2["super_nodes"][1]["member_ids"][:3]
    _, n, stale = apply_labels_to_hierarchy(h2, path)
    assert stale == ["L1_S00001"] and n == 2
