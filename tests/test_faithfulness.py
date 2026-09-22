import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.labeling import labeller_sample_ids
from tkh.eval.faithfulness import held_out_member_ids, build_premise


class _Snap:
    def __init__(self, ids):
        self.nodes = {nid: {"surface_form": f"form-{nid}"} for nid in ids}


def test_held_out_is_disjoint_from_labeller_input():
    members = sorted(f"n{i:03d}" for i in range(60))
    shown = set(labeller_sample_ids(members))
    held = held_out_member_ids(members)
    assert shown.isdisjoint(held)
    assert shown | set(held) == set(members)


def test_premise_only_uses_held_out_members():
    members = sorted(f"n{i:03d}" for i in range(60))
    snap = _Snap(members)
    held = held_out_member_ids(members)
    premise = build_premise(snap, held, max_members=15, rng=np.random.default_rng(0))
    for nid in labeller_sample_ids(members):
        assert f"form-{nid}," not in premise and f"form-{nid}." not in premise


def test_small_cluster_has_no_held_out():
    members = [f"n{i}" for i in range(20)]
    assert held_out_member_ids(members) == []
