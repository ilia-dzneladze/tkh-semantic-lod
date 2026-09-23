import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.io import Snapshot  # noqa: E402
from tkh.labeling import labeller_sample_ids  # noqa: E402
from tkh.eval.stats import wilson_ci, cohen_kappa, binomial_p_greater  # noqa: E402
from tkh.eval.blind import (  # noqa: E402
    make_intruder_items, score_intruder, make_gloss_items, score_gloss_ratings)


def test_wilson_interval_known_values():
    lo, hi = wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    assert wilson_ci(0, 10)[0] == 0.0 and wilson_ci(10, 10)[1] == 1.0
    assert wilson_ci(0, 0) == [None, None]


def test_kappa_and_binomial():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohen_kappa(list("abab"), list("aabb")) == 0.0
    assert binomial_p_greater(10, 10, 1 / 6) < 1e-6
    assert binomial_p_greater(1, 6, 1 / 6) > 0.5


def _world():
    """Two level-0 branches; branch A is all methods, branch B all datasets
    plus a few methods, so type-matched intruders exist on both sides."""
    snap = Snapshot(cutoff_year=2026)
    nodes = {}
    for i in range(40):
        nodes[f"a{i:02d}"] = {"type": "method", "surface_form": f"alpha method {i}"}
    for i in range(30):
        nodes[f"b{i:02d}"] = {"type": "dataset", "surface_form": f"beta dataset {i}"}
    for i in range(10):
        nodes[f"c{i:02d}"] = {"type": "method", "surface_form": f"gamma method {i}"}
    snap.nodes = nodes
    snap.concept_ids = set(nodes)
    a = sorted(n for n in nodes if n[0] == "a")
    b = sorted(n for n in nodes if n[0] in "bc")
    h = {"super_nodes": [
        {"id": "L0_A", "level": 0, "parent_id": None, "member_ids": a, "gloss": "Alpha methods."},
        {"id": "L0_B", "level": 0, "parent_id": None, "member_ids": b, "gloss": "Beta datasets."},
        {"id": "L1_A1", "level": 1, "parent_id": "L0_A", "member_ids": a[:20], "gloss": "First alphas."},
        {"id": "L1_A2", "level": 1, "parent_id": "L0_A", "member_ids": a[20:], "gloss": "Second alphas."},
        {"id": "L1_B1", "level": 1, "parent_id": "L0_B", "member_ids": b, "gloss": "All betas."},
    ]}
    return snap, h


def test_intruders_come_from_another_branch_and_match_a_shown_type():
    snap, h = _world()
    items, key = make_intruder_items(h, snap, {0: 2, 1: 3}, n_null=4, rng=np.random.default_rng(0))
    assert len(items) == len(key) == 9
    branch = {n: sn["id"] for sn in h["super_nodes"] if sn["level"] == 0 for n in sn["member_ids"]}
    for it in items:
        k = key[it["item_id"]]
        opts = k["option_node_ids"]
        assert len(it["options"]) == 6 and len(set(it["options"])) == 6
        assert it["options"] == [snap.nodes[n]["surface_form"] for n in opts]
        intr = opts[k["intruder_index"]]
        shown = [n for i, n in enumerate(opts) if i != k["intruder_index"]]
        assert snap.nodes[intr]["type"] in {snap.nodes[n]["type"] for n in shown}
        if k["kind"] == "real":
            assert len({branch[n] for n in shown}) == 1 and branch[intr] != branch[shown[0]]
    # nothing in the items file says which kind an item is
    assert all(set(it) == {"item_id", "options"} for it in items)


def test_intruder_scoring():
    snap, h = _world()
    items, key = make_intruder_items(h, snap, {1: 3}, n_null=3, rng=np.random.default_rng(1))
    perfect = {i: k["intruder_index"] for i, k in key.items()}
    r = score_intruder(key, perfect)
    assert r["real"]["rate"] == 1.0 and r["null"]["rate"] == 1.0 and r["n_rated"] == 6
    wrong = {i: (k["intruder_index"] + 1) % 6 for i, k in key.items()}
    assert score_intruder(key, wrong)["real"]["rate"] == 0.0


def test_gloss_items_use_only_members_the_labeller_never_saw():
    snap, h = _world()
    items, key = make_gloss_items({2026: h}, {2026: snap}, n_real=3, n_control=2,
                                  rng=np.random.default_rng(0), min_held_out=5)
    by_id = {sn["id"]: sn for sn in h["super_nodes"]}
    form_to_id = {v["surface_form"]: n for n, v in snap.nodes.items()}
    for it in items:
        k = key[it["item_id"]]
        src = by_id[k["member_source_id"]]
        shown_to_labeller = set(labeller_sample_ids(src["member_ids"]))
        ids = [form_to_id[f] for f in it["members"]]
        assert all(n in src["member_ids"] and n not in shown_to_labeller for n in ids)
        assert it["gloss"] == by_id[k["super_node_id"]]["gloss"]
        if k["kind"] == "control":
            assert k["member_source_id"] != k["super_node_id"]
        assert k["premise"] == "The cluster includes " + ", ".join(it["members"]) + "."
    assert all(set(it) == {"item_id", "gloss", "members"} for it in items)


def test_gloss_scoring_and_agreement():
    key = {"G1": {"kind": "real", "nli_label": "entailment"},
           "G2": {"kind": "real", "nli_label": "neutral"},
           "G3": {"kind": "real", "nli_label": "neutral"},
           "G4": {"kind": "control", "nli_label": "contradiction"}}
    ratings = {"G1": "accurate", "G2": "accurate", "G3": "vague", "G4": "wrong"}
    r = score_gloss_ratings(key, ratings, np.random.default_rng(0))
    assert r["real"]["accurate"]["k"] == 2 and r["control"]["wrong"]["rate"] == 1.0
    assert r["over_claim_rate_real"]["k"] == 0
    # of the two real glosses NLI calls not entailed, the rater calls one accurate
    assert r["real_nli_not_entailed_rated_accurate"]["k"] == 1
    assert r["real_nli_not_entailed_rated_accurate"]["n"] == 2
    with pytest.raises(ValueError):
        score_gloss_ratings(key, {**ratings, "G1": "great"}, np.random.default_rng(0))
