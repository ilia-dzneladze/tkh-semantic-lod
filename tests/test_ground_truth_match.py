import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.io import Snapshot  # noqa: E402
from tkh.eval.extrinsic import match_ground_truth_methods  # noqa: E402


def _snap(forms):
    """forms: {node_id: surface_form}, all typed as methods so they land in
    the surface index."""
    snap = Snapshot(cutoff_year=2026)
    snap.nodes = {nid: {"id": nid, "type": "method", "surface_form": sf}
                  for nid, sf in forms.items()}
    snap.concept_ids = set(forms)
    return snap


def test_exact_match_wins():
    snap = _snap({"a": "MACE", "b": "MACE-MP-0"})
    m = match_ground_truth_methods(snap, ["MACE"])
    assert m["MACE"]["match_type"] == "exact"
    assert m["MACE"]["node_ids"] == ["a"]


def test_substring_match_both_directions():
    snap = _snap({"a": "Universal HamGNN Hamiltonian model", "b": "DeepH"})
    m = match_ground_truth_methods(snap, ["HamGNN", "xDeepH"])
    assert m["HamGNN"]["match_type"] == "substring" and m["HamGNN"]["node_ids"] == ["a"]
    assert m["xDeepH"]["match_type"] == "substring" and m["xDeepH"]["node_ids"] == ["b"]


def test_short_surface_forms_never_match_by_substring():
    # element symbols are substrings of many method names; they are not
    # those methods. See DESIGN_NOTES.md section 16.
    snap = _snap({"n": "N", "p": "P", "si": "Si", "ace": "ACE", "pinn": "physics-informed network"})
    m = match_ground_truth_methods(snap, ["physics-informed"])
    assert m["physics-informed"]["node_ids"] == ["pinn"]
    m2 = match_ground_truth_methods(snap, ["MACE-F"])
    assert m2["MACE-F"]["node_ids"] == []


def test_unmatched_term_reports_none():
    snap = _snap({"a": "SchNet"})
    m = match_ground_truth_methods(snap, ["hybrid frameworks"])
    assert m["hybrid frameworks"]["match_type"] == "none"
    assert m["hybrid frameworks"]["node_ids"] == []
