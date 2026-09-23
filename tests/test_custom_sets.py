import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.io import Snapshot  # noqa: E402
from tkh.replies import extract_json  # noqa: E402
from tkh.labeling import (  # noqa: E402
    validate_template, label_set_template, write_labeling_input, write_labeller_request,
    apply_labels_to_hierarchy, LABEL_PROMPT_TEMPLATE, TEMPLATE_FILE)
from tkh.eval.blind import rating_reply_problems, glosses_not_applied  # noqa: E402


# replies from any labeller or rater

def test_reply_parsing_handles_fences_and_preamble():
    assert extract_json('[{"a": 1}]', list) == [{"a": 1}]
    assert extract_json('Here you go:\n```json\n{"I001": 3}\n```\nThanks', dict) == {"I001": 3}
    assert extract_json('Sure. {"G001": "vague"} Hope that helps.', dict) == {"G001": "vague"}
    with pytest.raises(ValueError):
        extract_json('{"I001": 3}', list)  # right JSON, wrong shape
    with pytest.raises(ValueError):
        extract_json("no json here", dict)


# custom labelling templates

def test_template_validation():
    assert validate_template(LABEL_PROMPT_TEMPLATE)
    assert validate_template("Label this {year} cluster:\n{member_lines}\nReply as {{json}}.")
    with pytest.raises(ValueError, match="unknown placeholders"):
        validate_template("{member_lines} {author}")
    with pytest.raises(ValueError, match="member_lines"):
        validate_template("Label the cluster from {year}.")
    with pytest.raises(ValueError, match="double any literal braces"):
        validate_template("{member_lines} and a stray {")


def _world():
    snap = Snapshot(cutoff_year=2026)
    forms = {f"meth_{i:02d}": f"method {i}" for i in range(6)}
    snap.nodes = {n: {"id": n, "type": "method", "surface_form": f} for n, f in forms.items()}
    snap.concept_ids = set(forms)
    h = {"super_nodes": [
        {"id": "L0_S00001", "level": 0, "parent_id": None, "member_ids": sorted(forms), "label": None, "gloss": None},
        {"id": "L1_S00001", "level": 1, "parent_id": "L0_S00001", "member_ids": ["meth_00", "meth_01", "meth_02"],
         "label": None, "gloss": None},
        {"id": "L1_S00002", "level": 1, "parent_id": "L0_S00001", "member_ids": ["meth_03", "meth_04", "meth_05"],
         "label": None, "gloss": None},
    ]}
    return snap, h


def test_custom_template_set_applies_only_with_its_own_template(tmp_path):
    snap, h = _world()
    custom = "Name the theme of these {total_n} items from {year}:\n{member_lines}\n"
    (tmp_path / TEMPLATE_FILE).write_text(custom, encoding="utf-8")
    year_dir = tmp_path / "2026"
    year_dir.mkdir()
    entries = write_labeling_input(h, snap, 2026, year_dir / "labeling_input.json", template=custom)
    assert all(e["prompt"].startswith("Name the theme of these") for e in entries)
    for e in entries:
        e["label"], e["gloss"] = "x", "y"
    (year_dir / "labeling_output.json").write_text(json.dumps(entries), encoding="utf-8")

    template = label_set_template(tmp_path)
    assert template == custom
    _, n, stale = apply_labels_to_hierarchy(_world()[1], year_dir / "labeling_output.json",
                                            snap=snap, year=2026, template=template)
    assert n == 3 and stale == []
    # the same labels checked against the default template look stale
    _, n, stale = apply_labels_to_hierarchy(_world()[1], year_dir / "labeling_output.json",
                                            snap=snap, year=2026)
    assert n == 0 and len(stale) == 3


def test_shipped_sets_without_a_template_file_use_the_default(tmp_path):
    assert label_set_template(tmp_path) == LABEL_PROMPT_TEMPLATE


def test_labeller_request_has_every_item_and_the_reply_format(tmp_path):
    snap, h = _world()
    entries = write_labeling_input(h, snap, 2026, tmp_path / "in.json")
    write_labeller_request(entries, 2026, tmp_path / "req.md")
    text = (tmp_path / "req.md").read_text(encoding="utf-8")
    assert all(f"=== item {e['super_node_id']} ===" in text for e in entries)
    assert '"super_node_id"' in text and "3 clusters" in text


# rating replies and label consistency

def test_rating_reply_validation():
    intr = [{"item_id": "I001", "options": list("abcdef")}, {"item_id": "I002", "options": list("abcdef")}]
    assert rating_reply_problems(intr, {"I001": 0, "I002": 5}, "intruder") == []
    probs = rating_reply_problems(intr, {"I001": 6, "I003": 1}, "intruder")
    assert any("unanswered" in p for p in probs) and any("don't exist" in p for p in probs)
    assert any("invalid" in p for p in probs)
    assert rating_reply_problems(intr, {"I001": True, "I002": 1}, "intruder")  # a bool isn't an index
    gloss = [{"item_id": "G001", "gloss": "g"}]
    assert rating_reply_problems(gloss, {"G001": "vague"}, "gloss") == []
    assert rating_reply_problems(gloss, {"G001": "great"}, "gloss")


def test_gloss_ratings_must_match_the_applied_labels():
    items = [{"item_id": "G001", "gloss": "Alpha methods."}, {"item_id": "G002", "gloss": "Something else."}]
    assert glosses_not_applied(items, ["Alpha methods.", "Beta datasets."]) == ["G002"]
    assert glosses_not_applied(items[:1], ["Alpha methods."]) == []
