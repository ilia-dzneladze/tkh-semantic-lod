"""Blind evaluations that use neither the clustering's embedding nor the
labeller's input as the ruler. Decision rules: DESIGN_NOTES.md section 15;
design: sections 21 and 22.

Usage:
  blind_eval.py make [OUT_DIR]   build the intruder and gloss-rating
      packets. Writes, per task, <task>_items.json (what the rater sees,
      no ids that reveal an item's kind), <task>_key.json (the answers,
      never shown to a rater) and <task>_rater_prompt.md (the exact text a
      rater is given). The NLI judge is run on the gloss items here, on
      exactly the members the rater sees, and its label goes in the key.
      OUT_DIR defaults to outputs/blind_eval. Keep the keys away from the
      rater while rating.
  blind_eval.py score            read outputs/blind_eval/<task>_ratings.json
      and write the results into metrics.json under "blind_eval".
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, SNAPSHOT_CUTOFFS  # noqa: E402
from tkh.eval.blind import (  # noqa: E402
    make_intruder_items, score_intruder, make_gloss_items, score_gloss_ratings)

DATA_PATH = ROOT / "data" / "tkh_collection10.json"
BLIND_DIR = ROOT / "outputs" / "blind_eval"
INTRUDER_YEAR = 2026
INTRUDER_PER_LEVEL = {0: 12, 1: 24, 2: 24}
INTRUDER_NULL = 20
GLOSS_REAL, GLOSS_CONTROL = 12, 6  # per snapshot
SEED = 0

INTRUDER_INSTRUCTIONS = """\
You are the rater in a blind evaluation. Each item below lists six terms,
numbered 0 to 5, taken from a knowledge graph built from machine-learning
and materials-science research papers. In an item, five of the terms may
belong together and one was taken from somewhere else. For each item, pick
the number of the ONE term that least belongs with the other five. If
nothing stands out, still give your best guess: every item needs exactly
one answer. Judge by meaning only.

Do not use any tools, files or searches. Everything you need is in this
message. Reply with ONLY a JSON object mapping each item id to the number
you picked, for example {"I001": 3, "I002": 0}, covering all %d items.
"""

GLOSS_INSTRUCTIONS = """\
You are the rater in a blind evaluation. Each item below has a one-sentence
gloss, written to describe a group of terms, and a list of terms. The list
is a sample of one group of terms from a knowledge graph built from
machine-learning and materials-science research papers. It may or may not
be the group the gloss was written for. Rate whether the listed terms
support the gloss:

  accurate: the listed terms clearly support what the gloss says, and it
            fits most of them.
  vague:    nothing in the gloss is contradicted, but it is too general to
            tell this list apart from many others, or it fits only a small
            part of the list.
  wrong:    the gloss says something the listed terms contradict, or it
            does not fit most of them.

Judge only by what the listed terms support, not by what you know about
the field. Do not use any tools, files or searches. Everything you need is
in this message. Reply with ONLY a JSON object mapping each item id to one
of "accurate", "vague" or "wrong", for example {"G001": "vague"}, covering
all %d items.
"""


def _write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def make(out_dir):
    from tkh.eval.faithfulness import nli_labels

    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_tkh(DATA_PATH)
    snaps = {y: build_snapshot(data, y) for y in SNAPSHOT_CUTOFFS}
    hier = {y: json.loads((ROOT / "outputs" / "snapshots" / str(y) / "hierarchy.json").read_text(encoding="utf-8"))
            for y in SNAPSHOT_CUTOFFS}

    rng = np.random.default_rng(SEED)
    items, key = make_intruder_items(hier[INTRUDER_YEAR], snaps[INTRUDER_YEAR],
                                     INTRUDER_PER_LEVEL, INTRUDER_NULL, rng)
    lines = [INTRUDER_INSTRUCTIONS % len(items)]
    for it in items:
        lines.append(it["item_id"])
        lines += [f"  {i}. {opt}" for i, opt in enumerate(it["options"])]
    _write(out_dir / "intruder_items.json", items)
    _write(out_dir / "intruder_key.json", key)
    (out_dir / "intruder_rater_prompt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    kinds = [k["kind"] for k in key.values()]
    print(f"intruder: {kinds.count('real')} real + {kinds.count('null')} null items "
          f"(asked for {sum(INTRUDER_PER_LEVEL.values())} + {INTRUDER_NULL})")

    rng = np.random.default_rng(SEED + 1)
    items, key = make_gloss_items(hier, snaps, GLOSS_REAL, GLOSS_CONTROL, rng)
    labels = nli_labels([[key[it["item_id"]]["premise"], it["gloss"]] for it in items])
    for it, label in zip(items, labels):
        key[it["item_id"]]["nli_label"] = label
    lines = [GLOSS_INSTRUCTIONS % len(items)]
    for it in items:
        lines.append(f"{it['item_id']}\n  gloss: {it['gloss']}\n  terms:")
        lines += [f"    - {m}" for m in it["members"]]
    _write(out_dir / "gloss_items.json", items)
    _write(out_dir / "gloss_key.json", key)
    (out_dir / "gloss_rater_prompt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    kinds = [k["kind"] for k in key.values()]
    print(f"gloss: {kinds.count('real')} real + {kinds.count('control')} control items")
    print(f"wrote packets to {out_dir}")


def _ratings(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("ratings", obj), {k: v for k, v in obj.items() if k != "ratings"}


def score():
    out = {}
    intr_key = json.loads((BLIND_DIR / "intruder_key.json").read_text(encoding="utf-8"))
    intr, intr_meta = _ratings(BLIND_DIR / "intruder_ratings.json")
    out["intruder"] = {**score_intruder(intr_key, intr), "rater": intr_meta}

    gloss_key = json.loads((BLIND_DIR / "gloss_key.json").read_text(encoding="utf-8"))
    gl, gl_meta = _ratings(BLIND_DIR / "gloss_ratings.json")
    out["gloss"] = {**score_gloss_ratings(gloss_key, gl, np.random.default_rng(SEED)), "rater": gl_meta}

    metrics_path = ROOT / "outputs" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["blind_eval"] = out
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    i = out["intruder"]
    print(f"intruder real {i['real']['k']}/{i['real']['n']} CI {np.round(i['real']['ci95'], 3).tolist()}, "
          f"null {i['null']['k']}/{i['null']['n']} CI {np.round(i['null']['ci95'], 3).tolist()}, chance {i['chance']:.3f}")
    for lvl, r in i["real_by_level"].items():
        print(f"  level {lvl}: {r['k']}/{r['n']} CI {np.round(r['ci95'], 3).tolist()} p={r['p_vs_chance']:.2g}")
    g = out["gloss"]
    for kind in ("real", "control"):
        print(f"gloss {kind}: " + ", ".join(f"{k} {v['k']}/{v['n']}" for k, v in g[kind].items()))
    print(f"kappa wrong~contradiction {g['agreement_wrong_vs_contradiction']['kappa']:.2f} "
          f"CI {np.round(g['agreement_wrong_vs_contradiction']['ci95'], 2).tolist()}; "
          f"3-way {g['agreement_3way']['kappa']:.2f}; real NLI-not-entailed rated accurate "
          f"{g['real_nli_not_entailed_rated_accurate']['k']}/{g['real_nli_not_entailed_rated_accurate']['n']}")
    print(f"wrote {metrics_path} [blind_eval]")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "make":
        make(Path(sys.argv[2]) if len(sys.argv) == 3 else BLIND_DIR)
    elif len(sys.argv) == 2 and sys.argv[1] == "score":
        score()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
