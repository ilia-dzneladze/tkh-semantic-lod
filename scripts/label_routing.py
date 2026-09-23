"""Label routing vs centroid routing on the 2026 hierarchy.

Usage: label_routing.py TAG [LABELS_DIR]. Without LABELS_DIR it scores
whatever labels are currently applied to hierarchy.json; with it, the
labels in LABELS_DIR/<year>/labeling_output.json are applied in memory
instead (outputs/labels_v1 for the first label set), leaving the shipped
hierarchy.json untouched. Results are stored under TAG in
outputs/label_routing.json, so runs for different label sets sit side by
side. Decision rule: DESIGN_NOTES.md section 15.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot  # noqa: E402
from tkh.pipeline import embed_concepts  # noqa: E402
from tkh.embeddings import encode_semantic  # noqa: E402
from tkh.labeling import apply_labels_to_hierarchy  # noqa: E402
from tkh.eval.extrinsic import (  # noqa: E402
    routing_pool_recall, load_type_a_questions, centroid_vectors)

YEAR = 2026
BUDGETS = [(2, 2), (3, 3), (4, 4), (6, 6), (8, 8)]
OUT_PATH = ROOT / "outputs" / "label_routing.json"


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: label_routing.py TAG [LABELS_DIR]")
    tag = sys.argv[1]
    snap = build_snapshot(load_tkh(ROOT / "data" / "tkh_collection10.json"), YEAR)
    h = json.loads((ROOT / "outputs" / "snapshots" / str(YEAR) / "hierarchy.json").read_text(encoding="utf-8"))
    if len(sys.argv) == 3:
        labels_path = Path(sys.argv[2]) / str(YEAR) / "labeling_output.json"
        # archived sets were written from an older prompt template, so only
        # the member-count check applies here, not the prompt fingerprint
        h, n_applied, stale = apply_labels_to_hierarchy(h, labels_path)
        if stale:
            sys.exit(f"{len(stale)} archived labels don't match the current clusters: {stale[:5]}")
        print(f"applied {n_applied} labels from {labels_path} (in memory only)")
    questions = load_type_a_questions(ROOT, snap, encode_semantic)

    labelled = [sn for sn in h["super_nodes"] if sn["level"] in (0, 1)]
    assert all(sn.get("gloss") for sn in labelled), "every level-0/1 super-node needs a label"
    lg = encode_semantic([f"{sn['label']}. {sn['gloss']}" for sn in labelled])
    label_vecs = dict(zip([sn["id"] for sn in labelled], lg))

    cache = {}
    embed_concepts(snap, cache)
    cent_vecs = centroid_vectors(h, cache)

    result = {}
    for name, vecs in (("label", label_vecs), ("centroid", cent_vecs)):
        rng = np.random.default_rng(0)
        result[name] = [routing_pool_recall(h, snap, questions, vecs, b0, b1, rng) for b0, b1 in BUDGETS]
        for r in result[name]:
            print(f"{tag} {name:8s} ({r['b0']},{r['b1']}): gt_in_pool={r['gt_in_pool_rate']:.3f} "
                  f"pool={r['mean_pool_fraction']:.3f} lift={r['lift_over_chance']:+.3f} "
                  f"CI{np.round(r['lift_ci95'], 3).tolist()}")

    all_runs = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    all_runs[tag] = {"year": YEAR, "n_questions": len(questions), **result}
    OUT_PATH.write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH} [{tag}]")


if __name__ == "__main__":
    main()
