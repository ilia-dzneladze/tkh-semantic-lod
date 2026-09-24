"""T6 intrinsic metrics: coherence against a random-labels null, stability
(perturbation and cross-snapshot) and label faithfulness (NLI over-claim
rate). Starts a fresh outputs/metrics.json; the later steps add their own
sections, starting with t6_extrinsic.py.

Usage: t6_evaluate.py [section ...]. With section names it reruns only
those and keeps the rest of the existing metrics.json.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, load_hierarchy, SNAPSHOT_CUTOFFS, DATA_PATH, OUTPUTS  # noqa: E402
from tkh.pipeline import LEVEL_TARGETS, ALPHA  # noqa: E402
from tkh.embeddings import encode_semantic  # noqa: E402
from tkh.eval.coherence import fit_tfidf, coherence_vs_null  # noqa: E402
from tkh.eval.stability import perturbation_stability, cross_snapshot_stability  # noqa: E402
from tkh.eval.faithfulness import (  # noqa: E402
    check_hierarchy_faithfulness, check_provenance_faithfulness, load_article_titles)
from tkh.labeling import unlabelled_super_nodes  # noqa: E402


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def run_coherence(snapshots, hierarchies):
    out = {}
    for year, snap in snapshots.items():
        X, ids, id_to_row = fit_tfidf(snap)
        out[year] = {
            level: coherence_vs_null(hierarchies[year], level, X, id_to_row, n_trials=30, seed=0)
            for level in range(len(LEVEL_TARGETS))
        }
        log(f"coherence {year} done")
    return out


def run_stability(snapshots, hierarchies):
    cross = cross_snapshot_stability(hierarchies, len(LEVEL_TARGETS))
    log("cross-snapshot stability done")

    year = max(snapshots)  # perturbation run on the largest/latest snapshot
    snap = snapshots[year]
    ids = sorted(snap.concept_ids)
    texts = [snap.nodes[nid]["surface_form"] or "" for nid in ids]
    vecs = encode_semantic(texts, show_progress_bar=True)
    cache = dict(zip(ids, vecs))
    log(f"perturbation embeddings ({year}) done")

    pert = perturbation_stability(snap, cache, LEVEL_TARGETS, alpha=ALPHA,
                                   original_hierarchy=hierarchies[year],
                                   n_seeds=5, remove_frac=0.10)
    log("perturbation stability done")
    return {"cross_snapshot": cross, "perturbation": {"year": year, **pert}}


def run_faithfulness(snapshots, hierarchies):
    out = {}
    titles = load_article_titles(ROOT / "data" / "collection10_articles.csv")
    for year, snap in snapshots.items():
        out[year] = check_hierarchy_faithfulness(hierarchies[year], snap, levels=(0, 1), seed=0)
        # same glosses against their members' source-paper titles, which
        # neither labeller nor clustering saw: DESIGN_NOTES.md section 22
        out[year]["provenance"] = check_provenance_faithfulness(
            hierarchies[year], snap, titles, levels=(0, 1), seed=0)
        r = out[year]
        if not r["n_checked"]:
            log(f"faithfulness {year}: nothing checkable ({r['n_labelled']} labelled)")
            continue
        log(f"faithfulness {year} done: checked {r['n_checked']}/{r['n_labelled']} "
            f"(contradiction real={r['real_contradiction_rate']:.3f} control={r['control_contradiction_rate']:.3f}; "
            f"not-entailed real={r['real_not_entailed_rate']:.3f} control={r['control_not_entailed_rate']:.3f})")
    return out


SECTIONS = ("coherence", "stability", "faithfulness")


def main():
    global T0
    T0 = time.time()
    only = sys.argv[1:] or list(SECTIONS)
    unknown = [s for s in only if s not in SECTIONS]
    if unknown:
        sys.exit(f"unknown section(s) {unknown}, expected any of {SECTIONS}")

    data = load_tkh(DATA_PATH)
    snapshots = {y: build_snapshot(data, y) for y in SNAPSHOT_CUTOFFS}
    hierarchies = {y: load_hierarchy(y) for y in SNAPSHOT_CUTOFFS}
    log("loaded snapshots and hierarchies")

    if "faithfulness" in only:
        missing = {y: len(unlabelled_super_nodes(h)) for y, h in hierarchies.items()}
        if any(missing.values()):
            sys.exit(f"hierarchy.json has unlabelled level-0/1 super-nodes {missing}. "
                     f"run_pipeline.py writes labels as null; run scripts/pipeline/t5_apply_labels.py first.")

    metrics_path = OUTPUTS / "metrics.json"
    metrics = {}
    if sys.argv[1:] and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "coherence" in only:
        metrics["coherence"] = run_coherence(snapshots, hierarchies)
    if "stability" in only:
        metrics["stability"] = run_stability(snapshots, hierarchies)
    if "faithfulness" in only:
        metrics["faithfulness"] = run_faithfulness(snapshots, hierarchies)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
