"""Build release/: what to publish so anyone can check their results
against these ones. Nothing here is a trained model: the method uses two
frozen, pinned models, and these are their outputs plus the checksums of
every result.

Usage: export_release.py RECORD_DIR [RELEASE_DIR]

RECORD_DIR comes from `reproduce_all.py --record RECORD_DIR` on a clean
tree, or is an existing release/model_outputs (to re-export after a
commit, so environment.json names the right code). Everything below is read back through replay, so each vector is bit
for bit the one the pipeline used. See DESIGN_NOTES.md section 23.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.model_outputs import ENV_VAR  # noqa: E402

DATA_PATH = ROOT / "data" / "tkh_collection10.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _partition(labels, ids):
    groups = {}
    for nid, lab in zip(ids, labels):
        groups.setdefault(int(lab), set()).add(nid)
    return {frozenset(g) for g in groups.values()}


def _versions():
    import importlib.metadata as md
    out = {}
    for pkg in ("torch", "numpy", "scipy", "scikit-learn", "sentence-transformers", "transformers"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    record = Path(sys.argv[1]).resolve()
    release = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else ROOT / "release"
    conflicts = record / "conflicts.log"
    if conflicts.exists() and conflicts.read_text(encoding="utf-8").strip():
        sys.exit(f"{conflicts} is not empty: a repeated model call gave different outputs, so the "
                 f"recording isn't a single consistent run. Record again.")
    os.environ[ENV_VAR] = f"replay:{record}"

    from tkh.io import load_tkh, build_all_snapshots
    from tkh.pipeline import embed_concepts, ALPHA, KNN_K, LEVEL_TARGETS
    from tkh.hypergraph import build_structural_affinity
    from tkh.embeddings import encode_semantic, semantic_knn_graph, _MODEL_NAME, _MODEL_REVISION
    from tkh.cluster import combine_affinities, sparse_upgma, cut_to_k_clusters
    from tkh.eval.extrinsic import METHOD_LIKE_TYPES, build_retrieval_texts
    from tkh.eval import faithfulness

    release.mkdir(parents=True, exist_ok=True)
    mo = release / "model_outputs"
    if mo.resolve() != record:  # re-exporting from release/model_outputs itself keeps it in place
        if mo.exists():
            shutil.rmtree(mo)
        shutil.copytree(record, mo, ignore=shutil.ignore_patterns("conflicts.log"))
    n_embed = len(list(mo.glob("embed_*.npy")))
    n_nli = len(list(mo.glob("nli_*.npy")))
    print(f"model_outputs: {n_embed} embedding calls, {n_nli} NLI calls")

    # node embeddings by id: the same calls run_pipeline makes, in the same order
    snaps = build_all_snapshots(load_tkh(DATA_PATH))
    cache = {}
    for year in sorted(snaps):
        embed_concepts(snaps[year], cache)
    ids = sorted(cache)
    np.savez(release / "node_embeddings.npz", node_ids=np.array(ids),
             embeddings=np.stack([cache[n] for n in ids]))
    print(f"node_embeddings: {len(ids)} concept nodes x {cache[ids[0]].shape[0]}")

    # retrieval embeddings: the extrinsic eval's enriched texts, 2026
    s26 = snaps[2026]
    method_ids = sorted(n for n in s26.concept_ids if s26.nodes[n]["type"] in METHOD_LIKE_TYPES)
    texts, _ = build_retrieval_texts(s26, method_ids, thin_threshold=20, max_context_terms=10)
    np.savez(release / "retrieval_embeddings_2026.npz", node_ids=np.array(method_ids),
             embeddings=encode_semantic(texts, batch_size=32, max_seq_length=64))
    print(f"retrieval_embeddings_2026: {len(method_ids)} method-like nodes")

    # merge trees, checked against the shipped hierarchies before writing
    for year in sorted(snaps):
        snap = snaps[year]
        ids_y = sorted(snap.concept_ids)
        emb = np.stack([cache[n] for n in ids_y])
        A_struct, _, _ = build_structural_affinity(snap, weighted=True)
        Z, forced = sparse_upgma(combine_affinities(A_struct, semantic_knn_graph(emb, k=KNN_K), alpha=ALPHA),
                                 len(ids_y))
        h = json.loads((ROOT / "outputs" / "snapshots" / str(year) / "hierarchy.json").read_text(encoding="utf-8"))
        for level, k in enumerate(LEVEL_TARGETS):
            shipped = {frozenset(sn["member_ids"]) for sn in h["super_nodes"] if sn["level"] == level}
            if _partition(cut_to_k_clusters(Z, len(ids_y), k), ids_y) != shipped:
                sys.exit(f"linkage for {year} doesn't cut into the shipped level-{level} partition; "
                         f"the outputs and the recording come from different runs")
        np.savez(release / f"linkage_{year}.npz", node_ids=np.array(ids_y), Z=Z, forced=forced)
    print("linkage: 4 merge trees, each cuts into the shipped hierarchy exactly")

    env = {
        "exported_on": date.today().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": _versions(),
        "models": {"embedding": f"{_MODEL_NAME}@{_MODEL_REVISION}",
                   "nli": f"{faithfulness._MODEL_NAME}@{faithfulness._MODEL_REVISION}"},
    }
    (release / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    card = (ROOT / "scripts" / "release_card.md").read_text(encoding="utf-8")
    (release / "README.md").write_text(card.format(
        n_embed=n_embed, n_nli=n_nli, n_nodes=len(ids), n_method=len(method_ids),
        platform=env["platform"], python=env["python"], torch=env["packages"]["torch"],
        embedding_model=env["models"]["embedding"], nli_model=env["models"]["nli"],
        git_commit=env["git_commit"] + (" plus uncommitted changes" if env["git_dirty"] else "")),
        encoding="utf-8")

    # figures are left out: matplotlib's PNG bytes aren't stable across versions
    listed = [(p, p.relative_to(ROOT)) for p in sorted((ROOT / "outputs").rglob("*"))
              if p.is_file() and p.suffix != ".png"]
    listed += [(p, Path("release") / p.relative_to(release)) for p in sorted(release.rglob("*"))
               if p.is_file() and p.name != "SHA256SUMS.txt"]
    lines = [f"{sha256(p)}  {rel.as_posix()}" for p, rel in listed]
    (release / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    size_mb = sum(p.stat().st_size for p in release.rglob("*") if p.is_file()) / 1e6
    print(f"SHA256SUMS.txt: {len(lines)} files; release is {size_mb:.1f} MB at {release}")


if __name__ == "__main__":
    main()
