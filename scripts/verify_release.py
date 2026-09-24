"""Check your results against the published ones.

Usage:
  verify_release.py checksums [SUMS]        compare every local output (and
      any downloaded release file) with SHA256SUMS.txt; SUMS defaults to
      release/SHA256SUMS.txt. Exits non-zero if anything present differs.
  verify_release.py compare-embeddings [RELEASE_DIR]   recompute the node
      embeddings with the real model on this machine, the same way
      run_pipeline.py does, and compare them with release/node_embeddings.npz.

See DESIGN_NOTES.md section 23 and the release README.
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.model_outputs import ENV_VAR  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checksums(sums_path):
    rows = [ln.split(None, 1) for ln in sums_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ok, bad, missing, not_downloaded = [], [], [], []
    for digest, rel in rows:
        path = ROOT / rel
        if not path.exists():
            (not_downloaded if rel.startswith("release/") else missing).append(rel)
        elif sha256(path) == digest:
            ok.append(rel)
        else:
            bad.append(rel)
    outputs_ok = sum(1 for r in ok if r.startswith("outputs/"))
    outputs_total = sum(1 for _, r in rows if r.startswith("outputs/"))
    print(f"outputs: {outputs_ok} of {outputs_total} identical")
    release_rows = [r for _, r in rows if r.startswith("release/")]
    print(f"release files: {sum(1 for r in ok if r.startswith('release/'))} of {len(release_rows)} identical, "
          f"{len(not_downloaded)} not downloaded")
    for rel in bad:
        print(f"  DIFFERENT  {rel}")
    for rel in missing:
        print(f"  MISSING    {rel}")
    if bad or missing:
        sys.exit(1)
    print("everything present matches")


def compare_embeddings(release_dir):
    if os.environ.get(ENV_VAR):
        sys.exit(f"unset {ENV_VAR}: this compares the real model's output with the published one")
    from tkh.io import load_tkh, build_all_snapshots, DATA_PATH
    from tkh.pipeline import embed_concepts, KNN_K
    from tkh.embeddings import semantic_knn_graph

    pub = np.load(release_dir / "node_embeddings.npz")
    pub_ids = [str(x) for x in pub["node_ids"]]
    pub_emb = pub["embeddings"]

    snaps = build_all_snapshots(load_tkh(DATA_PATH))
    cache = {}
    for year in sorted(snaps):
        embed_concepts(snaps[year], cache)
    if sorted(cache) != pub_ids:
        sys.exit("your concept node set differs from the published one; check the data file first")
    mine = np.stack([cache[n] for n in pub_ids])

    diff = np.abs(mine - pub_emb)
    exact = int(np.all(mine == pub_emb, axis=1).sum())
    m64, p64 = mine.astype(np.float64), pub_emb.astype(np.float64)
    cos = np.sum(m64 * p64, axis=1) / (np.linalg.norm(m64, axis=1) * np.linalg.norm(p64, axis=1))
    print(f"node embeddings: {exact} of {len(pub_ids)} bit-identical, max abs difference {diff.max():.3g}, "
          f"min cosine to published {cos.min():.9f}")

    # nearest neighbours are where tiny differences start to change clusters
    idx = {n: i for i, n in enumerate(pub_ids)}
    for year in sorted(snaps):
        rows = [idx[n] for n in sorted(snaps[year].concept_ids)]
        a = semantic_knn_graph(pub_emb[rows], k=KNN_K)
        b = semantic_knn_graph(mine[rows], k=KNN_K)
        same_edges = (a != 0).multiply(b != 0).nnz
        union = ((a != 0) + (b != 0)).nnz
        print(f"  {year}: k-NN graph edges identical {same_edges // 2} of {union // 2}"
              f"{'  (identical, so clustering inputs match)' if same_edges == union else ''}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "checksums" and len(args) <= 2:
        checksums(Path(args[1]) if len(args) == 2 else ROOT / "release" / "SHA256SUMS.txt")
    elif args and args[0] == "compare-embeddings" and len(args) <= 2:
        compare_embeddings(Path(args[1]) if len(args) == 2 else ROOT / "release")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
