"""Is change between snapshots localised to where the corpus changed (P5)?
Correlates each cluster's churn with its exposure to new hyperedges.
Decision rule written before running: DESIGN_NOTES.md section 15. Reads
the shipped hierarchy.json files, writes outputs/localisation.json and
copies the summary into outputs/metrics.json under "localisation".
Changes nothing in the shipped pipeline.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_snapshot, DATA_PATH  # noqa: E402
from tkh.pipeline import embed_concepts, KNN_K  # noqa: E402
from tkh.embeddings import semantic_knn_graph  # noqa: E402

YEARS = [2020, 2022, 2024, 2026]
MIN_SIZE = 3
N_BOOT = 2000


def load_clusters(year):
    h = json.loads((ROOT / "outputs" / "snapshots" / str(year) / "hierarchy.json").read_text(encoding="utf-8"))
    by_level = defaultdict(dict)
    for sn in h["super_nodes"]:
        by_level[sn["level"]][sn["id"]] = set(sn["member_ids"])
    return by_level


def semantic_exposure_of(snap_prev, snap_curr, cache):
    """Per old node: share of its k-NN neighbours at t+1 that are new nodes.
    Exploratory, added after the pre-registered test failed."""
    ids, emb = embed_concepts(snap_curr, cache)
    A = semantic_knn_graph(emb, k=KNN_K).tocsr()
    is_new = np.array([nid not in snap_prev.concept_ids for nid in ids])
    out = {}
    for i, nid in enumerate(ids):
        nbrs = A.indices[A.indptr[i]:A.indptr[i + 1]]
        if not is_new[i] and len(nbrs):
            out[nid] = float(is_new[nbrs].mean())
    return out


def cluster_rows(prev, curr, snap_prev, snap_curr, sem_exposure):
    """One row per prev cluster: churn on shared nodes, exposure to new edges at t+1."""
    common = set().union(*prev.values()) & set().union(*curr.values())
    node_to_curr = {n: cid for cid, mem in curr.items() for n in mem}
    curr_common_size = {cid: len(mem & common) for cid, mem in curr.items()}
    old_edges = {e["id"] for e in snap_prev.hyperedges}
    edges_of = defaultdict(set)
    for e in snap_curr.hyperedges:
        for m in e["members"]:
            edges_of[m].add(e["id"])
    rows = []
    for cid, mem in prev.items():
        if len(mem) < MIN_SIZE:
            continue
        c = mem & common
        overlap = Counter(node_to_curr[n] for n in c)
        best = max((k / (len(c) + curr_common_size[d] - k) for d, k in overlap.items()), default=0.0)
        touching = set().union(*(edges_of[n] for n in mem))
        exposure = len(touching - old_edges) / len(touching) if touching else 0.0
        # sorted: summing a set of strings in hash order made the mean differ
        # in the last bit between runs (DESIGN_NOTES.md section 23)
        sem = [sem_exposure[n] for n in sorted(mem) if n in sem_exposure]
        rows.append({"cluster": cid, "size": len(mem), "churn": 1.0 - best, "exposure": exposure,
                     "semantic_exposure": float(np.mean(sem)) if sem else 0.0})
    return rows


def spearman_and_partial(x, y, z):
    rho = stats.spearmanr(x, y).statistic
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    partial = float(np.corrcoef(res(rx), res(ry))[0, 1])
    return float(rho), partial


def summarise(rows, rng, key="exposure"):
    x = np.array([r[key] for r in rows])
    y = np.array([r["churn"] for r in rows])
    z = np.log(np.array([r["size"] for r in rows]))
    rho, partial = spearman_and_partial(x, y, z)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(rows), len(rows))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        boots.append(spearman_and_partial(x[i], y[i], z[i]))
    boots = np.array(boots)
    q_lo, q_hi = np.quantile(x, [0.25, 0.75])
    return {
        "n_clusters": len(rows),
        "spearman": rho, "spearman_ci95": np.quantile(boots[:, 0], [0.025, 0.975]).tolist(),
        "partial_spearman_given_size": partial,
        "partial_ci95": np.quantile(boots[:, 1], [0.025, 0.975]).tolist(),
        "mean_churn_low_exposure_quartile": float(y[x <= q_lo].mean()),
        "mean_churn_high_exposure_quartile": float(y[x >= q_hi].mean()),
        "mean_churn": float(y.mean()), "mean_exposure": float(x.mean()),
    }


def main():
    data = load_tkh(DATA_PATH)
    snaps = {y: build_snapshot(data, y) for y in YEARS}
    clusters = {y: load_clusters(y) for y in YEARS}
    rng = np.random.default_rng(0)
    cache = {}
    rows_by_level = defaultdict(list)
    for t, t1 in zip(YEARS, YEARS[1:]):
        sem = semantic_exposure_of(snaps[t], snaps[t1], cache)
        for level in sorted(clusters[t]):
            for r in cluster_rows(clusters[t][level], clusters[t1][level], snaps[t], snaps[t1], sem):
                rows_by_level[level].append({**r, "from_year": t, "to_year": t1})
    summary = {}
    for level, rows in sorted(rows_by_level.items()):
        s = summarise(rows, rng)
        s["localised"] = s["spearman_ci95"][0] > 0 and s["partial_ci95"][0] > 0
        low = [r["churn"] for r in rows if r["exposure"] < 0.02]
        s["n_clusters_under_2pct_new_edges"] = len(low)
        s["mean_churn_under_2pct_new_edges"] = float(np.mean(low)) if low else None
        s["exploratory_semantic_exposure"] = summarise(rows, rng, key="semantic_exposure")
        summary[level] = s
        e = s["exploratory_semantic_exposure"]
        print(f"   semantic (exploratory): rho={e['spearman']:+.2f} CI{np.round(e['spearman_ci95'], 2).tolist()}"
              f"  partial={e['partial_spearman_given_size']:+.2f} CI{np.round(e['partial_ci95'], 2).tolist()}"
              f"  mean sem exposure {e['mean_exposure']:.2f}")
        print(f"L{level}: n={s['n_clusters']}  rho={s['spearman']:+.2f} CI{np.round(s['spearman_ci95'], 2).tolist()}"
              f"  partial={s['partial_spearman_given_size']:+.2f} CI{np.round(s['partial_ci95'], 2).tolist()}"
              f"  churn low/high exposure {s['mean_churn_low_exposure_quartile']:.2f}/"
              f"{s['mean_churn_high_exposure_quartile']:.2f}  localised={s['localised']}")
    out = {"years": YEARS, "min_size": MIN_SIZE, "n_boot": N_BOOT, "summary": summary,
           "rows": {lvl: rows for lvl, rows in rows_by_level.items()}}
    (ROOT / "outputs" / "localisation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    metrics_path = ROOT / "outputs" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["localisation"] = {k: v for k, v in out.items() if k != "rows"}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
