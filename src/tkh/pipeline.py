"""The whole method: cluster each snapshot into levels (T2), track the
levels across snapshots (T3), collapse the hyperedges at each level (T4),
and assemble hierarchy.json.
"""

import numpy as np
import scipy.sparse as sp

from .hypergraph import build_structural_affinity
from .embeddings import encode_semantic, semantic_knn_graph
from .cluster import combine_affinities, sparse_upgma, cut_to_k_clusters
from .collapse import build_coarse_hyperedges, coarse_structural_affinity
from .temporal import clusters_from_labels, track_across_snapshots

LEVEL_TARGETS = [12, 50, 200]  # level 0 (coarsest) .. level 2 (finest super-node level)
ALPHA = 0.3  # see DESIGN_NOTES.md sections 7 and 17
KNN_K = 15
COARSENING = "dendrogram"  # or "multilevel"; see DESIGN_NOTES.md section 9


def coarsen_one_level(snap, ids, emb, fine_labels, k_target, alpha=ALPHA, size_normalize=False):
    """Cluster the super-nodes of one level into at most k_target groups,
    using the T4-collapsed hypergraph for structure and member-centroid
    embeddings for semantics. Returns (node-level labels, n_forced_merges).
    Laminar by construction: each fine super-node moves as one unit.

    size_normalize: divide structural weight between S and T by |S||T| and
    weight the linkage by member counts (average linkage over nodes).
    See DESIGN_NOTES.md section 15."""
    m = int(fine_labels.max()) + 1
    super_ids = [f"__super_{i}" for i in range(m)]
    mapping = {nid: super_ids[lab] for nid, lab in zip(ids, fine_labels)}
    coarse_edges, _, _ = build_coarse_hyperedges(snap.hyperedges, mapping, snap)
    A_struct = coarse_structural_affinity(coarse_edges, super_ids)
    counts = np.bincount(fine_labels, minlength=m).astype(float)
    if size_normalize:
        inv = sp.diags(1.0 / counts)
        A_struct = (inv @ A_struct @ inv).tocsr()

    centroids = np.zeros((m, emb.shape[1]))
    np.add.at(centroids, fine_labels, emb)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    A_sem = semantic_knn_graph(centroids, k=min(KNN_K, m - 1))

    Z, forced = sparse_upgma(combine_affinities(A_struct, A_sem, alpha=alpha), m,
                             sizes=counts if size_normalize else None)
    super_labels = cut_to_k_clusters(Z, m, k_target)
    return super_labels[fine_labels], int(forced.sum())


def build_levels(snap, ids, emb, alpha=ALPHA, level_targets=LEVEL_TARGETS,
                 coarsening=COARSENING, A_sem=None):
    """Labels for every level, finest last. The finest level is always a
    cut of the node-level tree. With coarsening="dendrogram" the coarser
    levels are cuts of the same tree; with "multilevel" each one clusters
    the super-nodes of the level below (coarsen_one_level). ids must be
    sorted(snap.concept_ids), row-aligned with emb."""
    A_struct, struct_ids = build_structural_affinity(snap)
    assert struct_ids == list(ids), "ids must be sorted(snap.concept_ids)"
    if A_sem is None:
        A_sem = semantic_knn_graph(emb, k=KNN_K)
    n = len(ids)
    Z, forced = sparse_upgma(combine_affinities(A_struct, A_sem, alpha=alpha), n)

    finest = len(level_targets) - 1
    labels_by_level = {finest: cut_to_k_clusters(Z, n, level_targets[finest])}
    forced_by_level = {finest: int(forced.sum())}
    for level_idx in range(finest - 1, -1, -1):
        if coarsening == "dendrogram":
            labels_by_level[level_idx] = cut_to_k_clusters(Z, n, level_targets[level_idx])
            forced_by_level[level_idx] = int(forced.sum())
        elif coarsening in ("multilevel", "multilevel_sizenorm"):
            labels_by_level[level_idx], forced_by_level[level_idx] = coarsen_one_level(
                snap, ids, emb, labels_by_level[level_idx + 1], level_targets[level_idx], alpha,
                size_normalize=(coarsening == "multilevel_sizenorm"))
        else:
            raise ValueError(f"unknown coarsening {coarsening!r}")
    return {"labels_by_level": dict(sorted(labels_by_level.items())),
            "forced_by_level": dict(sorted(forced_by_level.items()))}


def embed_concepts(snap, embedding_cache):
    """Sorted concept ids and their embedding matrix, filling the cache."""
    ids = sorted(snap.concept_ids)
    missing = [nid for nid in ids if nid not in embedding_cache]
    if missing:
        texts = [snap.nodes[nid]["surface_form"] or "" for nid in missing]
        for nid, v in zip(missing, encode_semantic(texts)):
            embedding_cache[nid] = v
    return ids, np.stack([embedding_cache[nid] for nid in ids])


def run_single_snapshot(snap, embedding_cache, alpha=ALPHA, level_targets=LEVEL_TARGETS,
                        coarsening=COARSENING):
    """{"ids", "labels_by_level", "forced_by_level"} for one snapshot."""
    ids, emb = embed_concepts(snap, embedding_cache)
    return {"ids": ids, **build_levels(snap, ids, emb, alpha=alpha, level_targets=level_targets,
                                       coarsening=coarsening)}


def _laminar_parents(labels_by_level):
    """{level: {label: parent label one level up}} for levels > 0. Raises
    if a cluster's members disagree on their parent, which can't happen
    for cuts of one tree but would catch a bug."""
    parents = {}
    for level_idx in range(1, len(labels_by_level)):
        mapping = {}
        for fine, coarse in zip(labels_by_level[level_idx], labels_by_level[level_idx - 1]):
            if mapping.setdefault(int(fine), int(coarse)) != int(coarse):
                raise AssertionError(f"laminarity violated at level {level_idx}: "
                                     f"label {int(fine)} has two parents")
        parents[level_idx] = mapping
    return parents


def run_all_snapshots(snapshots_by_year, alpha=ALPHA, level_targets=LEVEL_TARGETS,
                      coarsening=COARSENING, embedding_cache=None):
    """Cluster every snapshot and track each level across them. Returns
    what build_hierarchy_json and temporal_events.json need."""
    embedding_cache = {} if embedding_cache is None else embedding_cache
    years = sorted(snapshots_by_year)
    per_year = {}
    for year in years:
        per_year[year] = run_single_snapshot(snapshots_by_year[year], embedding_cache, alpha=alpha,
                                             level_targets=level_targets, coarsening=coarsening)
        per_year[year]["parents"] = _laminar_parents(per_year[year]["labels_by_level"])

    persistent_by_level_year, events_by_level = {}, {}
    for level_idx in range(len(level_targets)):
        labels_by_year = {y: (per_year[y]["labels_by_level"][level_idx], per_year[y]["ids"]) for y in years}
        persistent_by_level_year[level_idx], events_by_level[level_idx] = track_across_snapshots(
            labels_by_year, prefix=f"L{level_idx}_")

    return {
        "years": years, "per_year": per_year, "level_targets": level_targets,
        "persistent_by_level_year": persistent_by_level_year,
        "events_by_level": events_by_level,
    }


def build_hierarchy_json(result, year, snap):
    """The hierarchy.json for one snapshot: every super-node (persistent id,
    level, parent id, member ids, empty label and gloss for T5) and each
    level's T4-collapsed hyperedges. member_ids are concept node ids at
    every level, not child super-node ids: DESIGN_NOTES.md section 11."""
    per_year = result["per_year"][year]
    ids = per_year["ids"]
    super_nodes, node_to_persistent = [], {}
    for level_idx in range(len(result["level_targets"])):
        persistent = result["persistent_by_level_year"][level_idx][year]  # local label -> persistent id
        clusters = clusters_from_labels(per_year["labels_by_level"][level_idx], ids)
        node_to_persistent[level_idx] = {nid: persistent[lab] for lab, members in clusters.items()
                                         for nid in members}
        for lab, members in clusters.items():
            parent_id = (None if level_idx == 0 else
                         result["persistent_by_level_year"][level_idx - 1][year][per_year["parents"][level_idx][lab]])
            super_nodes.append({
                "id": persistent[lab], "level": level_idx, "parent_id": parent_id,
                "member_ids": sorted(members), "member_count": len(members),
                "label": None, "gloss": None,
            })

    # T4 applied to the original hyperedges at each level, not re-collapsed
    # from the level below
    edges_by_level, internal_by_level = {}, {}
    for level_idx, mapping in node_to_persistent.items():
        coarse_edges, internal_stats, summary = build_coarse_hyperedges(snap.hyperedges, mapping, snap)
        edges_by_level[level_idx] = {"edges": coarse_edges, "summary": summary}
        internal_by_level[level_idx] = internal_stats

    return {
        "year": year,
        "super_nodes": super_nodes,
        "coarse_hyperedges_by_level": edges_by_level,
        "internal_stats_by_level": internal_by_level,
    }
