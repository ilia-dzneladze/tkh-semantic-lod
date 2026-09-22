"""Glue: run T2's single-snapshot method at multiple levels, then hand the
per-level label sequences to T3 (temporal.py) and T4 (collapse.py) to
assemble the final per-snapshot, multi-level, temporally-tracked hierarchy.
"""
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from .hypergraph import build_structural_affinity
from .embeddings import encode_semantic, semantic_knn_graph
from .cluster import combine_affinities, sparse_upgma, cut_to_k_clusters
from .collapse import build_coarse_hyperedges, coarse_structural_affinity
from .temporal import clusters_from_labels, track_across_snapshots

LEVEL_TARGETS = [12, 50, 200]  # level 0 (coarsest) .. level 2 (finest super-node level)
ALPHA = 0.3  # see DESIGN_NOTES.md: structural/semantic combination (alpha ablation)
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
    """Laminar labels for every level. The finest level is always a cut of
    the node-level dendrogram. With coarsening="dendrogram" the coarser
    levels are cuts of that same dendrogram; with "multilevel" each coarser
    level clusters the super-nodes of the level below via
    coarsen_one_level. ids must be sorted(snap.concept_ids), row-aligned
    with emb."""
    A_struct, struct_ids, struct_stats = build_structural_affinity(snap, weighted=True)
    assert struct_ids == list(ids), "ids must match build_structural_affinity's ordering"
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
            "forced_by_level": dict(sorted(forced_by_level.items())),
            "struct_stats": struct_stats, "n_sem_edges": A_sem.nnz}


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
    """Returns dict: ids, labels_by_level {level_idx: np.array},
    forced_by_level, struct_stats, n_sem_edges."""
    ids, emb = embed_concepts(snap, embedding_cache)
    out = build_levels(snap, ids, emb, alpha=alpha, level_targets=level_targets,
                       coarsening=coarsening)
    return {"ids": ids, **out}


def _laminar_parents(labels_by_level, ids):
    """For each level > 0, map each local label to its coarser (level-1)
    local label, verifying every member of a fine cluster agrees on the
    same coarse parent (laminarity sanity check -- guaranteed by
    construction since all cuts come from ONE dendrogram, but we assert it
    to catch bugs)."""
    parents = {}  # level_idx -> {fine_local_label: coarse_local_label}
    n_levels = len(labels_by_level)
    for level_idx in range(1, n_levels):
        coarse_labels = labels_by_level[level_idx - 1]
        fine_labels = labels_by_level[level_idx]
        mapping = {}
        for fine_lab, coarse_lab in zip(fine_labels, coarse_labels):
            fine_lab, coarse_lab = int(fine_lab), int(coarse_lab)
            if fine_lab in mapping:
                assert mapping[fine_lab] == coarse_lab, (
                    f"laminarity violated at level {level_idx}: "
                    f"local label {fine_lab} maps to two different parents")
            else:
                mapping[fine_lab] = coarse_lab
        parents[level_idx] = mapping
    return parents


def run_all_snapshots(snapshots_by_year, alpha=ALPHA, level_targets=LEVEL_TARGETS,
                      coarsening=COARSENING, embedding_cache=None):
    """snapshots_by_year: {year: Snapshot}. Returns everything needed to
    write hierarchy.json per year + one temporal_events.json."""
    embedding_cache = {} if embedding_cache is None else embedding_cache
    years = sorted(snapshots_by_year)
    per_year = {}
    for year in years:
        per_year[year] = run_single_snapshot(
            snapshots_by_year[year], embedding_cache, alpha=alpha, level_targets=level_targets,
            coarsening=coarsening)
        per_year[year]["parents"] = _laminar_parents(per_year[year]["labels_by_level"], per_year[year]["ids"])

    n_levels = len(level_targets)
    persistent_by_level_year = {}
    events_by_level = {}
    for level_idx in range(n_levels):
        labels_by_year = {y: (per_year[y]["labels_by_level"][level_idx], per_year[y]["ids"]) for y in years}
        ids_by_year = {y: per_year[y]["ids"] for y in years}
        persistent_by_year, events = track_across_snapshots(
            labels_by_year, ids_by_year, prefix=f"L{level_idx}_")
        persistent_by_level_year[level_idx] = persistent_by_year
        events_by_level[level_idx] = events

    return {
        "years": years, "per_year": per_year, "level_targets": level_targets,
        "persistent_by_level_year": persistent_by_level_year,
        "events_by_level": events_by_level,
    }


def build_hierarchy_json(result, year, snap):
    """Assemble the final per-snapshot hierarchy.json structure: for each
    level, for each super-node: persistent id, level, parent id, member ids,
    plus the T4-collapsed coarse hyperedges for that level.

    member_ids are raw concept node ids at every level, not child
    super-node ids. Why: DESIGN_NOTES.md section 11."""
    per_year = result["per_year"][year]
    ids = per_year["ids"]
    level_targets = result["level_targets"]
    n_levels = len(level_targets)

    super_nodes = []
    node_to_persistent = {}  # level_idx -> {node_id: persistent_id}, for building parent-of-raw-node at finest level
    local_to_persistent = {}  # level_idx -> {local_label: persistent_id}

    for level_idx in range(n_levels):
        local_to_persistent[level_idx] = result["persistent_by_level_year"][level_idx][year]

    for level_idx in range(n_levels):
        labels = per_year["labels_by_level"][level_idx]
        clusters = clusters_from_labels(labels, ids)  # local_label -> frozenset(node_ids)
        node_to_persistent[level_idx] = {
            nid: local_to_persistent[level_idx][lab]
            for lab, members in clusters.items() for nid in members
        }

    for level_idx in range(n_levels):
        clusters = clusters_from_labels(per_year["labels_by_level"][level_idx], ids)
        parents = per_year["parents"].get(level_idx)  # fine->coarse local label map, level_idx>0 only
        for local_lab, members in clusters.items():
            pid = local_to_persistent[level_idx][local_lab]
            if level_idx == 0:
                parent_id = None
            else:
                parent_local = parents[local_lab]
                parent_id = local_to_persistent[level_idx - 1][parent_local]

            member_ids = sorted(members)  # raw concept node ids, at every level

            super_nodes.append({
                "id": pid, "level": level_idx, "parent_id": parent_id,
                "member_ids": member_ids, "member_count": len(members),
                "label": None, "gloss": None,  # filled in by T5
            })

    internal_stats_by_level, edge_summary_by_level = {}, {}
    for level_idx in range(n_levels):
        # T4 collapse applied directly to the RAW hyperedges against this
        # level's node->persistent-supernode mapping (avoids compounding
        # approximation from re-collapsing an already-collapsed level).
        coarse_edges, internal_stats, edge_summary = build_coarse_hyperedges(
            snap.hyperedges, node_to_persistent[level_idx], snap)
        internal_stats_by_level[level_idx] = internal_stats
        edge_summary_by_level[level_idx] = {"edges": coarse_edges, "summary": edge_summary}

    return {
        "year": year,
        "super_nodes": super_nodes,
        "coarse_hyperedges_by_level": edge_summary_by_level,
        "internal_stats_by_level": internal_stats_by_level,
    }
