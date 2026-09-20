"""T2: joint structural+semantic affinity -> laminar multi-level hierarchy.

combine_affinities: DESIGN_NOTES.md section 7.
sparse_upgma, forced merges: DESIGN_NOTES.md section 8.
"""
import heapq
import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster


def _normalize_affinity(A):
    """Scale nonzero values to [0, 1] via division by the 99th percentile
    (percentile rather than max to avoid one outlier edge compressing
    everything else near 0)."""
    if A.nnz == 0:
        return A.tocsr()
    vals = A.data
    scale = np.percentile(vals, 99) or vals.max() or 1.0
    A2 = A.copy().tocsr()
    A2.data = np.clip(A2.data / scale, 0.0, 1.0)
    return A2


def combine_affinities(A_struct, A_sem, alpha=0.5):
    """Union of two sparse graphs (assumed same shape/index space), combined
    as a weighted sum. Missing entries in either graph contribute 0 for that
    signal (no imputation)."""
    An = _normalize_affinity(A_struct)
    Bn = _normalize_affinity(A_sem)
    combined = (alpha * An + (1 - alpha) * Bn).tocsr()
    combined.eliminate_zeros()
    return combined


def sparse_upgma(A, n):
    """Average-linkage agglomerative clustering on a sparse similarity graph.

    Returns (Z, forced_mask):
      Z: (n-1, 4) scipy linkage matrix [id1, id2, distance, cluster_size].
         distance = 1 - similarity, so higher similarity merges first.
      forced_mask: (n-1,) bool, True where a merge had no graph edge as
         evidence (components merged purely to keep the tree connected).
    """
    A = A.tocsr()
    adjacency = [dict() for _ in range(n)]
    for i in range(n):
        start, end = A.indptr[i], A.indptr[i + 1]
        for jj in range(start, end):
            j = A.indices[jj]
            if j == i:
                continue
            adjacency[i][j] = A.data[jj]

    alive = set(range(n))
    size = {i: 1 for i in range(n)}
    next_id = n
    heap = []
    for i in range(n):
        for j, s in adjacency[i].items():
            if j > i:
                heapq.heappush(heap, (-s, i, j))

    Z = []
    forced = []
    merges_needed = n - 1

    while len(Z) < merges_needed:
        a = b = None
        sim = None
        while heap:
            neg_s, i, j = heapq.heappop(heap)
            if i in alive and j in alive:
                a, b, sim = i, j, -neg_s
                break
        if a is None:
            # no real edges left: force-merge the two smallest remaining
            # components (no structural/semantic evidence for this merge)
            remaining = sorted(alive, key=lambda c: size[c])
            a, b = remaining[0], remaining[1]
            sim = 0.0
            is_forced = True
        else:
            is_forced = False

        new_size = size[a] + size[b]
        Z.append([a, b, 1.0 - sim, new_size])
        forced.append(is_forced)

        # Lance-Williams average-linkage update. Missing edge = similarity 0,
        # not imputed. See DESIGN_NOTES.md section 8 for why that specific
        # choice keeps the dendrogram distances non-decreasing.
        merged_neighbors = {}
        for x, s in adjacency[a].items():
            if x in alive and x != b:
                merged_neighbors[x] = merged_neighbors.get(x, 0.0) + size[a] * s
        for x, s in adjacency[b].items():
            if x in alive and x != a:
                merged_neighbors[x] = merged_neighbors.get(x, 0.0) + size[b] * s

        alive.discard(a)
        alive.discard(b)
        new_id = next_id
        next_id += 1
        size[new_id] = new_size
        adjacency.append({})
        for x, weighted_sum in merged_neighbors.items():
            new_sim = weighted_sum / new_size
            adjacency[new_id][x] = new_sim
            adjacency[x][new_id] = new_sim
            heapq.heappush(heap, (-new_sim, min(x, new_id), max(x, new_id)))
        alive.add(new_id)

    return np.array(Z, dtype=float), np.array(forced, dtype=bool)


def cut_to_k_clusters(Z, n, k):
    """Cut the dendrogram to (at most) k clusters. Returns a length-n array
    of cluster labels (0-indexed, contiguous)."""
    k = min(k, n)
    labels = fcluster(Z, t=k, criterion="maxclust")
    uniq = {old: i for i, old in enumerate(sorted(set(labels)))}
    return np.array([uniq[l] for l in labels])
