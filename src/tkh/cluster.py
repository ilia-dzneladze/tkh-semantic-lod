"""T2: blend the structural and semantic graphs, build one average-linkage
tree over them, and cut it into levels.

combine_affinities: DESIGN_NOTES.md section 7.
sparse_upgma and forced merges: DESIGN_NOTES.md section 8.
"""
import heapq
import numpy as np
from scipy.cluster.hierarchy import fcluster


def _normalize_affinity(A):
    """Divide by the 99th percentile of the nonzero values and clip to
    [0, 1], so one outlier edge doesn't squash everything else toward 0."""
    if A.nnz == 0:
        return A.tocsr()
    scale = np.percentile(A.data, 99) or A.data.max() or 1.0
    A2 = A.copy().tocsr()
    A2.data = np.clip(A2.data / scale, 0.0, 1.0)
    return A2


def combine_affinities(A_struct, A_sem, alpha):
    """alpha * structure + (1 - alpha) * semantics, each normalised first.
    A pair missing from one graph gets 0 from that graph."""
    An = _normalize_affinity(A_struct)
    Bn = _normalize_affinity(A_sem)
    combined = (alpha * An + (1 - alpha) * Bn).tocsr()
    combined.eliminate_zeros()
    return combined


def sparse_upgma(A, n, sizes=None):
    """Average-linkage agglomerative clustering on a sparse similarity graph.

    sizes: optional initial weight per point (default 1 each). With member
    counts here, clustering super-nodes averages over underlying nodes.

    Returns (Z, forced):
      Z: scipy linkage matrix, rows [id1, id2, 1 - similarity, n points].
      forced: True for each merge that had no edge behind it (the graph
         ran out of edges and two components were joined to finish the tree).
    """
    A = A.tocsr()
    adjacency = [dict() for _ in range(n)]
    for i in range(n):
        for jj in range(A.indptr[i], A.indptr[i + 1]):
            if A.indices[jj] != i:
                adjacency[i][A.indices[jj]] = A.data[jj]

    alive = set(range(n))
    size = {i: (1 if sizes is None else sizes[i]) for i in range(n)}  # linkage weights
    npts = {i: 1 for i in range(n)}  # point counts, for Z's 4th column
    next_id = n
    heap = []
    for i in range(n):
        for j, s in adjacency[i].items():
            if j > i:
                heapq.heappush(heap, (-s, i, j))

    Z, forced = [], []
    while len(Z) < n - 1:
        a = None
        while heap:
            neg_s, i, j = heapq.heappop(heap)
            if i in alive and j in alive:
                a, b, sim = i, j, -neg_s
                break
        is_forced = a is None
        if is_forced:
            # no edges left: join the two smallest components, with no evidence
            a, b = sorted(alive, key=lambda c: size[c])[:2]
            sim = 0.0

        new_id = next_id
        next_id += 1
        size[new_id] = size[a] + size[b]
        npts[new_id] = npts[a] + npts[b]
        Z.append([a, b, 1.0 - sim, npts[new_id]])
        forced.append(is_forced)

        # average-linkage update; a missing edge counts as similarity 0,
        # which keeps merge heights monotone (DESIGN_NOTES.md section 8)
        merged_neighbors = {}
        for x, s in adjacency[a].items():
            if x in alive and x != b:
                merged_neighbors[x] = merged_neighbors.get(x, 0.0) + size[a] * s
        for x, s in adjacency[b].items():
            if x in alive and x != a:
                merged_neighbors[x] = merged_neighbors.get(x, 0.0) + size[b] * s

        alive.discard(a)
        alive.discard(b)
        adjacency.append({})
        for x, weighted_sum in merged_neighbors.items():
            new_sim = weighted_sum / size[new_id]
            adjacency[new_id][x] = new_sim
            adjacency[x][new_id] = new_sim
            heapq.heappush(heap, (-new_sim, min(x, new_id), max(x, new_id)))
        alive.add(new_id)

    return np.array(Z, dtype=float), np.array(forced, dtype=bool)


def cut_to_k_clusters(Z, n, k):
    """Cut the tree into at most k clusters. Returns labels 0..k-1 per point."""
    k = min(k, n)
    labels = fcluster(Z, t=k, criterion="maxclust")
    uniq = {old: i for i, old in enumerate(sorted(set(labels)))}
    return np.array([uniq[l] for l in labels])
