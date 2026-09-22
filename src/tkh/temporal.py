"""T3: temporal coupling -- persistent super-node identity across snapshots.

Each snapshot is clustered independently, then matched to the next one by
Jaccard overlap on the nodes both snapshots share. Why matching instead of
warm-starting, and what the
thresholds below mean in practice: DESIGN_NOTES.md section 10.

Event types: birth, death, stable, grow, shrink, merge, split.
"""
from collections import defaultdict

STABLE_JACCARD = 0.5      # >= this and roughly same size -> "stable"
MATCH_THRESHOLD = 0.15    # below this, no match is considered adequate
SIZE_CHANGE_RATIO = 0.2   # relative size change beyond which stable -> grow/shrink


def clusters_from_labels(labels, ids):
    """label array + parallel node-id list -> {label: frozenset(node_ids)}."""
    out = defaultdict(set)
    for nid, lab in zip(ids, labels):
        out[int(lab)].add(nid)
    return {k: frozenset(v) for k, v in out.items()}


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def match_snapshots(prev_clusters, curr_clusters, common_ids=None):
    """prev_clusters, curr_clusters: {local_label: frozenset(node_ids)}.

    Returns {(prev_label, curr_label): jaccard} for every pair with nonzero
    overlap. Jaccard is computed on nodes present in both snapshots
    (common_ids), so nodes new at t+1 don't dilute the match. If
    common_ids is None, the union of all prev-cluster members is used.
    See DESIGN_NOTES.md section 10.
    """
    if common_ids is None:
        common_ids = frozenset().union(*prev_clusters.values()) if prev_clusters else frozenset()
    prev_r = {pl: pset & common_ids for pl, pset in prev_clusters.items()}
    curr_r = {cl: cset & common_ids for cl, cset in curr_clusters.items()}
    sims = {}
    for pl, pset in prev_r.items():
        for cl, cset in curr_r.items():
            j = _jaccard(pset, cset)
            if j > 0:
                sims[(pl, cl)] = j
    return sims


def classify_events(prev_clusters, curr_clusters, prev_persistent_ids, next_id_counter, prefix="",
                    common_ids=None):
    """Assign persistent ids to curr_clusters and log events.

    prev_persistent_ids: {prev_local_label: persistent_id}
    next_id_counter: mutable single-element list [int], for minting new ids
    prefix: prepended to minted ids (e.g. "L0_") -- callers tracking
        multiple levels MUST use a distinct prefix per level, otherwise
        ids minted independently at different levels collide (same string,
        different entities).
    common_ids: node ids present in both snapshots, passed to match_snapshots.

    Returns (curr_persistent_ids, events) where events is a list of dicts.
    """
    sims = match_snapshots(prev_clusters, curr_clusters, common_ids)

    best_succ = defaultdict(lambda: (None, 0.0))   # prev_label -> (curr_label, jaccard)
    best_pred = defaultdict(lambda: (None, 0.0))   # curr_label -> (prev_label, jaccard)
    for (pl, cl), j in sims.items():
        if j > best_succ[pl][1]:
            best_succ[pl] = (cl, j)
        if j > best_pred[cl][1]:
            best_pred[cl] = (pl, j)

    # group curr labels by which prev labels claim them as best successor
    predecessors_of = defaultdict(list)  # curr_label -> [prev_label, ...]
    for pl, (cl, j) in best_succ.items():
        if cl is not None and j >= MATCH_THRESHOLD:
            predecessors_of[cl].append(pl)

    curr_persistent_ids = {}
    events = []

    def mint_id():
        next_id_counter[0] += 1
        return f"{prefix}S{next_id_counter[0]:05d}"

    handled_prev = set()

    for cl, preds in predecessors_of.items():
        curr_size = len(curr_clusters[cl])
        if len(preds) == 1:
            pl = preds[0]
            handled_prev.add(pl)
            j = best_succ[pl][1]
            prev_size = len(prev_clusters[pl])
            pid = prev_persistent_ids.get(pl, None) or mint_id()
            # split check: does pl also feed other curr clusters above threshold?
            split_targets = [c2 for (p2, c2), j2 in sims.items()
                              if p2 == pl and c2 != cl and j2 >= MATCH_THRESHOLD]
            if split_targets:
                curr_persistent_ids[cl] = pid
                events.append({"type": "split", "from": pid,
                                "_into_local": [cl] + split_targets, "jaccard": j})
            else:
                ratio = (curr_size - prev_size) / prev_size if prev_size else 1.0
                if j >= STABLE_JACCARD and abs(ratio) < SIZE_CHANGE_RATIO:
                    etype = "stable"
                elif ratio >= SIZE_CHANGE_RATIO:
                    etype = "grow"
                else:
                    etype = "shrink"
                curr_persistent_ids[cl] = pid
                events.append({"type": etype, "id": pid,
                                "prev_size": prev_size, "curr_size": curr_size, "jaccard": j})
        else:
            handled_prev.update(preds)
            surviving_pid = max(preds, key=lambda pl: len(prev_clusters[pl]))
            pid = prev_persistent_ids.get(surviving_pid, None) or mint_id()
            curr_persistent_ids[cl] = pid
            merged_pids = [prev_persistent_ids.get(pl, pl) for pl in preds]
            events.append({"type": "merge", "into": pid,
                            "from_persistent_ids": merged_pids, "curr_size": curr_size})

    # curr clusters with no adequate predecessor at all -> birth
    for cl in curr_clusters:
        if cl not in curr_persistent_ids:
            pid = mint_id()
            curr_persistent_ids[cl] = pid
            events.append({"type": "birth", "id": pid, "curr_size": len(curr_clusters[cl])})

    # prev clusters never claimed as anyone's predecessor -> death
    for pl, pid in prev_persistent_ids.items():
        if pl not in handled_prev:
            events.append({"type": "death", "id": pid, "prev_size": len(prev_clusters[pl])})

    # split targets are only known by local label until every curr cluster has an id
    for e in events:
        if e["type"] == "split":
            e["into"] = [curr_persistent_ids[c] for c in e.pop("_into_local")]

    return curr_persistent_ids, events


def track_across_snapshots(labels_by_year, ids_by_year, prefix=""):
    """labels_by_year / ids_by_year: {year: (labels_array, id_list)} for ONE level.

    prefix: prepended to every minted persistent id. Callers tracking
        multiple levels MUST pass a distinct prefix per level (e.g. "L0_",
        "L1_") -- otherwise ids minted independently at different levels
        collide as identical strings for unrelated super-nodes.

    Returns (persistent_ids_by_year, all_events).
    """
    years = sorted(labels_by_year)
    next_id_counter = [0]
    persistent_by_year = {}
    all_events = []

    first_year = years[0]
    clusters0 = clusters_from_labels(*labels_by_year[first_year])
    persistent0 = {}
    for lab in clusters0:
        next_id_counter[0] += 1
        persistent0[lab] = f"{prefix}S{next_id_counter[0]:05d}"
    persistent_by_year[first_year] = persistent0
    for lab, pid in persistent0.items():
        all_events.append({"type": "birth", "id": pid, "curr_size": len(clusters0[lab]),
                            "year": first_year, "note": "initial snapshot"})

    prev_clusters = clusters0
    prev_persistent = persistent0
    prev_year = first_year
    for year in years[1:]:
        curr_clusters = clusters_from_labels(*labels_by_year[year])
        common = frozenset(ids_by_year[prev_year]) & frozenset(ids_by_year[year])
        curr_persistent, events = classify_events(
            prev_clusters, curr_clusters, prev_persistent, next_id_counter, prefix=prefix,
            common_ids=common)
        for e in events:
            e["year"] = year
        all_events.extend(events)
        persistent_by_year[year] = curr_persistent
        prev_clusters, prev_persistent, prev_year = curr_clusters, curr_persistent, year

    return persistent_by_year, all_events
