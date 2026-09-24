"""T3: persistent super-node ids across snapshots.

Each snapshot is clustered on its own, then matched to the next by Jaccard
overlap on the nodes both snapshots contain. Why matching rather than
warm-starting, and what the thresholds do in practice: DESIGN_NOTES.md
section 10.

Event types: birth, death, stable, grow, shrink, merge, split.
"""
from collections import defaultdict

STABLE_JACCARD = 0.5      # "stable" needs at least this overlap...
SIZE_CHANGE_RATIO = 0.2   # ...and a relative size change under this, else grow/shrink
MATCH_THRESHOLD = 0.15    # below this overlap, two clusters aren't the same one


def clusters_from_labels(labels, ids):
    """{label: frozenset of node ids} from a label array and its id list."""
    out = defaultdict(set)
    for nid, lab in zip(ids, labels):
        out[int(lab)].add(nid)
    return {k: frozenset(v) for k, v in out.items()}


def _jaccard(a, b):
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _mint(counter, prefix):
    counter[0] += 1
    return f"{prefix}S{counter[0]:05d}"


def match_snapshots(prev_clusters, curr_clusters, common_ids=None):
    """{(prev label, curr label): Jaccard} for every overlapping pair,
    computed on common_ids only, so nodes new at t+1 don't dilute the
    match. common_ids defaults to every node in prev_clusters."""
    if common_ids is None:
        common_ids = frozenset().union(*prev_clusters.values())
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
    """Give each current cluster a persistent id and log what happened.

    prev_persistent_ids: {prev label: persistent id}, for every prev cluster.
    next_id_counter: a one-element list, the last id number minted.
    prefix: put in front of minted ids. Use a different one per level
        (e.g. "L0_"), or ids minted at different levels collide.

    Each prev cluster names its best-overlapping current cluster as its
    successor, if the overlap reaches MATCH_THRESHOLD. A current cluster
    claimed by one prev cluster keeps its id (stable, grow, shrink, or
    split if the prev cluster also overlaps other current clusters that
    much); one claimed by several is a merge and keeps the biggest one's
    id; one claimed by none is a birth. Unclaimed prev clusters die.
    Returns (curr_persistent_ids, events)."""
    sims = match_snapshots(prev_clusters, curr_clusters, common_ids)

    best_succ = {}  # prev label -> (curr label, jaccard)
    for (pl, cl), j in sims.items():
        if j > best_succ.get(pl, (None, 0.0))[1]:
            best_succ[pl] = (cl, j)
    predecessors_of = defaultdict(list)  # curr label -> prev labels claiming it
    for pl, (cl, j) in best_succ.items():
        if j >= MATCH_THRESHOLD:
            predecessors_of[cl].append(pl)

    curr_persistent_ids, events, claimed_prev = {}, [], set()
    for cl, preds in predecessors_of.items():
        claimed_prev.update(preds)
        curr_size = len(curr_clusters[cl])
        if len(preds) > 1:
            pid = prev_persistent_ids[max(preds, key=lambda pl: len(prev_clusters[pl]))]
            curr_persistent_ids[cl] = pid
            events.append({"type": "merge", "into": pid,
                           "from_persistent_ids": [prev_persistent_ids[pl] for pl in preds],
                           "curr_size": curr_size})
            continue
        pl = preds[0]
        pid = prev_persistent_ids[pl]
        curr_persistent_ids[cl] = pid
        j = best_succ[pl][1]
        split_targets = [c2 for (p2, c2), j2 in sims.items()
                         if p2 == pl and c2 != cl and j2 >= MATCH_THRESHOLD]
        if split_targets:
            events.append({"type": "split", "from": pid, "_into_local": [cl] + split_targets, "jaccard": j})
            continue
        prev_size = len(prev_clusters[pl])
        ratio = (curr_size - prev_size) / prev_size if prev_size else 1.0
        if j >= STABLE_JACCARD and abs(ratio) < SIZE_CHANGE_RATIO:
            etype = "stable"
        elif ratio >= SIZE_CHANGE_RATIO:
            etype = "grow"
        else:
            etype = "shrink"
        events.append({"type": etype, "id": pid, "prev_size": prev_size, "curr_size": curr_size, "jaccard": j})

    for cl in curr_clusters:
        if cl not in curr_persistent_ids:
            pid = _mint(next_id_counter, prefix)
            curr_persistent_ids[cl] = pid
            events.append({"type": "birth", "id": pid, "curr_size": len(curr_clusters[cl])})

    for pl, pid in prev_persistent_ids.items():
        if pl not in claimed_prev:
            events.append({"type": "death", "id": pid, "prev_size": len(prev_clusters[pl])})

    # split targets only have ids once every current cluster has one
    for e in events:
        if e["type"] == "split":
            e["into"] = [curr_persistent_ids[c] for c in e.pop("_into_local")]

    return curr_persistent_ids, events


def track_across_snapshots(labels_by_year, prefix=""):
    """labels_by_year: {year: (labels, ids)} for one level. Every cluster
    in the first year is born; later years are matched to the year before.
    prefix: see classify_events. Returns (persistent ids by year, events)."""
    years = sorted(labels_by_year)
    counter = [0]
    prev_clusters = clusters_from_labels(*labels_by_year[years[0]])
    prev_persistent = {lab: _mint(counter, prefix) for lab in prev_clusters}
    persistent_by_year = {years[0]: prev_persistent}
    all_events = [{"type": "birth", "id": pid, "curr_size": len(prev_clusters[lab]),
                   "year": years[0], "note": "initial snapshot"} for lab, pid in prev_persistent.items()]

    for prev_year, year in zip(years, years[1:]):
        curr_clusters = clusters_from_labels(*labels_by_year[year])
        common = frozenset(labels_by_year[prev_year][1]) & frozenset(labels_by_year[year][1])
        curr_persistent, events = classify_events(
            prev_clusters, curr_clusters, prev_persistent, counter, prefix=prefix, common_ids=common)
        for e in events:
            e["year"] = year
        all_events.extend(events)
        persistent_by_year[year] = curr_persistent
        prev_clusters, prev_persistent = curr_clusters, curr_persistent

    return persistent_by_year, all_events
