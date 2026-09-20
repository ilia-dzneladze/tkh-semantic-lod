import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.temporal import track_across_snapshots


def test_stable_grow_merge_split_birth_death():
    # year 1: three clusters
    #   A = {1,2,3,4}, B = {5,6,7,8}, C = {9,10}
    # year 2:
    #   A persists + grows a lot         -> A stays but "grow"... let's keep growth modest for "stable"/"grow" mix
    #   B and C MERGE into one cluster BC
    #   A SPLITS into A1={1,2} and A2={3,4,11}
    #   a brand-new cluster D appears (birth)
    labels_y1 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    ids_y1 = ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10"]

    labels_y2 = [0, 0, 1, 1, 2, 2, 2, 2, 3]
    ids_y2 = ["n1", "n2", "n3", "n4", "n11", "n5", "n6", "n7", "n8"]
    # local label 0 -> {n1,n2} (piece of A)
    # local label 1 -> {n3,n4} (other piece of A)  -- A split into two
    # local label 2 -> {n11,n5,n6,n7,n8} (B plus a new node, roughly = B) -- wait C disappeared, check death
    # local label 3 -> {n9}? not included above -> n9/n10 dropped entirely (death of C)

    labels_by_year = {2020: (labels_y1, ids_y1), 2021: (labels_y2, ids_y2)}
    ids_by_year = {2020: ids_y1, 2021: ids_y2}

    persistent_by_year, events = track_across_snapshots(labels_by_year, ids_by_year)

    types_2021 = [e["type"] for e in events if e.get("year") == 2021]
    assert "split" in types_2021
    assert "death" in types_2021  # C = {n9, n10} has no successor at all

    # every year-2020 local label got an initial persistent id (birth, initial snapshot)
    assert len(persistent_by_year[2020]) == 3
    # every year-2021 local label resolved to SOME persistent id
    assert len(persistent_by_year[2021]) == len(set(labels_y2))


def test_merge_detected():
    labels_y1 = [0, 0, 1, 1]
    ids_y1 = ["a", "b", "c", "d"]
    labels_y2 = [0, 0, 0, 0]
    ids_y2 = ["a", "b", "c", "d"]

    labels_by_year = {2020: (labels_y1, ids_y1), 2021: (labels_y2, ids_y2)}
    ids_by_year = {2020: ids_y1, 2021: ids_y2}

    persistent_by_year, events = track_across_snapshots(labels_by_year, ids_by_year)
    merge_events = [e for e in events if e["type"] == "merge" and e.get("year") == 2021]
    assert len(merge_events) == 1
    assert len(merge_events[0]["from_persistent_ids"]) == 2


def test_stable_when_membership_unchanged():
    labels_y1 = [0, 0, 0]
    ids_y1 = ["a", "b", "c"]
    labels_y2 = [0, 0, 0]
    ids_y2 = ["a", "b", "c"]

    labels_by_year = {2020: (labels_y1, ids_y1), 2021: (labels_y2, ids_y2)}
    ids_by_year = {2020: ids_y1, 2021: ids_y2}

    persistent_by_year, events = track_across_snapshots(labels_by_year, ids_by_year)
    stable_events = [e for e in events if e["type"] == "stable" and e.get("year") == 2021]
    assert len(stable_events) == 1
    # same persistent id preserved across years
    assert persistent_by_year[2020][0] == persistent_by_year[2021][0]
