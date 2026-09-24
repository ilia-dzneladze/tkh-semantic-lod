"""Structural invariant checks on generated hierarchy.json files.

member_ids are raw concept node ids at EVERY level (see pipeline.py
build_hierarchy_json); the tree is carried entirely by parent_id. So the
real laminarity check is: a super-node's children (other super-nodes with
parent_id == this id, at level+1) must EXACTLY PARTITION its member_ids --
no overlap, no leftover, no extra. That's P1 made concrete and checkable,
not just assumed.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "snapshots"


def validate_one(path):
    h = json.loads(path.read_text(encoding="utf-8"))
    super_nodes = h["super_nodes"]
    by_id = {}
    errors = []

    for sn in super_nodes:
        if sn["id"] in by_id:
            errors.append(f"duplicate super-node id {sn['id']} (cross-level collision?)")
        by_id[sn["id"]] = sn

    max_level = max(sn["level"] for sn in super_nodes)
    children_of = defaultdict(list)
    for sn in super_nodes:
        if sn["parent_id"] is not None:
            children_of[sn["parent_id"]].append(sn)

    for sn in super_nodes:
        if sn["id"] in sn["member_ids"]:
            errors.append(f"{sn['id']}: self-reference in member_ids")
        if sn["member_count"] != len(sn["member_ids"]):
            errors.append(f"{sn['id']}: member_count={sn['member_count']} != "
                           f"len(member_ids)={len(sn['member_ids'])}")
        if sn["parent_id"] is not None and sn["parent_id"] not in by_id:
            errors.append(f"{sn['id']}: parent {sn['parent_id']} not found")

        if sn["level"] < max_level:
            kids = children_of.get(sn["id"], [])
            if not kids:
                errors.append(f"{sn['id']} (level {sn['level']}): no children at level+1")
                continue
            own = set(sn["member_ids"])
            union = set()
            overlap = set()
            seen = set()
            for k in kids:
                kset = set(k["member_ids"])
                overlap |= (seen & kset)
                seen |= kset
                union |= kset
            if overlap:
                errors.append(f"{sn['id']}: children's member sets overlap ({len(overlap)} ids)")
            if union != own:
                missing = own - union
                extra = union - own
                if missing:
                    errors.append(f"{sn['id']}: {len(missing)} members not covered by any child")
                if extra:
                    errors.append(f"{sn['id']}: children cover {len(extra)} members not in parent")

    return errors


def main():
    all_errors = {}
    for hpath in sorted(OUT_DIR.glob("*/hierarchy.json")):
        errors = validate_one(hpath)
        status = "OK" if not errors else f"{len(errors)} ERRORS"
        print(f"{hpath.parent.name}: {status}")
        for e in errors[:10]:
            print(f"    {e}")
        if errors:
            all_errors[str(hpath)] = errors

    if all_errors:
        print("\nVALIDATION FAILED")
        sys.exit(1)
    print("\nall snapshots valid")


if __name__ == "__main__":
    main()
