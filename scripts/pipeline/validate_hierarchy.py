"""Check the structure of every outputs/snapshots/<year>/hierarchy.json.

member_ids are concept node ids at every level and the tree is carried by
parent_id, so laminarity (P1) comes down to: the children of each
super-node sit one level below it and split its members exactly, with no
overlap and nothing left over.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import OUTPUTS  # noqa: E402


def validate_one(path):
    super_nodes = json.loads(path.read_text(encoding="utf-8"))["super_nodes"]
    errors = []
    by_id = {}
    for sn in super_nodes:
        if sn["id"] in by_id:
            errors.append(f"duplicate super-node id {sn['id']}")
        by_id[sn["id"]] = sn
    children_of = defaultdict(list)
    for sn in super_nodes:
        if sn["parent_id"] is not None:
            children_of[sn["parent_id"]].append(sn)
    max_level = max(sn["level"] for sn in super_nodes)

    for sn in super_nodes:
        sid = sn["id"]
        if sn["member_count"] != len(sn["member_ids"]):
            errors.append(f"{sid}: member_count {sn['member_count']} != {len(sn['member_ids'])} member_ids")
        if (sn["parent_id"] is None) != (sn["level"] == 0):
            errors.append(f"{sid}: level {sn['level']} with parent {sn['parent_id']}")
        elif sn["parent_id"] is not None and sn["parent_id"] not in by_id:
            errors.append(f"{sid}: parent {sn['parent_id']} not found")
        if sn["level"] == max_level:
            continue
        kids = children_of[sid]
        if not kids:
            errors.append(f"{sid} (level {sn['level']}): no children")
            continue
        if any(k["level"] != sn["level"] + 1 for k in kids):
            errors.append(f"{sid}: a child isn't one level below it")
        child_members = Counter(m for k in kids for m in k["member_ids"])
        if any(c > 1 for c in child_members.values()):
            errors.append(f"{sid}: children's member sets overlap")
        if set(child_members) != set(sn["member_ids"]):
            errors.append(f"{sid}: children's members don't equal its own "
                          f"({len(set(sn['member_ids']) - set(child_members))} missing, "
                          f"{len(set(child_members) - set(sn['member_ids']))} extra)")
    return errors


def main():
    failed = False
    for path in sorted((OUTPUTS / "snapshots").glob("*/hierarchy.json")):
        errors = validate_one(path)
        print(f"{path.parent.name}: {'OK' if not errors else f'{len(errors)} ERRORS'}")
        for e in errors[:10]:
            print(f"    {e}")
        failed |= bool(errors)
    if failed:
        sys.exit("\nVALIDATION FAILED")
    print("\nall snapshots valid")


if __name__ == "__main__":
    main()
