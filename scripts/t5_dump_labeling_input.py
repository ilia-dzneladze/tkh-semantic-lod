"""T5 step 1: write the labelling prompts (levels 0-1) for every snapshot.
No LLM call here; this writes exactly what a labeller would see.

Usage:
  t5_dump_labeling_input.py                   refresh the shipped set's
      prompts in outputs/snapshots/<year>/labeling_input.json
  t5_dump_labeling_input.py --out DIR [--template FILE]
      start a new label set in DIR: <year>/labeling_input.json, a
      paste-ready <year>/labeller_request.md for any LLM or human, and
      prompt_template.txt (the template used, so the set can later be
      checked against the clusters it was written for).

A custom template is a text file with str.format placeholders from
{year}, {sample_n}, {total_n}, {type_line}, {member_lines}; it must use
{member_lines}, and literal braces must be doubled. The labeller is shown
the same seeded sample of members whatever the template, because the
faithfulness check holds out exactly the members the labeller didn't see.
See DESIGN_NOTES.md section 24.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.io import load_tkh, build_all_snapshots, SNAPSHOT_CUTOFFS  # noqa: E402
from tkh.labeling import (  # noqa: E402
    write_labeling_input, write_labeller_request, validate_template, LABEL_PROMPT_TEMPLATE, TEMPLATE_FILE)

DATA_PATH = ROOT / "data" / "tkh_collection10.json"
SHIPPED = ROOT / "outputs" / "snapshots"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="directory for a new label set (default: the shipped set)")
    ap.add_argument("--template", type=Path, help="custom prompt template (needs --out)")
    args = ap.parse_args()
    if args.template and not args.out:
        ap.error("--template needs --out, so the shipped set's prompts stay as they were labelled")

    template = LABEL_PROMPT_TEMPLATE
    if args.template:
        try:
            template = validate_template(args.template.read_text(encoding="utf-8"))
        except ValueError as e:
            ap.error(f"{args.template}: {e}")
    out_root = args.out.resolve() if args.out else SHIPPED
    if args.out:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / TEMPLATE_FILE).write_text(template, encoding="utf-8")

    snapshots = build_all_snapshots(load_tkh(DATA_PATH), cutoffs=SNAPSHOT_CUTOFFS)
    for year in sorted(snapshots):
        hierarchy = json.loads((SHIPPED / str(year) / "hierarchy.json").read_text(encoding="utf-8"))
        year_dir = out_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        entries = write_labeling_input(hierarchy, snapshots[year], year, year_dir / "labeling_input.json",
                                       levels=(0, 1), template=template)
        msg = f"{year}: wrote {len(entries)} prompts -> {year_dir / 'labeling_input.json'}"
        if args.out:
            write_labeller_request(entries, year, year_dir / "labeller_request.md")
            msg += f", request -> {year_dir / 'labeller_request.md'}"
        print(msg)
    if args.out:
        print(f"\nGive each <year>/labeller_request.md to your labeller, save each reply as "
              f"labels_<year>.json in one folder, then:\n"
              f"  python scripts/t5_import_labels.py REPLY_FOLDER --set {args.out}")


if __name__ == "__main__":
    main()
