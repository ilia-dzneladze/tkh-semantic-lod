"""Reproduce everything with one command, from a fresh clone.

  py -3.11 scripts/reproduce_all.py          (Windows)
  python3.11 scripts/reproduce_all.py        (macOS / Linux)

Run with any Python 3.11. If the repo's .venv doesn't exist it is created
and the pinned requirements are installed (on Linux, the CPU build of
torch first), then the script re-runs itself inside the venv and runs the
whole pipeline in order: T1 statistics, the hierarchies and temporal
events (T2-T4), the labels, validation and unit tests, every evaluation
(T6), the blind-rating scores and the figures. It stops at the first
failure. Later runs reuse the venv and skip installing unless
requirements.txt changed.

Options:
  --labels DIR     use your own label set instead of the shipped one
                   (see t5_dump_labeling_input.py --out)
  --blind DIR      use your own blind-rating set (see blind_eval.py make)
  --record DIR     also save every model output to DIR
  --replay DIR     use saved model outputs instead of the models, e.g. the
                   published release/model_outputs (no model download)
  --no-setup       use the current Python as it is; don't touch .venv
  --dry-run        print the steps and settings, run nothing, install nothing

The TKH export isn't in the repo. Unzip the data.zip that came with the
brief into data/ first (see the README).

Everything lands in outputs/ (figures in outputs/figures/). Afterwards,
`python scripts/verify_release.py checksums` compares the results with the
published ones. See DESIGN_NOTES.md sections 23 and 24.
"""
# standard library only up here: this has to start on a bare Python
import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
MARKER = VENV / ".tkh-installed"
ENV_VAR = "TKH_MODEL_OUTPUTS"  # same as tkh.model_outputs.ENV_VAR
TORCH_CPU = ["torch==2.14.0", "--index-url", "https://download.pytorch.org/whl/cpu"]
# SHA-256 of the data files my results came from
DATA_FILES = {
    "tkh_collection10.json": "5cc507de0432c111c2bbbc5b50899f952fd24f515313c7c4e99450d8d0211b3b",
    "questions.csv": "ad75e968e83c7afa5cb4048b3b792c85b0125c6535105ef3830310092bf3c326",
    "ground_truth.json": "ea4987e1e77931902c9907f44172b59f1f3d24a78cacf83bfd29149994f7b88a",
    "collection10_articles.csv": "64b52a7bfdf6379647be299661e0465977374e1001a045d89bbb82eebb77abdc",
}


def check_data():
    """Stop if a data file is missing; warn if one differs from mine."""
    data = ROOT / "data"
    missing = [name for name in DATA_FILES if not (data / name).is_file()]
    if missing:
        sys.exit(f"missing in {data}: {', '.join(missing)}\n"
                 "The TKH export isn't in the repo. Unzip the data.zip that came with the brief "
                 "into data/ so those files sit directly inside it (see the README).")
    for name, want in DATA_FILES.items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != want:
            print(f"warning: data/{name} differs from the file my results came from "
                  f"(line endings alone can do this); outputs may not match the checksums", flush=True)


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_environment():
    """Create .venv and install requirements if needed. Returns True when
    this process isn't the venv's Python and should hand over to it."""
    if sys.version_info[:2] != (3, 11):
        print(f"warning: requirements.txt was resolved on Python 3.11; this is "
              f"{platform.python_version()}. Install may fail.", flush=True)
    py = venv_python()
    if not py.exists():
        print(f"creating {VENV}", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    want = hashlib.sha256(REQUIREMENTS.read_bytes() + platform.system().encode()).hexdigest()
    if not MARKER.exists() or MARKER.read_text().strip() != want:
        print("installing pinned requirements (about 15 minutes the first time, mostly torch)", flush=True)
        pip = [str(py), "-m", "pip", "install", "--disable-pip-version-check"]
        if platform.system() == "Linux":
            subprocess.run(pip + TORCH_CPU, check=True)
        subprocess.run(pip + ["-r", str(REQUIREMENTS)], check=True)
        MARKER.write_text(want)
    return Path(sys.executable).resolve() != py.resolve()


def steps(args):
    labels = ["--labels", str(args.labels)] if args.labels else []
    blind = ["--dir", str(args.blind)] if args.blind else []
    return [
        ["scripts/pipeline/t1_describe.py"],
        ["scripts/pipeline/run_pipeline.py"],
        ["scripts/pipeline/t5_apply_labels.py", *labels],
        ["scripts/pipeline/validate_hierarchy.py"],
        ["-m", "pytest", "tests", "-q"],
        ["scripts/pipeline/t6_evaluate.py"],
        ["scripts/pipeline/t6_extrinsic.py"],
        ["scripts/pipeline/hypergraph_shuffle_null.py"],
        ["scripts/pipeline/structural_holdout.py"],
        ["scripts/pipeline/localisation.py"],
        ["scripts/pipeline/blind_eval.py", "score", *blind],
        ["scripts/pipeline/make_report_figures.py"],
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--record", type=Path, metavar="DIR")
    mode.add_argument("--replay", type=Path, metavar="DIR")
    ap.add_argument("--labels", type=Path, metavar="DIR")
    ap.add_argument("--blind", type=Path, metavar="DIR")
    ap.add_argument("--no-setup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    check_data()
    if not (args.no_setup or args.dry_run) and ensure_environment():
        # hand over to the venv's Python with the same arguments
        sys.exit(subprocess.run([str(venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]]).returncode)

    env = dict(os.environ)
    env.pop(ENV_VAR, None)
    if args.record or args.replay:
        kind, path = ("record", args.record) if args.record else ("replay", args.replay)
        env[ENV_VAR] = f"{kind}:{path.resolve()}"
        print(f"{ENV_VAR}={env[ENV_VAR]}")
    for flag, path in (("--labels", args.labels), ("--blind", args.blind)):
        if path and not path.is_dir():
            sys.exit(f"{flag} {path}: no such directory")

    todo = steps(args)
    if args.dry_run:
        if ENV_VAR not in env:
            print(f"{ENV_VAR} unset: real models")
        for step in todo:
            print("  python " + " ".join(step))
        return
    t_all = time.time()
    for step in todo:
        t0 = time.time()
        print(f"=== {' '.join(step)}", flush=True)
        # the unit tests set their own model-output mode
        step_env = {k: v for k, v in env.items() if k != ENV_VAR} if step[0] == "-m" else env
        result = subprocess.run([sys.executable, "-W", "ignore", *step], cwd=ROOT, env=step_env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        tail = [ln for ln in (result.stdout + result.stderr).splitlines()
                if ln.strip() and "Batches" not in ln and "Loading weights" not in ln][-2:]
        for ln in tail:
            print(f"    {ln[:160]}")
        print(f"    exit {result.returncode}, {time.time() - t0:.0f}s", flush=True)
        if result.returncode != 0:
            sys.exit(f"stopped: {' '.join(step)} failed")
    print(f"all {len(todo)} steps done in {(time.time() - t_all) / 60:.1f} min; "
          f"results in outputs/, figures in outputs/figures/")


if __name__ == "__main__":
    main()
