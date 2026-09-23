"""Run the whole README reproduction path in order, stopping at the first
failure.

Usage:
  reproduce_all.py                  run with the real models
  reproduce_all.py --record DIR     also save every model output to DIR
  reproduce_all.py --replay DIR     use the saved outputs in DIR instead of
                                    the models (no model download needed)

Replaying the published outputs checks that everything after the models
reproduces exactly; compare the result with verify_release.py checksums.
See DESIGN_NOTES.md section 23.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tkh.model_outputs import ENV_VAR  # noqa: E402

STEPS = [
    ["scripts/t1_describe.py"],
    ["scripts/run_pipeline.py"],
    ["scripts/t5_apply_labels.py"],
    ["scripts/validate_hierarchy.py"],
    ["-m", "pytest", "tests", "-q"],
    ["scripts/t6_evaluate.py"],
    ["scripts/t6_patch_extrinsic.py"],
    ["scripts/hypergraph_shuffle_null.py"],
    ["scripts/structural_holdout.py"],
    ["scripts/localisation.py"],
    ["scripts/blind_eval.py", "score"],
    ["scripts/make_report_figures.py"],
]


def main():
    env = dict(os.environ)
    args = sys.argv[1:]
    if args:
        if len(args) != 2 or args[0] not in ("--record", "--replay"):
            sys.exit(__doc__)
        env[ENV_VAR] = f"{args[0][2:]}:{Path(args[1]).resolve()}"
        print(f"{ENV_VAR}={env[ENV_VAR]}")
    else:
        env.pop(ENV_VAR, None)

    t_all = time.time()
    for step in STEPS:
        t0 = time.time()
        print(f"=== {' '.join(step)}", flush=True)
        # the unit tests run without the store: they set their own
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
    print(f"all {len(STEPS)} steps done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
