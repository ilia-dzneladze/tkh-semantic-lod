# Multi-resolution semantic abstraction over the TKH hypergraph

Intern-level assessment submission. Full writeup in `report.pdf`, design
rationale in `DESIGN_NOTES.md`, AI tool usage in `AI_USAGE.md`.

## Setup

Needs Python 3.11. On Windows without `python`/`python3` on PATH, use the
`py` launcher.

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` is the full pinned set (55 packages, resolved from the
loose `requirements.in`). Includes a CPU-only build of torch, no GPU
needed. First run downloads two models from Hugging Face and caches them
locally: `sentence-transformers/all-mpnet-base-v2` (~420MB, drives
clustering and retrieval) and `cross-encoder/nli-deberta-v3-base`
(~400MB, used only for the T6 label-faithfulness check).

## Reproducing the pipeline

Run from the repo root. All scripts use the venv's own Python explicitly
rather than relying on activation:

```
# T1: describe the four snapshots (no model download needed)
.venv\Scripts\python.exe scripts\t1_describe.py

# T2+T3+T4: build the laminar hierarchy at every snapshot, with
# cross-snapshot identity tracking and hyperedge collapse. ~85s on the
# reference machine (embedding pass dominates).
.venv\Scripts\python.exe scripts\run_pipeline.py

# Check the output's structural invariants (laminarity, no id collisions,
# parent/child consistency) rather than trusting a clean run
.venv\Scripts\python.exe scripts\validate_hierarchy.py

# Run the unit tests
.venv\Scripts\python.exe -m pytest tests\ -q
```

`run_pipeline.py` writes `outputs/snapshots/<year>/hierarchy.json` for
each of 2020/2022/2024/2026 and one `outputs/temporal_events.json`.

### T5, labeling

```
.venv\Scripts\python.exe scripts\t5_dump_labeling_input.py
```

writes `outputs/snapshots/<year>/labeling_input.json`, one prompt per
level-0/1 super-node with its real member text. The labeling step itself
is not a deterministic script: it needs something (an LLM, in this case
the coding agent reading the dumped files directly, see `AI_USAGE.md`) to
actually read the member lists and write a label and gloss. The already-
labeled `labeling_output.json` for each snapshot is checked into
`outputs/`. To re-merge labels into the hierarchy after regenerating them:

```
.venv\Scripts\python.exe scripts\t5_apply_labels.py
```

### T6, evaluation

```
.venv\Scripts\python.exe scripts\t6_evaluate.py
```

Runs coherence-vs-null, both stability measurements, label faithfulness,
and the extrinsic drill-down eval, and writes `outputs/metrics.json`. Takes
15-20 minutes on CPU, mostly the NLI faithfulness pass (4 snapshots) and
the retrieval candidate-pool embedding. The extrinsic section's final
branching factor was chosen after a parameter sweep documented in
`DESIGN_NOTES.md` section 14; to reproduce just that patch without
re-running the expensive stages:

```
.venv\Scripts\python.exe scripts\t6_patch_extrinsic.py
```

### Figures

```
.venv\Scripts\python.exe scripts\make_report_figures.py
```

Regenerates the three PNGs in `outputs/figures/` directly from
`outputs/metrics.json`, so a figure can't drift from the number it's
illustrating.

## Repo layout

```
data/                    shipped TKH export, questions, ground truth
src/tkh/                 the actual method: io, hypergraph, embeddings,
                          cluster, collapse, temporal, pipeline, labeling
src/tkh/eval/            T6: coherence, stability, faithfulness, extrinsic
scripts/                 runnable entry points, one per pipeline stage
tests/                   unit tests (T4 collapse rule, T3 event matching)
outputs/                 generated: hierarchy.json + labeling per
                          snapshot, temporal_events.json, metrics.json,
                          figures/
report.pdf                the write-up (Deliverable 0, method, evaluation)
DESIGN_NOTES.md           why each non-obvious choice was made
AI_USAGE.md               AI tool usage disclosure
requirements.in/.txt      loose and pinned dependency lists
```

`scripts/t2_build_hierarchy.py` is an early single-snapshot smoke test
from before `run_pipeline.py` existed, superseded by it, kept for
reference rather than as part of the reproduction path.

## Status

This is a first draft, not a finished/tuned result: alpha sits at its
default 0.5 with no ablation yet, the temporal-matching thresholds are
picked by feel, and the level-0 cluster-size skew noted in `report.pdf`
hasn't been addressed. See `report.pdf` section 5 for the specific list of
what's next.
