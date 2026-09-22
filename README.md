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

Installing takes about 15 minutes, mostly torch. `requirements.txt` is
the full pinned set (55 packages, resolved from the loose
`requirements.in`). Includes a CPU-only build of torch, no GPU
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
# cross-snapshot identity tracking and hyperedge collapse. 1.5 to 5 minutes
# on a laptop CPU (embedding pass dominates).
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
level-0/1 super-node with a seeded random sample of 25 members and the
node-type counts of all members. The labeling step itself is not a
deterministic script: it needs an LLM to read the prompts and write a
label and gloss (here, fresh agents with no repo access, see
`AI_USAGE.md` and `DESIGN_NOTES.md` section 12). The labeled
`labeling_output.json` for each snapshot is checked into `outputs/`; the
first, superseded label set is in `outputs/labels_v1/`. To import a
labeller's replies (a folder of `labels_<year>.json` arrays), check them,
and merge them into the hierarchy:

```
.venv\Scripts\python.exe scripts\t5_import_labels.py <folder>
.venv\Scripts\python.exe scripts\t5_apply_labels.py
```

### T6, evaluation

```
.venv\Scripts\python.exe scripts\t6_evaluate.py
```

Runs coherence-vs-null, both stability measurements, label faithfulness,
and the extrinsic drill-down eval, and writes a fresh `outputs/metrics.json`.
Takes 15-20 minutes on CPU, mostly the NLI faithfulness pass (4 snapshots)
and the retrieval candidate-pool embedding. Passing section names reruns
only those and updates them in place, e.g.
`scripts\t6_evaluate.py faithfulness`.

Four more steps add to `metrics.json` and must run after it, in this order
(a full `t6_evaluate.py` run starts `metrics.json` from scratch):

```
# extrinsic: leave-one-out drill-down vs flat with a paired CI, routing vs
# chance, and the branching/beta sweeps (DESIGN_NOTES.md section 14)
.venv\Scripts\python.exe scripts\t6_patch_extrinsic.py

# degree/arity-preserving hypergraph shuffle null for coherence
# (DESIGN_NOTES.md section 13), 4 to 10 minutes
.venv\Scripts\python.exe scripts\hypergraph_shuffle_null.py

# structural held-out coherence across alpha (section 13), 4 to 16 minutes;
# figure 6 needs this
.venv\Scripts\python.exe scripts\structural_holdout.py

# whether change between snapshots is localised (section 10), about 3 minutes
.venv\Scripts\python.exe scripts\localisation.py
```

The other scripts in `scripts\` (`alpha_sweep`, `level0_skew_check`,
`temporal_threshold_sweep`, `warm_start_sweep`, `rerank_sweep`,
`coarsening_compare`, `label_routing`, `pair_overlap`) are the
follow-up experiments described in `DESIGN_NOTES.md` section 15. Each
writes its own JSON in `outputs\` and none of them change the shipped
pipeline.

### Figures

```
.venv\Scripts\python.exe scripts\make_report_figures.py
```

Regenerates the PNGs in `outputs/figures/` directly from
`outputs/metrics.json`, so a figure can't drift from the number it's
illustrating.

## Repo layout

```
data/                    shipped TKH export, questions, ground truth
src/tkh/                 the actual method: io, hypergraph, embeddings,
                          cluster, collapse, temporal, pipeline, labeling
src/tkh/eval/            T6: coherence, stability, faithfulness, extrinsic
scripts/                 runnable entry points, one per pipeline stage
tests/                   unit tests (T4 collapse, T3 matching, faithfulness
                          held-out split, multilevel laminarity,
                          leave-one-out and paired statistics, held-out
                          cohesion, sparse UPGMA against scipy)
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

alpha = 0.3 was chosen by ablation. The temporal-matching threshold
sensitivity, the level-0 size-skew hypothesis and a warm-started variant
have all been checked (`DESIGN_NOTES.md` sections 10 and 15). Known open
gaps, each discussed in `DESIGN_NOTES.md`:

- The structural term is a weighted clique expansion, and in the shipped
  pipeline the T4 collapsed hypergraph is written out but not used to
  build coarser levels. A variant that uses it
  (`coarsening="multilevel"` in `pipeline.py`) was tested and lost on
  coherence (sections 3, 4, 9, 15; `scripts\coarsening_compare.py`).
- Labelling is not a rerunnable script, since it needs an LLM (section 12).
- On recall@20, leave-one-out drill-down shows no detectable difference
  from flat at 14 questions. Routing with labels beats chance at every
  budget, routing on centroids doesn't (section 14).
- Change between snapshots is not localised to where new hyperedges
  landed. Churn partly follows new semantic neighbours (exploratory), and
  much of it is global (section 10, `scripts\localisation.py`).
- The structure term predicts held-out hyperedges from papers it has
  seen, but generalises weakly to unseen papers; at alpha=0.3 it only
  clearly earns its place at level 0 (section 13).
