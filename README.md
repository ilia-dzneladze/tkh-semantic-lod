# Multi-resolution semantic abstraction over the TKH hypergraph

Intern-level assessment submission. The write-up is `report.pdf` (source
`report.md`), design rationale is in `DESIGN_NOTES.md`, and AI tool usage
is in `AI_USAGE.md`.

## Setup

Needs Python 3.11. In the commands below, `python` means the venv's
interpreter: `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on
macOS and Linux (or activate the venv). Forward slashes in script paths
work on all three.

Windows or macOS:

```
python3.11 -m venv .venv            # Windows without python on PATH: py -3.11 -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux: install the CPU build of torch first, then the rest. On Linux the
PyPI wheel for `torch==2.14.0` pulls in the CUDA 13 toolkit and several
nvidia packages that `requirements.txt` doesn't pin; the `+cpu` wheel
satisfies the same pin without them. I checked the package metadata and
the index, but I haven't run the Linux install end to end.

```
python3.11 -m venv .venv
python -m pip install --upgrade pip
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`requirements.txt` is the full pinned set, resolved from the loose
`requirements.in`. No GPU is needed. Installing takes about 15 minutes,
mostly torch. The first run downloads two models from Hugging Face, each
pinned to a fixed commit in the code: `sentence-transformers/all-mpnet-base-v2`
(about 420MB, drives clustering and retrieval) and
`cross-encoder/nli-deberta-v3-base` (about 400MB, the NLI faithfulness
judge).

## Reproducing everything

Run from the repo root, in this order. Times are for a laptop CPU.

```
python scripts/t1_describe.py            # T1 snapshot statistics, seconds
python scripts/run_pipeline.py           # T2-T4: hierarchies + temporal events, 1.5-5 min
python scripts/t5_apply_labels.py        # REQUIRED: run_pipeline writes labels as null
python scripts/validate_hierarchy.py     # laminarity and id checks
python -m pytest tests -q                # unit tests

python scripts/t6_evaluate.py            # T6: fresh metrics.json, 15-20 min
python scripts/t6_patch_extrinsic.py     # leave-one-out, routing, sweeps, ~5 min
python scripts/hypergraph_shuffle_null.py  # degree/arity-preserving null, 4-10 min
python scripts/structural_holdout.py     # held-out hyperedges across alpha, 4-16 min
python scripts/localisation.py           # is change localised, ~3 min
python scripts/blind_eval.py score       # blind intruder + gloss ratings into metrics.json
python scripts/make_report_figures.py    # figures from metrics.json
```

`t5_apply_labels.py` puts the checked-in labels back onto the freshly
built hierarchy. It applies a label only if the prompt rebuilt from the
cluster's current members is identical to the prompt the label was
written from, and it stops with an error otherwise. So if you change the
clustering (alpha, thresholds), the old labels won't silently land on new
clusters (`DESIGN_NOTES.md` section 19). `t6_evaluate.py` refuses to run
its label-dependent sections on an unlabelled hierarchy.

A full `t6_evaluate.py` run starts `metrics.json` from scratch, and every
step after it adds its own section, which is why the order matters.
Passing section names reruns only those and updates them in place, e.g.
`python scripts/t6_evaluate.py faithfulness`.

`python scripts/reproduce_all.py` runs all of the above in order and
stops at the first failure.

### Checking your results against mine

Nothing here is trained. The method uses two frozen models pinned to
exact commits, and everything after them is deterministic. So the only
place a rerun on another machine can differ is in the models' outputs:
the same text can embed differently in the last float bits depending on
hardware, libraries and even batch composition (`DESIGN_NOTES.md`
section 23). Every model output from my run, plus a SHA-256 checksum of
every result, is published as a Hugging Face dataset (see
`release/README.md` once downloaded). Put it in `release/`, then:

```
# do my results follow from my model outputs? (replay, no model download)
python scripts/reproduce_all.py --replay release/model_outputs
python scripts/verify_release.py checksums

# how far are your model outputs from mine?
python scripts/verify_release.py compare-embeddings
```

If the replay matches and your own run doesn't, the difference is in the
model outputs, not the code, and `compare-embeddings` shows whether it's
big enough to change any nearest neighbours.

The release is built by recording a full run and exporting it:
`python scripts/reproduce_all.py --record <dir>`, then
`python scripts/export_release.py <dir>`. The export refuses to write
anything unless the recorded merge trees cut back into the shipped
hierarchies exactly.

### What isn't a deterministic script

Two steps used an LLM and can't be rerun byte-for-byte, so their outputs
are checked in:

- **Labelling (T5).** `t5_dump_labeling_input.py` writes the exact prompt
  for every level-0/1 super-node (`outputs/snapshots/<year>/labeling_input.json`).
  Fresh agents with no repo access wrote the labels
  (`labeling_output.json`). `t5_import_labels.py <folder>` validates and
  imports a new set. The first, superseded set is in `outputs/labels_v1/`.
- **Blind ratings (T6).** `blind_eval.py make` builds the intruder-test and
  gloss-rating packets, answer keys and the exact rater prompts
  (`outputs/blind_eval/`). The ratings were made by fresh agents that saw
  only their own prompt file (`*_ratings.json` records the model and the
  tool calls). `blind_eval.py score` is deterministic given the ratings.

### Follow-up experiments

The other scripts in `scripts/` are the follow-up experiments in
`DESIGN_NOTES.md` section 15: `alpha_sweep`, `level0_skew_check`,
`temporal_threshold_sweep`, `warm_start_sweep`, `rerank_sweep`,
`coarsening_compare`, `label_routing`, `pair_overlap`,
`affinity_mass_share`, `signal_overlap`. Each writes its own JSON in
`outputs/` and none of them changes the shipped pipeline.
`rerank_sweep.json` and `coarsening_compare.json` predate the
ground-truth matcher fix (section 16) and their ground-truth-dependent
numbers were not regenerated. No conclusion rests on those numbers.
`scripts/t2_build_hierarchy.py` is an early smoke test, superseded by
`run_pipeline.py`.

## Repo layout

```
data/                 shipped TKH export, questions, ground truth
src/tkh/              the method: io, hypergraph, embeddings, cluster,
                      collapse, temporal, pipeline, labeling
src/tkh/eval/         T6: coherence, stability, faithfulness, extrinsic,
                      blind judgements, shared statistics
scripts/              runnable entry points
tests/                unit tests (collapse, temporal matching, faithfulness
                      held-out split, label staleness guard, multilevel
                      laminarity, UPGMA against scipy, held-out cohesion,
                      stability CIs, blind packets and scoring,
                      ground-truth matching, leave-one-out statistics)
outputs/              hierarchy.json + labelling per snapshot,
                      temporal_events.json, metrics.json, blind_eval/,
                      experiment JSONs, figures/
report.pdf            the write-up
DESIGN_NOTES.md       why each non-obvious choice was made
AI_USAGE.md           AI tool usage disclosure
requirements.in/.txt  loose and pinned dependency lists
```

## Status

What holds up: laminar hierarchies at all four snapshots; levels 1 and 2
coherent to a blind reader (intruder test), level 0 not shown to be; 0 of
48 glosses rated wrong by a blind rater; label routing beats chance at
every budget. Known gaps, each discussed in `DESIGN_NOTES.md`:

- The structural term is a weighted clique expansion, and at alpha=0.3 it
  carries about 7% of the affinity mass, not 30% (sections 3, 17).
- The T4 collapse is applied at every level and written out, but the
  shipped levels are cuts of one dendrogram, so it doesn't build them. The
  variant where it does lost on coherence (sections 9, 15).
- The perturbation stability test only removes hyperedges, so it rewards
  ignoring structure, and alpha was partly chosen on it (section 17).
- Change between snapshots is not localised to where new hyperedges
  landed (section 10). The matching threshold is sensitive.
- Snapshot membership follows the hyperedge year, so each early snapshot
  holds a handful of nodes the corpus hadn't seen yet: temporal honesty of
  the labeller's input is about 98%, not guaranteed (section 2).
- On recall@20, drill-down shows no detectable difference from flat at 12
  questions (sections 14, 16).
- All blind judgements come from one LLM rater, with no second rater and
  no human pass (sections 21, 22).
