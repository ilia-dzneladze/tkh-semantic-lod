# Multi-resolution semantic abstraction over the TKH hypergraph

Intern-level assessment submission. The write-up is `report.pdf` (source
`report.md`), design rationale is in `DESIGN_NOTES.md`, and AI tool usage
is in `AI_USAGE.md`.

## Setup

`scripts/reproduce_all.py` does all of this for you (next section). To
set up by hand instead:

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

One command, from a fresh clone, with any Python 3.11:

```
py -3.11 scripts/reproduce_all.py        # Windows
python3.11 scripts/reproduce_all.py      # macOS / Linux
```

It creates `.venv`, installs the pinned requirements (the step above,
about 15 minutes the first time), re-runs itself inside the venv, and then
runs the whole pipeline in order, stopping at the first failure: T1
statistics, the hierarchies and temporal events, the labels, validation
and unit tests, every evaluation, the blind-rating scores and the
figures. Everything lands in `outputs/`, figures in `outputs/figures/`.
The pipeline itself takes about 30 minutes on a laptop CPU. Later runs
reuse the venv and skip installing. `--dry-run` prints the steps without
running anything, and `--no-setup` uses whatever Python you run it with.

The same steps one by one, if you'd rather:

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

### Labels and blind ratings: the sample sets, or your own

Two steps need an LLM or a person, so they can't be rerun bit for bit:
the labels and glosses (T5) and the blind ratings (T6). The ones I used
are checked in and are what a plain run uses:

- **sample label set**: `outputs/snapshots/<year>/labeling_output.json`,
  248 labels written by fresh agents with no repo access from the prompts
  in `labeling_input.json` (the first, superseded set is in
  `outputs/labels_v1/`);
- **sample rating set**: `outputs/blind_eval/`, the intruder and gloss
  packets, their answer keys, the exact rater prompts, and the ratings
  (`*_ratings.json` records the rater model and its tool calls).

To make your own, with any LLM or by hand, and with your own prompt if
you like:

```
# 1. prompts for a new label set; --template is optional
python scripts/t5_dump_labeling_input.py --out my_labels --template my_prompt.txt
#    give each my_labels/<year>/labeller_request.md to your labeller and
#    save each reply as replies/labels_<year>.json (fenced JSON is fine)
python scripts/t5_import_labels.py replies --set my_labels
python scripts/t5_apply_labels.py --labels my_labels

# 2. blind-rating packets for those labels; instruction files are optional
python scripts/blind_eval.py make --out my_blind --intruder-instructions intr.txt --gloss-instructions gloss.txt
#    give my_blind/intruder_rater_prompt.md and gloss_rater_prompt.md to a
#    rater who has NOT seen the labels, keys or repo; save the two replies
python scripts/blind_eval.py import my_blind --intruder intruder_reply.txt --gloss gloss_reply.txt --rater "who rated"

# 3. the whole run on your sets
python scripts/reproduce_all.py --labels my_labels --blind my_blind
```

A prompt template is a text file with placeholders from `{year}`,
`{sample_n}`, `{total_n}`, `{type_line}` and `{member_lines}`; it must
use `{member_lines}`, and literal braces must be doubled. Rater
instruction files must contain `{n_items}`. The defaults are in
`src/tkh/labeling.py` and `scripts/blind_eval.py`. The imports check
every reply (every item answered, word limits, valid answers) and write
nothing if anything is wrong, and scoring refuses a rating set made for
different labels than the ones applied. `--replay` doesn't combine with
your own sets, because the recorded NLI outputs only cover my glosses. See
`DESIGN_NOTES.md` section 24.

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
