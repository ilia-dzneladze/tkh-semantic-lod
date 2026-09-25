# AI usage

**Tool.** Claude Code, and no other AI tool. It ran on Sonnet 5 for the
build and the first follow-up experiments, on Opus 5 for the first review,
the correctness fixes and the next experiments, and on Opus 5.5 for the
last passes: closing the review, publishing the model outputs, the
one-command reproduction, the documentation cleanup, the LaTeX report,
the visualisation and the simplification pass. The labels were
written by Claude Code sub-agents on Sonnet 5 and the blind ratings by
sub-agents on Opus 5.5, as described below.

**Division of labour.** I made the design calls and decided what to
accept. The agent wrote, ran and debugged nearly all of the code, ran the
checks, and drafted the documentation, including this file, to my
direction. Each entry gives the prompt that started a phase, what the
agent did, what I accepted or changed, and what was checked before I
trusted it. Entries are in the order they happened, so a later one
sometimes corrects an earlier one, and where that happens it's noted.

Prompts are quoted as I typed them, with spelling and grammar corrected
and long ones abridged where marked. Where a prompt was short and leaned
on the conversation so far, what it referred to follows it in brackets.

**Summary.** One row per phase, in order. The sections below have the
prompts and the detail.

| Phase | What the agent did | What I accepted or changed | How it was checked |
|---|---|---|---|
| Planning, feasibility | Proposed Python, dependencies and four design decisions | Chose the joint affinity graph, MPNet, NLI; moved labelling from an API to agents | I worked through the 1/(k-1) weighting until I could explain it |
| T1-T5 build | Wrote the modules and the pipeline | Caught an id self-reference bug and wrong `member_ids` by reading `hierarchy.json` | Validator script written for it |
| Code review, T5, T6, T7 | Six code fixes, first labels by sub-agents, first metrics, first report | Rejected two first results (NLI all "neutral", zero recall) rather than report them | Both traced to causes; numbers unchanged by the cosmetic fixes |
| Alpha sweep, reranking | 19-value alpha sweep; rerank sweep | Took alpha=0.3; the rerank gain later turned out to be one question | Rules written first; smoke test reproduced shipped numbers |
| Review on Opus 5 | Found six places where code and docs disagreed | Had all fixed | 16 tests, validator |
| Multilevel T4 | Built T4-driven coarser levels, two variants | Both failed the rule written first, so they didn't ship | Refactor reproduced shipped hierarchies exactly |
| Relabelling, leave-one-out, held-out edges | Fresh labellers without the questions; LOO extrinsic; structural holdout | Accepted the new labels; kept alpha=0.3 | Read the sub-agents' tool logs; rules written first |
| Literature, localisation | Read 8 PDFs and cited 5; tested whether change is localised | It isn't, and the report says so | Two overclaims in its own drafts fixed |
| Clean reproduction, skeptical review | UPGMA-vs-scipy test; clean runs; found the element-matcher bug, temporal leak and the 7% structural share | Acted on five findings; disagreed that TF-IDF coherence is worthless | 36 tests; clean `git archive` rerun |
| Items 6-11 | Blind intruder and gloss raters, CIs, NLI check, 5-page report | Rejected its first bootstrap CI (interval above its own estimate) | 49 tests; citations checked by web search |
| Release, one-command run | Record and replay of model outputs, Hugging Face release, `reproduce_all.py`, custom label sets | I uploaded the dataset | 45 outputs byte for byte from a bare folder in 72 min |
| Cleanups, figure, simplification | Rewrote docs, split `scripts/`, drew the levels 0-2 figure, removed duplication | Chose how much to change before upload | 64 tests; replay byte-identical |
| Final touch-ups | Graded the repo against the brief; temporal honesty audit; level-0 table in the report; this summary | Took three of its five suggestions | Rule written first; report rebuilt at 5 pages |

## Planning

Prompt: "Read the .md file in the root folder that outlines the task and
explain what the task is about... Then scaffold a plan: what language to
use, what dependencies, and what would be the best way to implement the
solution... If you need my approval for design choices, ask, and I will
weigh the trade-offs."

I set it up this way so the design decisions came back to me rather than
being settled inside the code. The agent read the brief and the data,
proposed Python and a dependency list, and surfaced four decisions. I
chose a joint weighted affinity graph (`alpha * structural + (1-alpha) *
semantic`), which makes the balance between structure and meaning one
parameter that can be swept (DESIGN_NOTES section 7), general-purpose
MPNet over a domain-specific embedding, an LLM API for labelling, and an
NLI model for the faithfulness check. I accepted the language and
dependencies as proposed. Once I confirmed that API calls
would be billed outside my subscription, I switched labelling to agents
reading dumped cluster prompts (DESIGN_NOTES section 12).

I set alpha to 0.5 as a deliberate placeholder and flagged it for an
ablation once the evaluation existed, since the right balance depends on
how noisy the structure turns out to be. That ablation came later ("Alpha
ablation and relabelling").

## Feasibility check

Prompt: "Look over the plan and see if it's coherent and really doable
given my skills as a third-year CS student, and continue if you think it
will land me the job."

The agent's assessment was that most components assemble known library
pieces, but the hypergraph affinity construction is graduate-level and I'd
need to understand it rather than accept the formula. I worked through it
with the agent until I could explain why the 1/(arity-1) weighting gives
each hyperedge a total weight linear in its arity rather than quadratic.
On its own initiative it also simplified the T6 null model to random
labels instead of a degree-preserving hypergraph shuffle, which I accepted
as a trade-off the brief allows. The shuffle null was built later.

## T1-T5 implementation

No single prompt per task. This was continuous direction: I reviewed each
module's output as it was produced and asked follow-up questions or
corrections in place.

Accepted as proposed, once I understood the reasoning: the structural
weighting (first described as Zhou et al.'s method, corrected in the final
review to a weighted clique expansion), the sparse UPGMA, the T3 Jaccard
matching and the T4 collapse rule.

What I caught myself, by auditing the generated `hierarchy.json` by hand
instead of trusting a clean run: a self-reference bug in the persistent-id
scheme, and `member_ids` not matching the brief's own definition of a
laminar partition. Both were fixed and re-checked with a validator script
written for exactly this.

## Code review pass

Prompt: "Go over what's been written so far: what the scripts do with the
data and what we are getting ready for. And look at the logic: is it
optimal and coherent?"

The agent re-read all eight source modules and reported six issues: dead
code, an inconsistent stats dictionary, a docstring calling a union k-NN
graph "mutual", an undocumented correctness assumption in the merge step,
and two leftovers from the `member_ids` fix. I had it fix all six and
rerun the pipeline to confirm the numbers didn't change, since fixes to
dead or cosmetic code shouldn't change output. The assumption (a missing
edge counts as similarity zero) is the one I worked through the proof of
with the agent rather than take on faith, because the laminar guarantee
depends on it.

## T5 labelling

Prompt: "Continue with T5 labelling." [A label and a one-sentence gloss
per super-node at levels 0 and 1, as the brief asks.]

The agent dumped cluster member lists to files and labelled them with four
parallel sub-agents, one per snapshot, from a fixed prompt template
written to file so the process stayed auditable. I spot-checked word
limits (9 entries over the 6-word and 25-word limits were fixed) and read
the low-confidence clusters each sub-agent flagged. This set was later
replaced, twice ("Alpha ablation and relabelling", "Relabelling without
access to the questions").

## T6 evaluation

Prompt: "Continue with T6 evaluation." [Coherence, stability,
faithfulness and the extrinsic task.]

I rejected two first results rather than report them. The NLI check
returned "neutral" on nearly every pair, including obvious mismatches.
That traced to bare short phrases giving the model too little context, and
was fixed by building sentence-style premises from the members. (The
premise also turned out to use a subset of the labeller's own input, which
the final review caught.) The retrieval eval returned zero recall on every
question. I didn't accept "the corpus is just this hard" without checking,
and it was a real embedding problem, bare acronyms against sentence-long
questions, fixed by enriching short names with their hyperedge
neighbourhood.

For the drill-down branching factor I started at (3, 3), narrow on
purpose. It came back at about half of flat's recall, which I read as a
too-narrow branch excluding answers early rather than a limit of the
hierarchy, so I had the agent sweep wider values. (5, 5) matched flat at
about 16% of the candidates. That didn't survive later changes.

## T7 write-up

Prompt: "Continue with the T7 write-up."

Figures are generated from `metrics.json`, not hand-copied numbers. I
reviewed `report.md` against the metrics before treating it as final,
including the T4 collapse numbers against the current `hierarchy.json`.

## Alpha ablation and relabelling

Prompt: "Okay, generate a plan for each fixable iteration, starting with
the alpha ablation, and keep it in a separate ignored file called
FIXES.md. If a fix brings an improvement, keep it and commit. Generate the
plan and start the alpha sweep: 5-95, 10-90, 15-85, ..., 95-5."

The agent wrote `scripts/experiments/alpha_sweep.py` (19 values, scored on
coherence and stability only, the metrics that don't need labels). Before
trusting a 19-value run it smoke-tested alpha=0.5 alone, which reproduced
the shipped numbers to the digit. The decision rule was written before the
sweep in the private working file and is copied into `DESIGN_NOTES.md`
section 15. Several values passed it. After seeing the results I added a
stricter bar, beating 0.5 on each of the nine per-level numbers, and only
0.3 passed. That bar came after the results, so it's a robustness check,
not the rule. The agent applied 0.3 to the real pipeline, not just the
sweep's lightweight version, and reran the validator and tests.

Prompt: "Okay, add this to report.md as a finding, redo the labelling for
0.3, see whether the end result is really better with alpha at 0.3, and
give a logical justification for it in report.md."

The clusters changed, so the agent relabelled all 248 from scratch the
same way. I checked all four outputs with my own word-count script rather
than the sub-agents' self-reports (0 violations) and read the clusters
each flagged as hard calls. Rerunning T6 showed that (5, 5), right under
alpha=0.5, no longer matched flat (0.027 against 0.0325 recall). The agent
re-swept and (8, 8) matched again.

## Beating flat, not just matching it

Prompt: "Okay, now on to the next likely fix that will improve retrieval
compared to the baseline. It's really interesting that neither of the last
two could beat it, only match it. Try a new fix, test it, and conclude."

Before any code, I had the agent explain why no branching value could beat
flat, rather than try more values: drill-down ranks a subset of flat's
pool with flat's own scoring function, so parity is the ceiling. It wrote
`scripts/experiments/rerank_sweep.py` to blend each candidate's own score
with its level-1 ancestor's label+gloss score, with the rule (beat flat
outright at no more candidates than the (8, 8) pool) written down first.
beta=0.6 passed, recall 0.0423 against 0.0325, and the gain held against
the full unrestricted pool too. I picked 0.6 over 1.0, which had slightly
higher recall but worse precision and discards the node's own signal, and
over a dip at 0.9 that looked like noise over 14 questions. The agent
wired `beta` into `hierarchy_drilldown` and regenerated `metrics.json`
through the real code path. The final review showed this whole gain came
from one question after tuning on the test set.

## Final review and correctness fixes

Tool: Claude Code on Opus 5.

Prompt: "Look at the current version of my project. Do you think it is
sufficient to land me the job? ... I want your honest opinion, and if you
think we can iterate on some things and make them better, tell me
specifically what and how. Be thorough."

The agent read the brief, all source, the docs and `metrics.json`, and
checked claims against the code instead of the docs. Where they disagreed:

- The NLI premise was the first 15 sorted members, a strict subset of the
  25 the labeller saw, while the docstring and report said the full list.
- Temporal Jaccard used the full union including nodes new at t+1, while
  the docs said shared nodes. The agent measured the effect first: at
  level 1, 2022 to 2024, 14 of 50 clusters fell below MATCH_THRESHOLD,
  against 3 of 50 on shared nodes.
- The structural affinity was described as hypergraph-native, following
  Zhou et al. In the code it's a weighted clique expansion with no
  spectral step.
- The T4 collapsed hypergraph is computed but never used by the method.
- The "beats flat" result comes from one question (Q14), after tuning on
  the same questions.
- This file said every module had unit tests, and misstated the alpha
  rule.

It also measured that the (8, 8) drill-down pool held 80% of ground-truth
nodes at 25% of candidates. That later turned out to depend on the labels.

Prompt: "Yes, start with the P0 fixes." [The review's highest-priority
findings, mostly places where the code and the documents disagreed.]

What the agent changed, all accepted:

- `temporal.py`: Jaccard on shared nodes only, split events record
  persistent ids, three new tests. Every snapshot's member sets were
  checked identical before and after, so only matching changed. Reassigned
  persistent ids were remapped in the label files by exact member set, not
  relabelled. The threshold sweep was rerun, and MATCH_THRESHOLD is still
  sensitive.
- `labeling.py`, `eval/faithfulness.py`: one function defines the
  labeller's input, the NLI premise is a seeded sample of held-out members
  only, clusters with fewer than 5 are skipped and counted, and both
  contradiction and not-entailed rates are reported. Three new tests check
  the held-out set is disjoint from the labeller's input. Contradiction
  rose from 5-11% to 14-21%.
- `t6_evaluate.py` can rerun single sections, and the shuffle-null script
  writes into `metrics.json`. It reproduced its earlier values exactly.
- Docstrings, `DESIGN_NOTES.md`, the README and this file corrected to
  match the code, and pointers to my private working file replaced with
  DESIGN_NOTES pointers.

Verification: 16 tests pass, the laminarity validator passes on all four
snapshots, all 248 labels re-applied. The type bias in the labeller's
sample was found and documented here but only fixed with the relabel.

## Multilevel coarsening (T4 driving the coarser levels)

Tool: Claude Code on Opus 5.

Prompt: "Now start on P1: make T4 drive the coarser levels."

The agent wrote the keep-or-discard rule into DESIGN_NOTES section 15
before running anything, then implemented `coarsen_one_level` in
`pipeline.py` and `coarse_structural_affinity` in `collapse.py`, and
refactored clustering into one `build_levels` function shared by the
pipeline and the perturbation check.
`scripts/experiments/coarsening_compare.py` first checks that the
refactored dendrogram path reproduces the shipped hierarchies exactly,
which it does.

The first variant failed the rule. The agent proposed one principled fix
(size-normalised coarse affinity with count-weighted average linkage),
wrote a second rule before running it, and noted it was chosen after
seeing a failure. That failed too, so the pipeline stays on the dendrogram
and the multilevel code stays in as a non-default option. I held to
stopping there rather than try variants until one passed, which would
have fitted the method to the rule instead of testing it.

The same script found that routing on member centroids instead of labels
gives no lift over chance, so the 80%-at-25% routing result depends on
the labels. The agent corrected DESIGN_NOTES section 14 and the report.

Verification: 21 tests, including laminarity and the size budget for both
multilevel variants on a toy hypergraph and a check of the weighted UPGMA
arithmetic. The shipped outputs didn't change.

## Relabelling without access to the questions

Tool: Claude Code on Opus 5 for setup and evaluation; four fresh Claude
Code sub-agents on Sonnet 5, the same model as the original labels, for
the labelling.

Prompt: "Start with step 1, the relabelling." [Step 1 of the next three:
relabel without access to the questions, re-evaluate the extrinsic task
by leave-one-out, and test structure on held-out hyperedges.]

The main agent had read the question files earlier in the session, so it
didn't write labels itself and didn't use forked agents, which would
inherit that context. It asked me how to run the labelling, and I chose
fresh sub-agents, so the question files the main agent had read were never
in the labellers' context. Before any new labels existed it wrote the
decision rule into DESIGN_NOTES section 15, archived the first labels in
`outputs/labels_v1/`, and measured their routing as a baseline.

Code changes: `labeller_sample_ids` draws a seeded random sample
(`sampling="first"` reproduces the old one), the prompt includes the
node-type counts it had always claimed to, faithfulness takes the sampling
mode so its held-out set matches the labels being checked,
`routing_pool_recall` is shared by `label_routing.py` and
`coarsening_compare.py`, and `t5_import_labels.py` validates and merges
replies. Two new tests.

Each sub-agent got a prompt with only the path to its batch file, in a
scratch folder holding nothing else, and an output path, and was told to
use no other files or tools. No tool was technically blocked, so the main
agent read the sub-agents' tool logs afterwards: each made two reads of
its own batch file and one write, and no log mentions the question files.
All 248 labels passed the id and word-limit checks with no edits.

Result under the rule: label routing still clearly beat chance (lift
0.42, CI 0.23 to 0.57, against 0.55 for the old labels), contradiction
improved to 5-14% from 14-21%, and recall@20 tied flat instead of beating
it. Settings weren't retuned. I accepted the new labels as the shipped
set. (Routing numbers were later redone after the ground-truth fix; the
verdict held.)

## Extrinsic re-evaluation with leave-one-out

Tool: Claude Code on Opus 5.

Prompt: "Continue with step 2." [Leave-one-out re-evaluation of the
extrinsic task.]

The agent wrote the rule into DESIGN_NOTES section 15 first, then added
`leave_one_out_select` and `paired_comparison` to `eval/extrinsic.py`,
with four tests, one checking that a held-out question's own score can't
influence the setting chosen for it. It extended
`scripts/pipeline/t6_extrinsic.py` to write the leave-one-out result, the
in-sample comparison and the routing curves into `metrics.json`, and added
a routing figure, checking its two colours with the dataviz skill's
palette validator and looking at the rendered image before using it.

Result under the rule: leave-one-out drill-down against flat was -1.5
points of recall@20 (CI -6.3 to +2.9, p = 0.59), so no detectable
difference, and label routing beat chance at every budget. Nothing in the
hierarchy or labels changed. (Redone after the ground-truth fix: -8.7
points, CI -25.9 to +0.7, same verdict.)

## Structural held-out coherence

Tool: Claude Code on Opus 5.

Prompt: "Continue with step 3." [Structural coherence on held-out
hyperedges.]

Rule first, into DESIGN_NOTES section 15. The agent found that every edge
has a `provenance.article_id` and added whole-paper holdout as a stricter
second scheme next to random-edge holdout, since edges from one paper are
correlated. It added `heldout_edge_cohesion` to `eval/coherence.py` with
two tests, wrote `scripts/pipeline/structural_holdout.py` (7 alphas, 2
schemes, 5 seeds, TF-IDF coherence on the same clusterings) and a
trade-off figure. After rendering the figure it replaced an alpha label
that sat next to the wrong series with a ring on the shipped alpha. It
also found alpha=1.0 is degenerate (about 1,100 forced merges, one
cluster) and left it out of the figure with a note instead of reading it
as "structure carries nothing".

Result under the rule: at alpha=0.3 structure earns its place only at
level 0, most of its predictive power doesn't survive holding out whole
papers, and no alpha dominates 0.3. I kept alpha at 0.3.

## Newer literature

Tool: Claude Code on Opus 5.

Prompt: "I've put new literature under literature/new/. See if anything
there could be useful for our case and cite the most important papers;
maybe something can help more than the current sources."

I collected the eight PDFs myself, using a search prompt the agent had
written for another LLM. The agent extracted their text with PyMuPDF and
read the abstracts, methods and conclusions. It picked five to cite, each
tied to one design choice: Ruggeri et al. (pair overlap and
detectability), Ma et al. AdE (feature-aware clique expansion), Asgari et
al. (No-Smoothing against smoothed dynamic community detection), SHyPar
(local against spectral coarsening) and DeWolfe and Theberge (edge
clustering, overlapping communities). It left out Gong et al., Kirkley and
FeClustRE as less relevant. Since Ruggeri et al. make a testable claim, it
wrote `scripts/experiments/pair_overlap.py` to count how often concept
pairs recur across hyperedges and papers (1.7% across papers in 2026) and
used that to explain the weak paper-level held-out result.

Two of its drafts claimed more than the data showed, and it fixed both
before I saw them. One said held-out pairs were "covered by the same
paper's other edges", but the exact pair almost never repeats, so it
became connection through the paper's other edges. The other used
"transformer" as an example concept, which isn't in the corpus, so it
became MAE after checking. Papers are cited by arXiv id. The agent didn't
verify published venues at this point.

## Localisation of change (P5)

Tool: Claude Code on Opus 5.

Prompt: "Yes, start with the localisation test." [P5: is change between
snapshots confined to where the corpus changed?]

Rule first: per-cluster churn against the share of new hyperedges around
the cluster, Spearman with a bootstrap CI, cluster size partialled out.
Then `scripts/pipeline/localisation.py`, which reads the shipped
hierarchies and doesn't depend on the section 10 event thresholds. The
test failed at every level. Before writing that up, the agent broke the
result down by transition and checked the exposure ranges to rule out a
bug or a pooling artefact. It then added one exploratory measure after
seeing the failure, the share of new nodes among an old node's k-NN
neighbours, which does correlate with churn at levels 1 and 2. The docs
label it post hoc, and the pre-registered verdict stands. The explanation
that fixed cluster counts force global reshuffling is the agent's
hypothesis and is marked untested.

## UPGMA check, clean reproduction and stale-number pass

Tool: Claude Code on Opus 5.

Prompt: "Continue with the original list of tasks, so we finish those first
and then we can experiment."

The agent added `tests/test_upgma_vs_scipy.py`, comparing `sparse_upgma`
with scipy's dense average linkage on random graphs (merge heights, cuts,
forced merges, point weights), and checked that the comparison fails
against weighted average linkage, so it isn't passing trivially.

For the reproduction it copied exactly the files git would ship into a
scratch folder, built a fresh venv from `requirements.txt` and ran every
README step in order. The hierarchies, temporal events and shuffle null
came out byte-identical, and every aggregate in `metrics.json` matched.
Three problems turned up, and the agent fixed each:

- The README didn't say a full `t6_evaluate.py` run rewrites
  `metrics.json` from scratch, so following it silently dropped a figure.
  The later steps are now listed as required.
- The committed `metrics.json` held per-cluster coherence under persistent
  ids from before the temporal-matching fix: 51 keys at 2026 level 2
  pointed at clusters that no longer exist. The agent replaced it with the
  reproduced file after confirming the values were the same under the new
  ids and every aggregate matched.
- The T1 quality-note sample depended on set iteration order.
  `build_snapshot` now iterates in sorted order.

It also replaced optimistic README timings with measured ones and checked
every number in `report.md` against `metrics.json`. It reported they
matched. That wasn't fully true: a later audit found two stale passages
("Closing the rest of the review").

## Skeptical review pass and the fixes from it

Tool: Claude Code on Opus 5, one session, used as a reviewer first and
then to implement the fixes I picked.

Prompt (abridged): "You are reviewing an internship assessment submission
in this repo, as a skeptical senior researcher on the hiring team would...
Your job is to find gaps: anything that would cost points or raise doubts
with a careful reviewer. Do not edit any files. You may run code to verify
a suspicion. Prefer checking over guessing... Separate two kinds of
finding: gaps the docs already admit and gaps nobody has mentioned."

The read-only constraint was deliberate. I wanted findings with evidence
before anything changed, so the agent reproduced the pipeline from a clean
`git archive` of HEAD in a scratch folder. The hierarchies and temporal
events came back byte-identical and the faithfulness numbers reproduced
exactly. It also found that `t6_evaluate.py` crashed with a KeyError if
you followed the README literally, because `run_pipeline.py` clears the
labels and the README didn't say to reapply them.

Findings I acted on, and what I had it do:

- The ground-truth matcher resolved expected method names to chemical
  elements ("N", "P", "S" and "Si" are substrings of "physics-informed").
  It quantified this first: 47 of 147 ground-truth node ids were surface
  forms of two characters or less. Fixed by requiring both sides of a
  substring match to be at least four characters, pinned by
  `tests/test_ground_truth_match.py`, and every extrinsic number rerun
  (DESIGN_NOTES section 16).
- Temporal honesty was claimed to hold by construction and doesn't. The
  agent traced it from a 2020 label naming EquiformerV2 back to
  `build_snapshot` and counted the affected nodes. I chose to document the
  count rather than filter and relabel (section 2).
- alpha isn't the mixing weight it reads as. It measured the structural
  share of the affinity mass, about 7% at alpha=0.3, and I had it write
  `scripts/experiments/affinity_mass_share.py` so the number is
  reproducible (section 17).
- The perturbation stability measure only removes hyperedges, so it
  rewards ignoring the hypergraph, and alpha was partly selected on it.
  The agent re-read the sweep without it. 0.3 still wins, which I hadn't
  assumed going in.
- The formal statement was prose with no objective in it. I had it
  rewritten with the actual definitions and the honest framing that
  average linkage optimises nothing globally.

What I didn't accept: the agent's stronger reading of the TF-IDF check as
circular. It's a weaker independence claim than I'd written, and I took
its measurement (69% of MPNet neighbour pairs share a token, against 7% of
random pairs) into the report as a named assumption, but I don't agree it
makes the check worthless, and the report says what I think.

Verification: 36 tests (4 new), the validator passes on all four
snapshots, the figures regenerate and now read their question count from
`metrics.json` instead of hardcoding 14, and the pre-registered rules were
re-read against the rerun numbers rather than rewritten (DESIGN_NOTES
section 18). Left open at this point: report length, README gaps, the
cross-snapshot CIs and the unvalidated NLI judge, all handled in the next
section.

## Closing the rest of the review: items 6 to 11

Tool: Claude Code on Opus 5.5, same session. The blind raters were two
further Claude Code subagents on Opus 5.5 (`claude-opus-5-5`).

Prompt: "Continue with tasks 6-11, end to end, and test the
implementation."

The items were TF-IDF independence (6), validating the faithfulness judge
(7), stale numbers (8), report format (9), reproducibility (10) and the
cross-snapshot CIs (11). Before starting, the agent asked me who should do
the blind ratings, pointing out it couldn't rate itself, having already
seen the labels, the controls and the questions. I chose fresh subagents
over rating by hand.

What it built, and what was checked:

- Reproducibility (10). Both models pinned to the commits these results
  came from. Labels are applied only if the prompt rebuilt from the
  cluster's current members matches the label's prompt byte for byte; all
  248 pass, and a test swaps two clusters' members under the same ids.
  `t6_evaluate.py` stops with an instruction instead of a KeyError. The
  README got one ordered path, platform-neutral commands and a Linux torch
  step. The Linux claim rests on the PyPI metadata for `torch==2.14.0` and
  the PyTorch CPU index listing `2.14.0+cpu`. Nobody ran a Linux install.
- Cross-snapshot CIs (11). The agent's first replacement, a node bootstrap
  with replacement, was wrong: the level-2 interval came out entirely
  above the estimate (0.651 to 0.682 around 0.647). It diagnosed why
  (duplicated nodes add same-cluster pairs, which drive ARI), switched to
  half-sampling without replacement, checked the half-sample mean matches
  the full ARI, and added a many-small-clusters test that the old
  estimator fails. I accepted it once every interval contained its
  estimate.
- Coherence independence (6). `scripts/experiments/signal_overlap.py`
  makes the TF-IDF/MPNet overlap reproducible, and the blind intruder test
  is in `eval/blind.py` and `scripts/pipeline/blind_eval.py`.
- Faithfulness (7). Wilson CIs on every rate, a blind gloss rating
  compared with NLI on identical items, and a check against source-paper
  titles.
- For each new check the agent wrote the rule into DESIGN_NOTES section 15
  before generating any data, and the results apply those rules as
  written, including the two that didn't go my way: level 0 fails the
  intruder test, and the provenance check is inconclusive.

The raters. Each was a fresh general-purpose subagent with no
conversation context, given a short instruction plus the path to a folder
holding only its own item file. The answer keys stayed in a separate
folder outside the repo until the ratings were in. The agent read both
transcripts: each subagent made one Read, of its own file, then answered.
Answers were saved from the transcripts, not retyped, and each
`*_ratings.json` records the model and the tool calls. The exact rater
prompts are in `outputs/blind_eval/*_rater_prompt.md`. The labels were
written by Sonnet, so an Opus rater limits self-preference but doesn't
remove it. It's one rater per task, with no second rater and no human
pass.

Report (9). The agent rewrote `report.md` from about 8,000 words to a
5-page version with a T1 section, the formal statement as formulas, the
new results and a reference list, rendered the PDF with a local helper
(`scripts/make_report_pdf.py`, gitignored) and measured the page count.
It checked three new citations by web search before using them (Clauset,
Moore and Newman 2008; Peixoto 2014; Chang et al. 2009), and the venues of
Greene et al. 2010, Loukas 2019 and Agarwal et al. 2006. One claim in its
own draft was wrong, that SHyPar doesn't coarsen hypergraphs. It does, and
the sentence was rewritten before the PDF was built.

Number audit (8). The agent checked every number in the new report
against the output JSONs. That found one real overclaim carried through
several versions: the warm start's coherence cost, written as "level 2
loses 17-41%", was the 2026 snapshot alone. Averaged over the snapshots a
warm start can change, it's 7% to 18%, and mixed at 2022. The verdict
still holds, and DESIGN_NOTES section 10 has the corrected numbers. It
also means the earlier claim that every report number matched
`metrics.json` wasn't true at the time.

Verification: 49 tests (13 new). The agent copied exactly the files git
would ship into an empty folder and ran every README step in order with
the existing venv, so the install itself wasn't retested. Every step
exited cleanly in about 30 minutes. T1 statistics, temporal events and all
four hierarchies came back byte-identical, and all eight sections of
`metrics.json` matched the committed file exactly.

## Publishing the model outputs for verification

Tool: Claude Code on Opus 5.5, same session.

Prompts: I asked whether the "weights generated at the end" could be
uploaded to Hugging Face so a reproducer could check my results, then
"Yes, go ahead, and then tell me the commands to upload the embeddings to
Hugging Face."

The agent first corrected my premise: nothing is trained, so there are no
weights, and what varies between machines is the output of the two frozen
models. It proposed publishing those outputs plus checksums, and I agreed.
Before building anything it measured how fragile the outputs are: 514 of
5,428 node embeddings differ in the last bits depending only on batching.
That decided the design. Outputs are recorded and replayed per whole model
call (`src/tkh/model_outputs.py`), not per text. It routed every embedding
and NLI call through that module, wrote `reproduce_all.py`,
`export_release.py` and `verify_release.py`, and 5 tests, two of which
check that replay never loads either model.

Verification was three clean-folder runs of the whole pipeline. Recording
gave outputs identical to the repo, so recording changes nothing. The
export re-derives the four merge trees and refuses to write unless they
cut into the shipped hierarchies exactly, which they do. The final replay,
from a fresh copy of the final code with no model calls, reproduced all 45
output files and all 37 release files to the byte in 8 minutes.
`compare-embeddings` on this machine found all 5,428 embeddings and all
four nearest-neighbour graphs identical to the published ones.

Along the way the agent found and fixed two real problems (DESIGN_NOTES
section 23): pinned models contacted the Hub on every load and stalled
when the network dropped, and `localisation.json` changed in the 16th
digit between runs because a mean was summed over a Python set in hash
order. Every earlier check had compared only `metrics.json`, so that had
slipped through two "clean reproductions". The checksum over every output
file caught it, and the file is now stable under two hash seeds.

It also made one mistake and caught it before relying on it. A sed edit
meant to point its comparison script at the new run failed silently, and
the first "identical" result compared the wrong folder. It noticed the
path hadn't changed, rewrote the script to take the folder as an argument,
and reran. It stopped one replay run partway because the localisation fix
had made it obsolete.

I did the upload myself. The export it's derived from shipped in this
public repo at the time (it was taken out later, see "Final touch-ups"),
so the dataset is public too
(`iliadzneladze/tkh-multires-outputs`), and the agent switched its
visibility at my request.

## One-command reproduction, and custom label and rating sets

Tool: Claude Code on Opus 5.5, same session.

Prompts: "Can we make a single command that reproduces the whole
experiment, one pass from nothing to the built hypergraph plus the
evaluation, with all the figures and everything?" and "Is there a way to
ship sample labels and blind ratings, the ones we built
non-deterministically with the fresh agents, and also give the reproducer
the freedom to make their own with a custom prompt, with CLI arguments that
point to those custom labels and blind ratings?"

What the agent built. `scripts/reproduce_all.py` starts from a bare Python
3.11: it creates `.venv`, installs the pinned requirements (CPU torch first
on Linux), hands over to the venv and runs everything through the
figures. It records what it installed, so later runs skip the install.
For label and rating sets, `t5_dump_labeling_input.py --out --template`
writes the prompts, a paste-ready request per snapshot and the template;
`t5_import_labels.py --set` and `blind_eval.py import` check replies, raw
or fenced JSON, and write nothing if any reply is wrong;
`t5_apply_labels.py --labels` applies a set with the template it was made
with; `blind_eval.py make` takes custom rater instructions; `score --dir`
refuses ratings made for different labels; and `reproduce_all.py
--labels --blind` runs everything on custom sets. 7 new tests. The
reasoning is in DESIGN_NOTES section 24, including why the labeller's
sample size stays fixed at 25.

What was checked:

- The shipped sets stay the default and byte-identical. The rebuilt rating
  packets matched the shipped ones file for file, with a fresh NLI run,
  and every output still matched the published checksums.
- The custom workflow ran end to end in a scratch copy with a stand-in
  labeller and raters: a custom template, one fenced reply, 248 labels
  applied, new packets, scoring. The refusals worked: the shipped ratings
  against custom labels, a bad template, a malformed reply.
- That test found a real bug. A failed rating import still rewrote the
  valid half and wiped the rater description. The agent made both imports
  all or nothing and retested.
- The one-command run, from a copy with no `.venv`, took 72 minutes, about
  17 of them creating the venv and installing. All 45 outputs matched the
  published checksums and all 7 figures were produced. A second run reused
  the venv in under a second. The models were already in this machine's
  cache, so the first-time download wasn't exercised, and it's the same
  machine, so this doesn't stand in for a different one. That's what the
  release and `compare-embeddings` are for.

## Documentation cleanup

Tool: Claude Code on Opus 5.5, same session.

Prompt: "Could you clean up the repo and make it less messy, particularly
the .md files tracked in git and the report? For the last step I will
write the report by hand."

The tracked docs had grown by accretion, each pass adding "update"
paragraphs on top of earlier ones. The agent rewrote `DESIGN_NOTES.md` to
state the current result in each section, with each earlier mistake kept
as a sentence or two and superseded numbers dropped unless a decision was
made on them. Section numbers stayed the same so the pointers in the code
and the report still resolve, and a map at the top groups them by topic.
It restructured the README around the one-command path and tightened
this file. In the code it replaced comments that still claimed the old
"beats flat" result with short pointers and fixed two section pointers
that named the wrong section. It spot-checked restated numbers against
the output files (the blind-eval counts, perturbation ARI, the affinity
mass share), which corrected one: perturbation ARI at level 1 is 0.77, not
0.78. Only comments and docstrings changed in the code, and the tests and
validator were rerun afterwards.


## Code and folder cleanup before upload

Tool: Claude Code on Opus 5.5, same session.

Prompt: "Is the code readable, easy to understand and easy to navigate?
Maybe some file and folder management? I am really close to uploading."

The agent reviewed the layout and code and reported what a reviewer would
trip on, then asked how much to change so close to upload. I chose the
cleanup plus a folder split. What it changed:

- `scripts/` was one flat folder of 26 files. It is now `pipeline/` (the
  steps `reproduce_all.py` runs), `experiments/` (the section 15 sweeps)
  and `release/`. `reproduce_all.py` and `verify_release.py` stayed at the
  top because the published Hugging Face card points at those paths.
  `t6_patch_extrinsic.py` became `t6_extrinsic.py`, since it's a required
  step and not a patch. Every path in the docs, the report source and the
  code messages was updated, and a check confirmed every referenced script
  path exists.
- Settings that had been copied into scripts (`ALPHA`, `LEVEL_TARGETS`,
  the k-NN size and the data path) are now imported from `src/tkh`, so
  changing the pipeline can't leave an experiment on an old value. The
  values were identical, so nothing about the results changed.
- Dead code removed: `t2_build_hierarchy.py`, an early smoke test still on
  alpha=0.5, and two functions in `eval/extrinsic.py` that nothing called.
- The `.gitignore` had been ignoring itself, so the published repo had
  none. It now holds only generic patterns, and the names of my private
  working files moved to `.git/info/exclude`, which stays local.

Verification: 61 tests pass, every script imports and resolves the repo
root from its new location, and a full replay run in a clean copy of
exactly the files that would be pushed reproduced all 45 outputs and all
37 release files byte for byte against the published checksums. That run
took 143 minutes of wall time, but the laptop was in standby for about two
hours of it, according to the Windows power log.

## Static visualisation of levels 0-2

Tool: Claude Code on Opus 5.5, same session.

Prompt: I pointed it at the optional deliverable in the brief, "a static
visualisation of levels 0-2 across snapshots (a screenshot is enough). Do
not build an interactive frontend", and asked it to make it.

The agent chose the form: one column per snapshot with the 12, 50 and 200
super-nodes as nested bars, each child inside its parent's span and height
equal to node count on one scale for all four snapshots, plus ribbons
between columns sized by the level-0 members each snapshot shares with the
next. Colour follows the persistent level-0 id. The eight ids alive at
three or more snapshots get the eight hues of a palette it ran through the
dataviz validator, the rest are gray, and ids born at a snapshot are drawn
pale at every level. It's `fig_hierarchy` in
`scripts/pipeline/make_report_figures.py`, so `reproduce_all.py` redraws
it, and it reads only the shipped `hierarchy.json` files. The PNG is
excepted from the `.gitignore` so it shows on GitHub without a rerun, the
README shows it, and the report got one sentence pointing at it rather
than a figure, to stay at five pages.

Verification: before drawing, the agent checked that every level is an
exact partition of its parent at all four snapshots (child member counts
sum to the parent's) and that no node leaves between snapshots, which is
what makes the nested bars and the ribbons exact rather than approximate.
Its "born" marking gives 152 births after 2020, the same count as
`temporal_events.json`. The palette passes every check, but three hues sit
under 3:1 contrast on white, so every block is numbered or labelled. The
agent rendered the figure and looked at it three times, fixing a title
collision, missing id numbers on mid-sized blocks, two legend swatches
that looked the same and a clipped legend. The ribbons are recomputed from
member overlap, not read from `temporal_events.json`, so the figure is a
second view of the matching, not a check on it. The report still builds
at five pages.

## Simplification pass

Tool: Claude Code on Opus 5.5, same session.

Prompt: "Do one pass over the whole repo and simplify the language and
the code. I don't want it to seem complex and convoluted, just straight
to the point. Also make it more correct, because keeping things simple
usually deals with a lot of the complexity."

The agent read every module and script before changing anything, and
held one constraint throughout: the 45 checksummed outputs and every
recorded model call had to stay byte-identical, so each code change had
to be a pure restructuring. What changed:

- Duplication. The 1/(k-1) clique expansion was written out three times
  (the fine-level affinity, the coarse affinity and the shuffle null) and
  is now one function, `hypergraph.py`, `clique_expansion`.
  `t6_evaluate.py` computed an extrinsic section that `t6_extrinsic.py`
  overwrote one step later, so it now lives only in `t6_extrinsic.py`,
  which also scores each drill-down setting once and reads the sweeps and
  the shipped row from that grid instead of rerunning them in three loops.
  Loading a hierarchy and replacing one section of `metrics.json` are
  helpers in `tkh.io` instead of the same lines in six scripts, and the
  t-interval helper that existed twice is in `eval/stats.py`.
- Dead code. `projection_loss_report` (never run, although DESIGN_NOTES
  quoted its numbers), `clique_explosion_comparison` and its test, an
  unused table in `classify_events`, a redundant argument to
  `track_across_snapshots` and statistics nothing read.
- Correctness. The 94% and 63% in DESIGN_NOTES sections 3 and 4 are now
  printed by `t1_describe.py`; before, no script produced them.
  `export_release.py` would have checksummed the untracked figure PDFs on
  a re-export, so a clean clone would fail verification; it now leaves
  `outputs/figures/` out. `validate_hierarchy.py` also checks each child
  sits exactly one level below its parent, and the agent confirmed it
  flags a deliberately broken file. `t6_extrinsic.py` stops with an
  instruction if labels are missing instead of quietly using fewer.
  Lookups that fell back silently on data that's always present now index
  directly, so a real gap would fail loudly.
- Language. Docstrings say what the code does, with the reasons left in
  DESIGN_NOTES. The README, DESIGN_NOTES and two paragraphs of the report
  were tightened, with pointers updated to the new function names. The
  release card template no longer asks people not to redistribute the
  data, to match the public dataset.

The experiment scripts only got docstring edits. They produced
checksummed outputs that the pipeline doesn't regenerate, so a change to
their logic couldn't be checked by a replay.

Since the clique expansion is now one small function, the agent added
four tests for it: the 1/(k-1) weights, articles and authors dropped
from edges, and the high-arity share. Before, affinity construction had
no direct test.

Verification: pyflakes finds nothing, all 27 scripts import, and 64 tests
pass (the removed function's test went, the four new ones came in). Before
the full run, a replay of `run_pipeline.py` and `t5_apply_labels.py` gave
byte-identical hierarchies, temporal events and T1 statistics, with all
248 labels applying, so every labelling prompt is unchanged, and the
rewritten `t6_extrinsic.py` gave a byte-identical `metrics.json`. Then a
clean copy of exactly the files that would be pushed ran the whole
pipeline on the published model outputs: all 12 steps passed in 16
minutes, and all 45 outputs and all 37 release files matched the published
checksums byte for byte. The report still builds at five pages. The card
already on Hugging Face still has the old data paragraph; changing it
means re-uploading its README and `SHA256SUMS.txt`, which hasn't been
done.

## Final touch-ups

Tool: Claude Code on Opus 5.5, a new session.

Prompts: "Do a full pass through the repo, using the selected file as a
grading metric. How much out of 100 would you write me on this
assignment, would you hire me as an intern and what would you do as last
touch-up steps. I have a couple hours of work left on this thing before
submission" [the selected file was the brief], then "do 2, 3 and 4".

The agent read the brief, the report source, the core modules and the
docs, ran the tests and the validator, and gave a per-criterion estimate.
It put the biggest loss on T4, since the brief says the collapse rule
must be used by the method and the shipped levels don't use it. It
suggested five touch-ups. I took three: a table of the actual level-0
labels in the report, a count of temporal leaks in the labels in place of
the single EquiformerV2 example, and this summary table plus a reading
guide at the top of DESIGN_NOTES. I left out changing the method this
late, and it pointed out that anything touching the clustering would
invalidate the labels and blind ratings.

The temporal honesty audit (`scripts/experiments/temporal_honesty_audit.py`,
DESIGN_NOTES section 25). The agent wrote the rule into DESIGN_NOTES
before running anything, with a 2% bar. The first run flagged 94 of 186
labels, nearly all on field vocabulary. It then added a check for
whether a sampled member the corpus had already seen contained the term,
which cleared 76 of the 154 matches. It read the other 78 itself and
listed seven matches in five labels as real leaks, checking each against
the members the labeller was actually shown. For example, the 2020
"DeePMD-kit" traces to a sampled "DeePMD-kit v2" first seen in 2023, and
the 2022 "MPNN" is a false alarm because the corpus had MPNN from 2017.
The verdicts are a list in the script, so the numbers regenerate, but
they're the agent's reading, not mine. 5 of 186 is over the 2% bar, so
the report now says this part of P6 fails.

The report. The agent added the level-0 table and the audit result, and
cut elsewhere to stay at five pages. It rebuilt the PDF with Tectonic and
looked at every page. One shell edit went wrong: a sed meant to fix table
row endings also matched dollar signs and corrupted the maths throughout
`report.tex`. The agent caught it in its own check of the file, restored
it from git (it had been clean before this session) and redid the edits
with exact string replacement.

Verification: the audit's summary and intervals come from its JSON; the
five leaking labels were checked against their labeller inputs; the
report builds at five pages with no LaTeX warnings; 64 tests still pass.
No pipeline code and no checksummed output changed. The audit writes one
new file, `outputs/temporal_honesty_audit.json`, which isn't in the
published checksums.

Prompts: "Should I just get rid of the data from the repo?", then "make
the repo public, just get rid of the data, and put in readme on how to
reproduce. Keep huggingface cause its whatever". The agent recommended
making the repo private instead, to keep one-command reproduction, and I
chose a public repo without the data. It checked that the repo was
already public and pointed out that the data stays in the git history,
since removing it now only affects later commits. It removed `data/`
from the repo (the local copy stays, ignored), added the unzip step to
the README, and made `reproduce_all.py` stop with that instruction if a
data file is missing. It also warns if a file's SHA-256 differs from
mine. `load_tkh` gives the same instruction when a single script is run.
Checked by moving `data/` aside: both the one command and a single
pipeline step stop with the message, and with the data back the dry run
and the tests pass.


## Verification, overall

There are 64 unit tests now. They started at 16 (the T4 collapse rule, T3
matching and the faithfulness held-out split) and grew with each pass:
sparse UPGMA against scipy, held-out edge cohesion, the leave-one-out and
paired statistics, the multilevel variants, the ground-truth matcher, the
label staleness guard, the stability intervals, the blind packets and
scoring, model-output record and replay, custom label and rating sets,
and the clique expansion. The coherence null still has no direct test.
`scripts/pipeline/validate_hierarchy.py` checks the laminar-partition
property exactly on every snapshot. Both were rerun after every
non-trivial change, including cosmetic ones. The strongest check is the
last one: a from-nothing run that reproduced all 45 output files byte for
byte against published checksums. I take responsibility for every number
and claim in the submission, whether I typed the underlying code or not.
