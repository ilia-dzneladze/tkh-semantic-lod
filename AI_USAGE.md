# AI usage

Tool: Claude Code, running on Sonnet 5 for the build and follow-up
experiments, on Opus 5 for the review and correctness fixes, and on Opus
5.5 for the last pass, which also used Opus 5.5 subagents as blind raters
(last section). No other AI tool was used. Below is every phase where it did real work, the prompt that
kicked each one off, what I accepted or changed, and what I checked before
trusting the output.

## Planning

Prompt: "Read the .md file in the root folder outlining the task, explain
what the task is about... Then scaffold a plan, what language to use, what
dependencies, what would be the best way to implement the solution... if
you need approval from me for design choices, ask, I will weigh the
trade-offs."

The agent read the brief and the data, proposed Python plus a specific
dependency list, and surfaced four decisions for me instead of picking them
itself: how to combine structural and semantic signal (I picked a joint
weighted affinity graph, `alpha * structural + (1-alpha) * semantic`), the
embedding model (I picked general-purpose MPNet over a domain-specific
one), the labeling approach (I initially picked an API call, see below),
and the faithfulness check method (I picked an NLI model). I accepted the
language/dependency choices as proposed. I asked separately whether the
API calls would cost money against my Pro subscription, confirmed they
would, and switched the labeling plan to have the agent read cluster
contents directly instead.

I set alpha to 0.5 as a starting point rather than a final answer. My
expectation going in was that the right balance between structural and
semantic signal would depend on how noisy the hyperedge structure turned
out to be once I could actually see coherence and stability numbers, not
something to guess correctly up front, so I flagged it from the start as
a value to revisit once the evaluation existed rather than something to
tune blind. I didn't get to that ablation in the first pass, it went into
`report.md`'s next-steps section at the time; I came back and ran it
afterward, see "Alpha ablation and relabeling" below.

## Feasibility check

Prompt: "Look over the plan, see if it's coherent and really doable
according to my skills as a 3rd year cs student, and continue if you think
this will land me the job."

The agent gave an honest assessment rather than blanket reassurance: most
components are assembling known library pieces, but the hypergraph Laplacian
construction is graduate-level and I'd need to actually understand it, not
just accept the formula. I went through that with the agent until I could
restate why the 1/(arity-1) weighting is correct. It also simplified the
T6 null model (random-labels instead of a full degree-preserving hypergraph
shuffle) on its own initiative, which I accepted as a reasonable
risk/effort trade-off explicitly permitted by the brief.

## T1-T5 implementation

No single new prompt per task, this was continuous direction: I reviewed
each module's output as it was produced and asked follow-up questions or
corrections in place, rather than writing a fresh detailed spec for T2, T3,
T4 individually.

What I accepted: the Zhou et al. hypergraph affinity weighting (later
corrected: it is a weighted clique expansion, not their method; see the
final review section), the sparse
UPGMA implementation, the T3 Jaccard-matching mechanism, the T4 collapse
rule, all as proposed after I understood the reasoning.

What I caught and had fixed myself, not the agent: reviewing the generated
`hierarchy.json` directly (not just trusting a clean run) surfaced a
self-reference bug in the persistent-id scheme and a design error where
`member_ids` didn't match the brief's own definition of a laminar partition.
Both got fixed and re-verified against a validator script I had written for
exactly this purpose.

## Code review pass

Prompt: "Go over what's been written now. What the scripts do with the data
and what we are getting ready for. And look at the logic, is it optimal and
coherent?"

The agent re-read all eight source modules and reported six issues: dead
code, an inconsistent stats dictionary, a docstring that mislabeled a
union k-NN graph as "mutual," an undocumented correctness assumption in the
clustering merge step, and two pieces of now-redundant code from the earlier
member_ids fix. I had it fix all six and rerun the full pipeline to confirm
the numbers didn't change, since a real fix to dead/cosmetic code shouldn't
change output. The correctness assumption (treating a missing edge as
similarity zero) is the one I actually worked through the proof of with the
agent rather than take on faith, since it affects whether the whole
hierarchy's laminar structure is provably sound.

## T5 labeling

Prompt: "continue with T5 labeling."

The agent dumped cluster member lists to files and then labeled them
itself (four parallel sub-agents, one per snapshot, reading real member
text rather than any automated shortcut), following a fixed prompt template
I'd have it write out so the process stayed auditable. I accepted the
labels after spot-checking word limits (found and fixed 9 entries that had
slipped past the 6-word/25-word limits) and reviewing the low-confidence
clusters each sub-agent flagged on its own.

## T6 evaluation

Prompt: "continue with T6 evaluation."

This phase had two points I worked through directly rather than accept the
first result: the label-faithfulness check initially returned "neutral" on
nearly every pair including obvious mismatches, which needed real diagnosis
(traced to bare short phrases not giving the NLI model enough context, fixed
by aggregating members into sentence-style premises; the premise turned out
to use a subset of the labeller's own input, which was caught and fixed in
the final review, see the last section); and the
extrinsic retrieval eval initially returned zero recall on every question,
which I didn't accept as "the corpus is just this hard" without checking,
and which turned out to be a real embedding problem (bare acronyms don't
relate to full-sentence questions) with a real fix (enriching short names
with their hypergraph neighborhood).

The branching factor (b0, b1) for the drill-down eval is the other place I
changed course based on the numbers rather than the first setting I tried.
I'd picked (3,3) as an initial guess, narrow on purpose since the whole
point of drilling down is to score fewer candidates than the flat
baseline, but a narrow branch is also exactly the kind of choice that can
exclude the right answer before the final ranking step ever sees it, which
is a predictable failure mode of any coarse-to-fine search, not specific
to this corpus. When (3,3) came back at roughly half the flat baseline's
recall, that matched what I'd have expected from too narrow a branch
rather than a real limitation of the hierarchy, so instead of reporting it
I had the agent sweep wider values. (5,5) matched the flat baseline
exactly at about 16% of the candidates, which is the number that made it
into the report at the time. It didn't survive later changes; see the
sections below.

## T7 write-up

Prompt: "continue with the T7 write-up."

Figures were generated directly from `metrics.json`, not hand-copied
numbers, so they can't drift from the underlying data. I reviewed
`report.md` for accuracy against the actual metrics before treating it as
final, including re-checking the T4 collapse numbers cited in the text
against the current `hierarchy.json`.

## Alpha ablation and relabeling

Prompt: "Okay, generate a plan for each fixable iteration, starting with
alpha ablation, keep them in a seperate ignored file called FIXES.md, and
if a fix will bring betterment then keep it and commit. generate the plan
and start the alpha sweep: 5-95, 10-90, 15-85, ..., 95-5"

I had the agent write `scripts/alpha_sweep.py` to sweep alpha in 5% steps
(19 values), scored by coherence and stability only, the two T6 metrics
that don't need hand-written labels. Before trusting a 19-value run I had
it smoke-test the script against alpha=0.5 alone first; it reproduced the
shipped `metrics.json` numbers to the exact digit, which is what made me
comfortable letting the full sweep run unattended after that. The decision
rule written down before the sweep (in a private working file, copied into
`DESIGN_NOTES.md` section 15) was: replace 0.5 only if some value beats it
on both averaged coherence z-score and averaged stability ARI. Several
values passed that. After seeing the results I added a stricter check,
beating 0.5 on each of the nine per-level numbers separately, and
alpha=0.3 was the only value that passed it. That stricter check came
after the results, so it's a robustness check, not the pre-registered rule. I had the agent apply
it to the real pipeline, not just the sweep's lightweight version,
regenerate `hierarchy.json`, and re-check the structural validator and
unit tests before doing anything else with it.

Prompt: "Okay, add this to the report.md as a finding, do the relabeling
for 0.3 and see if the end result is really better with alpha equalling
0.3 and also provide a logical justification for this in the report.md"

Moving alpha meant the clusters changed, not just their weights, so the
old labels (248 of them, levels 0-1 across 4 snapshots) no longer
described the right members. I had the agent relabel from scratch the
same way as the original T5 pass, four parallel sub-agents, one per
snapshot, reading real member lists. I independently verified all four
outputs myself with a word-count script rather than trusting each
sub-agent's self-report (0 violations across 248 entries), and looked at
the specific clusters each sub-agent flagged as hard calls rather than
just taking the "all good" summary at face value.

Re-running the full T6 suite against the new labels surfaced a real
regression I hadn't anticipated: the branching factor (5,5), exactly right
under alpha=0.5, no longer matched the flat baseline under alpha=0.3
(0.027 recall vs. flat's 0.0325). I had the agent re-sweep it the same way
as the first time rather than assume the old value still held; (8,8)
matches flat again. That's the clearest illustration in this whole pass of
why changing one parameter isn't a one-line commit here, it has knock-on
effects through labels, faithfulness, and the extrinsic eval that all
needed rechecking, not assuming.

## Beating flat baseline, not just matching it

Prompt: "Okay, now on to the next likely fix that will increase the
retreival efficiency compares to baseline, it's really interesting that
none of the last two could beat it, only match it. Try a new fix, test,
and conclude"

Before writing any code I had the agent work out why every branching-factor
value tried so far could only tie flat baseline recall, never beat it: the
drill-down eval only ever ranks a subset of flat's candidate pool using
flat's own scoring function, which structurally caps it at parity. I had it
write `scripts/rerank_sweep.py` to test a different mechanism, blending
each candidate's own embedding score with its level-1 ancestor's
label+gloss score before ranking, information flat has no access to at
all. Decision rule went into `FIXES.md` before running it: only keep this
if some beta value beats flat's recall outright, not just ties it, at no
more candidates than the shipped (8,8) pool.

beta=0.6 did: mean recall 0.0423 against flat's 0.0325, mean precision
0.0286 against flat's 0.0143, same 774 candidates. Before accepting that, I
had the agent rerun the same beta sweep against the full unrestricted pool
to check the gain wasn't just an artifact of which nodes the (8,8)
restriction happens to keep, it held there too. I picked beta=0.6 over
beta=1.0 myself even though beta=1.0 scored marginally higher recall,
since it came with visibly worse precision and meant discarding the node's
own embedding signal entirely, and over beta=0.9's single-point dip, which
reads as noise from averaging over only 14 questions rather than a real
effect. I had the agent wire the result into the real pipeline (not just
the sweep script): `beta` added to `hierarchy_drilldown` in
`src/tkh/eval/extrinsic.py`, `outputs/metrics.json` regenerated through the
actual shipped code path (not the standalone sweep) as a second check that
the numbers held, and a new figure added rather than silently dropping the
finding into text only.

## Verification habits, generally

There are 36 unit tests. They started at 16, covering the T4 collapse
rule, T3 temporal matching and the faithfulness held-out split, and grew
with each later pass: sparse UPGMA against scipy, the held-out edge
cohesion measure, the leave-one-out and paired statistics, the multilevel
variants, and the ground-truth matcher. Affinity construction and the
coherence null still have no direct tests. The pipeline output is checked
by a structural validator
(`scripts/validate_hierarchy.py`) that verifies the laminar-partition
property exactly, not by assumption. I re-ran both
after every non-trivial code change, including purely cosmetic ones, to
confirm they didn't silently change behavior. I take responsibility for
every number and claim in `report.pdf` and `DESIGN_NOTES.md`, whether I
typed the underlying code or not.

## Final review and correctness fixes

Tool: Claude Code on Opus 5.

Prompt: "Look at my current version of the project. Do you think it is
sufficient to land me the job? ... I want your honest opinion, and if you
think we can iterate on some things, make some things better, than tell me
what specifically and how. Be thorough"

The agent read the brief, all source, the docs and `metrics.json`, and
checked several claims against the code instead of trusting the docs.
Findings where docs and code disagreed:

- The faithfulness NLI premise was built from the first 15 sorted members,
  a strict subset of the 25 the labeller saw, while the docstring and
  report said it used the full member list.
- Temporal Jaccard used the full union including nodes new at t+1, while
  the docstring and DESIGN_NOTES said it was restricted to shared nodes.
  The agent measured the effect first: at level 1, 2022 to 2024, 14 of 50
  clusters fell below MATCH_THRESHOLD, against 3 of 50 restricted.
- The structural affinity was described as hypergraph-native and following
  Zhou et al.'s spectral method. In the code it's a weighted clique
  expansion with no spectral step.
- The T4 collapsed hypergraph is computed but never used by the method.
- The extrinsic "beats flat" result comes from a single question (Q14),
  after tuning on the same questions.
- This file said every module had unit tests, and it misstated the alpha
  decision rule (both fixed above).

It also ran one new diagnostic: the (8, 8) drill-down pool holds 80% of
ground-truth nodes at 25% of candidates, against about 25% for chance.
(Later found to depend on the labels; see the next section.)

Prompt: "yes, start with the P0 fixes"

What the agent changed, and what I accepted:

- `temporal.py`: Jaccard on shared nodes only, and split events now record
  persistent ids. Three new tests. The pipeline was rerun and every
  snapshot's cluster member sets were checked identical before and after,
  so only matching changed. Persistent ids that got reassigned were remapped
  in the label files by exact member-set match, not relabelled. The
  threshold sweep was rerun: MATCH_THRESHOLD is still sensitive.
- `labeling.py`, `eval/faithfulness.py`: one function defines the
  labeller's input, and the NLI premise is now a seeded sample of held-out
  members only. Clusters with fewer than 5 held-out members are skipped and
  counted. Now reports both contradiction and not-entailed rates. Three
  new tests check the held-out set is disjoint from the labeller's input.
  Contradiction rose from 5-11% to 14-21%.
- `t6_evaluate.py` can rerun single sections. The shuffle-null script also
  writes into `metrics.json`, and it reproduced its earlier values
  exactly.
- Docstrings, DESIGN_NOTES (sections 3, 4, 7, 9, 10, 12, 13, 14, new 15),
  README and this file corrected to match the code. Dangling references to
  my private working file were replaced with DESIGN_NOTES pointers.

Verification: 16/16 tests pass, `validate_hierarchy.py` passes on all four
snapshots, all 248 labels re-applied. The type bias in the labeller's
sample was found and documented but not fixed, because that needs
relabelling.

## Multilevel coarsening (T4 driving the coarser levels)

Tool: Claude Code on Opus 5.

Prompt: "now start on P1, make T4 drive the coarser levels"

The agent wrote the keep-or-discard rule into DESIGN_NOTES section 15
before running anything. Then it implemented `coarsen_one_level` in
`pipeline.py` and `coarse_structural_affinity` in `collapse.py`: each
coarser level clusters the super-nodes below it on the T4-collapsed
hypergraph plus centroid embeddings. It refactored clustering into one
`build_levels` function so the pipeline and the perturbation check share
code. Before comparing anything, `scripts/coarsening_compare.py` checks
that the refactored dendrogram path reproduces the shipped hierarchy.json
member sets exactly. It does.

The first multilevel variant failed the rule. The agent proposed one
principled fix (size-normalised coarse affinity with count-weighted
average linkage), wrote a second rule for it before running, and noted
that it was chosen after seeing a failure. That also failed, so per the
rule the shipped pipeline stays on the dendrogram and the multilevel code
stays in as a non-default option. I accepted stopping there instead of
trying more variants until one passed.

The same script found that routing on member centroids instead of labels
gives no lift over chance, so the earlier 80%-at-25% routing result
depends on the labels. The agent corrected DESIGN_NOTES section 14 and
the report.

Verification: 21 tests pass, including laminarity and the size budget for
both multilevel variants on a toy hypergraph, and a check of the weighted
UPGMA arithmetic. The laminarity validator passes on the shipped outputs,
which didn't change.

## Relabelling without access to the questions

Tool: Claude Code on Opus 5 for setup and evaluation; four fresh Claude
Code sub-agents on Sonnet 5 (the same model as the original labels) for
the labelling.

Prompt: "start with step 1, the relabelling"

The main agent had read the question files earlier in the session, so it
didn't write labels itself, and it didn't use forked agents because they
would inherit that context. It asked me how to run the labelling, and I
chose fresh sub-agents. Before any new labels existed, it wrote the
decision rule into DESIGN_NOTES section 15, archived the first labels in
`outputs/labels_v1/`, and measured their routing as a baseline.

Code changes: `labeller_sample_ids` now draws a seeded random sample
(`sampling="first"` reproduces the old one), the prompt includes the
node-type counts it had always claimed to include, faithfulness takes the
sampling mode so its held-out set matches the labels being checked,
`routing_pool_recall` is shared by `scripts/label_routing.py` and
`scripts/coarsening_compare.py`, and `scripts/t5_import_labels.py`
validates and merges replies. Two new tests.

Each sub-agent got a prompt with only the path to its batch file (in a
scratch directory holding nothing else) and an output path, and was told
to use no other files or tools. No tool was technically blocked, so the
main agent then read the sub-agents' tool logs: each made two reads of its
own batch file and one write, and no log mentions the question files. All
248 labels passed the id and word-limit checks with no edits.

Results, run through the pre-registered rule: label routing still clearly
beats chance (lift 0.42, CI 0.23 to 0.57, against 0.55 for the old labels),
faithfulness improved (contradiction 5-14% against 14-21%), and recall@20
now ties flat instead of beating it. Settings were not retuned. I accepted
the new labels as the shipped set.

## Extrinsic re-evaluation with leave-one-out

Tool: Claude Code on Opus 5.

Prompt: "Continye with step 2"

The agent wrote the decision rule into DESIGN_NOTES section 15 first,
then added `leave_one_out_select` and `paired_comparison` to
`eval/extrinsic.py` (four new tests, including one checking that a
held-out question's own score can't influence the setting chosen for it),
extended `scripts/t6_patch_extrinsic.py` to write the leave-one-out result,
the in-sample comparison and the routing curves into metrics.json, and
added a routing figure. It checked the figure's two colours with the
dataviz skill's palette validator and looked at the rendered image before
using it.

Result, read through the pre-registered rule: leave-one-out drill-down
against flat is -1.5 points of recall@20 (CI -6.3 to +2.9, p = 0.59), so
no detectable difference, and label routing beats chance at every budget.
I accepted reporting it that way. Nothing in the hierarchy or labels
changed. (These numbers were later redone after a ground-truth bug was
found; see the last section. The verdict didn't change.)

## Structural held-out coherence

Tool: Claude Code on Opus 5.

Prompt: "continue with step 3"

The agent wrote the decision rule into DESIGN_NOTES section 15 first. It
checked the data for a way to hold out whole papers (every edge has a
`provenance.article_id`) and added that as a stricter second scheme next to
random-edge holdout, since edges from one paper are correlated. It added
`heldout_edge_cohesion` to `eval/coherence.py` with two tests, wrote
`scripts/structural_holdout.py` (7 alphas, 2 schemes, 5 seeds, TF-IDF
coherence on the same clusterings), and added a trade-off figure. After
rendering the figure it replaced an alpha label that sat next to the
wrong series with a ring on the shipped alpha. It also found that
alpha=1.0 is degenerate (about 1,100 forced merges, one cluster) and left
it out of the figure with a note, rather than reading it as "structure
carries nothing".

Result, read through the pre-registered rule: at alpha=0.3 structure
earns its place only at level 0, most of its predictive power doesn't
survive holding out whole papers, and no alpha dominates 0.3. I kept
alpha at 0.3.

## Newer literature

Tool: Claude Code on Opus 5.

Prompt: "I got new literature under literature/new/, see if anything
could be useful for our case from there and cite the most important
ones, maybe there is something that can help more than the current stuff"

I collected the eight PDFs myself (the agent had earlier written a search
prompt I could give another LLM). The agent extracted their text with PyMuPDF and read
the abstracts, method sections and conclusions. It picked five to cite,
each tied to one design choice: Ruggeri et al. (pair overlap and
detectability), Ma et al. AdE (feature-aware clique expansion), Asgari et
al. (No-Smoothing vs smoothed dynamic community detection), SHyPar
(local vs spectral coarsening) and DeWolfe and Theberge (edge clustering,
overlapping communities). It left out Gong et al. (spectral hypergraph
embedding), Kirkley (choosing snapshot windows) and FeClustRE (LLM
labels for app-review clusters) as less directly relevant. Since Ruggeri
et al. make a testable claim, it wrote `scripts/pair_overlap.py` to count
how often concept pairs recur across hyperedges and papers (1.7% across
papers in 2026) and used that to explain the weak paper-level held-out
result. Two of its drafts claimed more than the data showed, and it fixed
both before I saw them. One said held-out pairs were "covered by the same
paper's other edges", but the exact pair almost never repeats, so it
rewrote that as connection through the paper's other edges. The other
used "transformer" as an example concept, which isn't in the corpus, so
it replaced it with MAE after checking. All papers are cited by arXiv id.
The agent did not verify published venues, so that's on me to check.

## Localisation of change (P5)

Tool: Claude Code on Opus 5.

Prompt: "yes, start with the localisation test"

The agent wrote the decision rule into DESIGN_NOTES section 15 before
writing any code: per-cluster churn against the share of new hyperedges
around the cluster, Spearman with a bootstrap CI, with cluster size
partialled out. It then wrote `scripts/localisation.py`, which reads
the shipped hierarchies and doesn't depend on the section 10 event
thresholds. The test failed at every level. Before writing that up, the
agent broke the result down by transition and checked the exposure
ranges, to rule out a bug or a pooling artefact. It then added one
exploratory measure after seeing the failure, the share of new nodes
among an old node's k-NN neighbours, which does correlate with churn at
levels 1 and 2. The docs label that measure post hoc, and the
pre-registered verdict ("not localised") stands. The explanation that
fixed cluster counts force global reshuffling is the agent's hypothesis
and is marked untested.

## UPGMA check, clean reproduction and stale-number pass

Tool: Claude Code on Opus 5.

Prompt: "continue with the original list of tasks so we finish first and
then we can experiment"

The agent added `tests/test_upgma_vs_scipy.py`, which compares
`sparse_upgma` with scipy's dense average linkage on random graphs:
merge heights, cuts, forced merges and point weights. It also checked
that the comparison fails against weighted average linkage, so the test
isn't passing trivially.

For the reproduction it copied exactly the files git would ship into a
scratch folder, built a fresh venv from `requirements.txt` and ran every
README step in order. Everything ran. The hierarchies, temporal events
and hypergraph-shuffle null came out byte-identical, and every aggregate
number in `metrics.json` matched. The run found three problems, and the
agent fixed each one:
- The README didn't say that a full `t6_evaluate.py` run rewrites
  `metrics.json` from scratch, so following it silently dropped figure 6.
  It now lists `structural_holdout.py` and `localisation.py` as required
  steps after it.
- The committed `metrics.json` held per-cluster coherence under persistent
  ids from before the temporal-matching fix. The values were right but
  51 keys at 2026 level 2 pointed at clusters that no longer exist. The
  agent replaced it with the reproduced file after confirming the values
  were the same under the new ids and that every aggregate matched.
- The T1 quality-note sample depended on set iteration order. `io.py`,
  `build_snapshot` now iterates in sorted order, and `t1_describe.py`
  output is byte-identical across runs.

The README timings were optimistic, so the agent replaced them with the
measured ranges. It then checked every number in `report.md` against
`metrics.json`. They matched, and it updated two stale passages: the
verified list, and a next step asking for an alpha=1.0 run that had
already been done.

## Skeptical review pass and the fixes from it

Tool: Claude Code on Opus 5, one session, used as a reviewer first and
then to implement the fixes I picked.

Prompt (abridged): "You are reviewing an internship assessment submission
in this repo, as a skeptical senior researcher on the hiring team would...
Your job is to find gaps: anything that would cost points or raise doubts
with a careful reviewer. Do not edit any files. You may run code to verify
a suspicion. Prefer checking over guessing... Separate two kinds of
finding: gaps the docs already admit and gaps nobody has mentioned."

The read-only constraint was deliberate on my part. I wanted findings with
evidence behind them before anything got changed, so the agent reproduced
the pipeline in a scratch copy of the repo rather than working in place. It
ran `run_pipeline.py` from a clean `git archive` of HEAD and confirmed the
hierarchies and `temporal_events.json` come back byte-identical and the
faithfulness numbers reproduce exactly, which is the first independent
check that the reproduction claim in this file actually holds. It also
found that `t6_evaluate.py` crashes with a KeyError if you follow the
README literally, because `run_pipeline.py` clears the labels and the
README only mentions `t5_apply_labels.py` under importing new labels.

Findings I acted on, and what I had it do:

- The ground-truth matcher resolved expected method names to chemical
  elements ("N", "P", "S" and "Si" are substrings of "physics-informed").
  It quantified this before proposing anything: 47 of 147 ground-truth
  node ids were surface forms of two characters or less. Fixed by
  requiring both sides of a substring match to be at least four
  characters, with `tests/test_ground_truth_match.py` pinning it, and
  every extrinsic number rerun. DESIGN_NOTES section 16.
- Temporal honesty was claimed as holding by construction and does not.
  The agent traced it from a 2020 label naming EquiformerV2 back to
  `build_snapshot`, and counted the affected nodes per snapshot. I chose
  to document the count and leave the nodes in rather than filter and
  relabel; the reasoning and what it costs are in DESIGN_NOTES section 2.
- alpha is not the mixing weight it reads as. It measured the realised
  structural share of the affinity mass (about 7% at alpha=0.3, not 30%).
  I had it write `scripts/affinity_mass_share.py` so the number is
  reproducible rather than a one-off. DESIGN_NOTES section 17.
- The perturbation stability measure only removes hyperedges, so it
  rewards a clustering for ignoring the hypergraph, and alpha was partly
  selected on it. The agent re-read the alpha sweep with that metric
  dropped. alpha=0.3 still wins, which I was glad about but had not
  assumed going in.
- The formal statement was prose with no objective in it. I had it
  rewritten with the actual definitions and with the honest framing that
  average linkage optimises nothing globally, so what I have is a
  criterion and a nesting guarantee.

What I did not accept: the agent's stronger reading of the TF-IDF
coherence check as circular. It is a weaker independence claim than I had
written, and I took its measurement (69% of MPNet neighbour pairs share a
token against 7% of random pairs) into the report as a named assumption,
but I don't agree it makes the check worthless, and the report says what I
think rather than what it suggested.

Verification of the fix pass itself: 36 tests pass (4 new), the laminarity
validator passes on all four snapshots, the figures regenerate from
`metrics.json` and now read their question count from it rather than
hardcoding 14, and the pre-registered rules in DESIGN_NOTES section 15
were re-read against the rerun numbers rather than rewritten. Section 18
says exactly which rules were rerun and which were untouched. Both rules
that depend on ground truth still pass, by a smaller margin, and that is
written down next to the old numbers.

Still open from the same review, deliberately not fixed here: the report
is well over the 3-5 page limit and has no T1 section or reference list,
the README is Windows-only and doesn't say `t5_apply_labels.py` is
required after `run_pipeline.py`, the cross-snapshot CIs are t-intervals
over three transitions, and the NLI faithfulness judge has never been
validated against a blind human rating.

## Closing the rest of the review: items 6 to 11

Tool: Claude Code on Opus 5.5, the same session as the section above. The
blind raters were two further Claude Code subagents on Opus 5.5
(`claude-opus-5-5`), described below.

Prompt: "continuw with 6-11 tasks, end-to-end, test implementation"

These were the open items the section above lists, plus the two I had
deliberately left: TF-IDF independence (6), validating the faithfulness
judge (7), stale numbers (8), report format (9), reproducibility (10) and
the cross-snapshot CIs (11). Before starting, the agent asked me one
question, because it was mine to answer: who should do the blind ratings
that items 6 and 7 needed. It pointed out that it couldn't be the rater
itself, since it had already seen the labels, the controls and the
questions. I chose fresh subagents over rating by hand.

What it built, and what I checked:

- Reproducibility (10). Both Hugging Face models are pinned to the
  commits these results came from. A label is now applied only if the
  prompt rebuilt from the cluster's current members matches the prompt
  the label was written from byte for byte. The agent confirmed all 248
  shipped labels pass, and wrote a test that swaps two clusters' members
  under the same ids. `t6_evaluate.py` stops with an instruction instead of
  a KeyError on an unlabelled hierarchy. The README now has one ordered
  path, platform-neutral commands and a Linux torch step. The Linux claim
  rests on the PyPI dependency metadata for `torch==2.14.0` (CUDA 13 and
  nvidia packages on Linux) and the PyTorch CPU index listing
  `2.14.0+cpu`. Nobody ran a Linux install.
- Cross-snapshot CIs (11). The agent's first replacement, a node bootstrap
  with replacement, was wrong. The level-2 interval came out entirely
  above the point estimate (0.651 to 0.682 around 0.647). It diagnosed the
  cause: duplicated nodes add same-cluster pairs, and ARI is driven by
  those. It switched to half-sampling without replacement, checked that
  the half-sample mean matches the full ARI, and added a test with many
  small clusters. It showed the old estimator fails that test before
  relying on it. I only accepted the change once every interval
  contained its estimate.
- Coherence independence (6). `scripts/signal_overlap.py` makes the
  TF-IDF/MPNet overlap reproducible: 69% of neighbour pairs share a term,
  against 7% of random pairs. The blind intruder test is in
  `eval/blind.py` and `scripts/blind_eval.py`.
- Faithfulness (7). Wilson CIs on every rate, a blind gloss rating
  compared with NLI on identical items, and a check against source-paper
  titles.
- For each new check, the agent wrote the keep-or-report rule into
  DESIGN_NOTES section 15 before generating any data. The results sections
  apply those rules as written, including the two that didn't come out
  my way: level 0 fails the intruder test, and the provenance check is
  inconclusive.

The raters. Each was a fresh general-purpose subagent with no
conversation context. Its prompt was a short instruction plus the path to
a folder containing only its own item file. The answer keys stayed in a
separate folder outside the repo until the ratings were in. The agent then
read both transcripts. Each subagent made exactly one Read, of its own
file, then returned its answers. The answers were saved from the
transcripts rather than retyped, and each `*_ratings.json` records the
model and the tool calls. The exact rater prompts are in
`outputs/blind_eval/*_rater_prompt.md`. The labels had been written by
Sonnet agents, so an Opus rater limits self-preference but doesn't remove
it. It's one rater per task, with no second rater and no human pass, and
the docs say so.

Report (9). The agent rewrote `report.md` from about 8,000 words to a
5-page version with a T1 section, the formal statement as actual
formulas, the new results and a reference list. It rendered the PDF and
measured the page count rather than estimating it. The local PDF helper
(`scripts/make_report_pdf.py`, gitignored) was changed to put figures
inline. The long draft is kept locally as `report_long_draft.md`. It
checked three new citations by web search before using them: Clauset,
Moore and Newman 2008; Peixoto 2014; and Chang et al. 2009. It did the
same for the venues of Greene et al. 2010, Loukas 2019 and Agarwal et
al. 2006. One claim in its own draft was wrong: that SHyPar doesn't
coarsen hypergraphs. It does, and the agent rewrote the sentence before
the PDF was built.

Number audit (8). The agent checked every number in the new report
against the output JSONs. That turned up one real overclaim, carried
through several earlier versions: the warm start's coherence cost,
written as "level 2 loses 17-41%". That range is the 2026 snapshot alone.
Averaged over the snapshots a warm start can change, the cost is 7% to
18%, and at 2022 it is mixed. The verdict under the pre-registered rule
still holds, but DESIGN_NOTES section 10 now gives the corrected numbers,
and says the rule never put a number on "meaningfully". This also means
the claim in "UPGMA check, clean reproduction and stale-number pass"
above, that every number in `report.md` matched `metrics.json`, wasn't
true at the time. The (8,8) drill-down passage and the warm-start range
were both stale or wrong.

Verification: 49 tests pass (13 new across the label guard, stability
CIs, the statistics helpers and the blind packets). The laminarity
validator passes. And the full README path was rerun from a fresh copy of
exactly the files that would ship; the next paragraph has what it
reproduced. The agent copied exactly the files git would ship (tracked plus new,
non-ignored) into an empty folder and ran every README step in order,
using the existing venv, so the dependency install itself wasn't retested
here. Every step exited cleanly in about 30 minutes. The T1 statistics,
the temporal events and all four hierarchies came back byte-identical,
and all eight sections of `metrics.json` matched the committed file
exactly, including the new blind-eval scores, the half-sampling intervals
and the provenance check.

## Publishing the model outputs for verification

Tool: Claude Code on Opus 5.5, same session.

Prompts: I asked whether the "weights generated at the end" could be
uploaded to Hugging Face so a reproducer could check my results, then
"yes go ahead, and then tell me commands to upload the embedding to
huggingface".

The agent first corrected my premise: nothing in the project is trained,
so there are no weights. What varies between machines is the output of
the two frozen models. It proposed publishing those outputs plus
checksums, and I agreed. Before building anything, it measured how
fragile the outputs are. On this machine, 514 of 5,428 node embeddings
differ in the last bits depending only on how they were batched. That
decided the design: outputs are recorded and replayed per whole model
call (`src/tkh/model_outputs.py`), not per text. It routed every
embedding and NLI call through that one module, and wrote
`reproduce_all.py`, `export_release.py` and `verify_release.py`, plus 5
tests. Two of those tests check that replay never loads either model.

Verification was three clean-folder runs of the whole pipeline:
- Recording: outputs identical to the repo. This showed recording doesn't
  change anything.
- Export: re-derives the four merge trees and refuses to write unless
  they cut into the shipped hierarchies exactly. They do.
- Final replay: a fresh copy of the final code, with no model calls,
  reproduced all 45 output files and all 37 release files to the byte, in
  8 minutes.

`compare-embeddings` on this machine found all 5,428 embeddings
bit-identical to the published ones, and all four nearest-neighbour
graphs identical.

Along the way the agent found and fixed two real problems (DESIGN_NOTES
section 23). Loading a pinned model contacted the Hub every time and
stalled on retries when the network dropped, so both models now load from
the local cache first. And `localisation.json` changed in the 16th digit
between runs, because a mean was summed over a Python set in hash order.
Every earlier check had compared only `metrics.json`, where the rank-based
summary never moved, so this had slipped through two "clean
reproductions". The checksum over every output file caught it, and it is
now proven stable under two different hash seeds.

It also made one mistake of its own, and caught it before relying on it.
A sed edit meant to point its comparison script at the new run failed
silently, and the first "identical" result compared the wrong folder. It
noticed the path hadn't changed, rewrote the script to take the folder as
an argument, and reran the comparison. It stopped one replay run partway
through because the localisation fix had made it obsolete, and replaced
it with the final run above.

Not done: the upload itself, which needs my account. The data in the
release is derived from Constructor's TKH export, so it goes to a
private dataset repo unless they agree otherwise.
