# AI usage

Tool: Claude Code, running on Sonnet 5 for the build and follow-up
experiments, and on Opus 5 for the final review and correctness fixes (last
section). No other AI tool was used. Below is every phase where it did real work, the prompt that
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

What I accepted: the Zhou et al. hypergraph affinity weighting, the sparse
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
into the report.

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

There are 16 unit tests, covering the T4 collapse rule, T3 temporal
matching, and the faithfulness held-out split. The other modules
(clustering, affinity construction, coherence, stability, extrinsic) have
no unit tests. The pipeline output is checked by a structural validator
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
