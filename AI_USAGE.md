# AI usage

Tool: Claude Code, running on Sonnet 5, for the entire project. No other AI
tool was used. Below is every phase where it did real work, the prompt that
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
by aggregating full member lists into sentence-style premises); and the
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
rule (only replace 0.5 if some value beats it on every metric at every
level, not just on average) went into `FIXES.md` before the sweep ran, not
chosen after seeing which value happened to look good.

alpha=0.3 was the only value that passed that bar. I had the agent apply
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

Every module has unit tests (10 total) and the pipeline output is checked
by a structural validator (`scripts/validate_hierarchy.py`) that verifies
the laminar-partition property exactly, not by assumption. I re-ran both
after every non-trivial code change, including purely cosmetic ones, to
confirm they didn't silently change behavior. I take responsibility for
every number and claim in `report.pdf` and `DESIGN_NOTES.md`, whether I
typed the underlying code or not.
