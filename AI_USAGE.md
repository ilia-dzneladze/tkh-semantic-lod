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
tune blind. I didn't get to that ablation in this pass, it's in
`report.md`'s next-steps section.

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

## Verification habits, generally

Every module has unit tests (10 total) and the pipeline output is checked
by a structural validator (`scripts/validate_hierarchy.py`) that verifies
the laminar-partition property exactly, not by assumption. I re-ran both
after every non-trivial code change, including purely cosmetic ones, to
confirm they didn't silently change behavior. I take responsibility for
every number and claim in `report.pdf` and `DESIGN_NOTES.md`, whether I
typed the underlying code or not.
