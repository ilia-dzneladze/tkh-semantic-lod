# Design notes

Why the code does what it does, where that isn't obvious. Each section
starts with the files and functions it covers, and the code points back
here by section number. Where I got something wrong along the way, the
section says so briefly and then gives the current state. The full history
is in `AI_USAGE.md` and git.

This file is long, and most of it is detail behind a sentence in the
report. If you only have time for a few sections, read these, roughly in
the order of the brief's grading:

- 7 and 17: how structure and meaning are combined, and why alpha=0.3
  turned out to give structure about 7% of the say
- 8: why average linkage with missing entries read as 0 is laminar by
  construction, and the test that checks it against scipy
- 13 and 21: how coherence is measured without the embedding that built
  the clusters, and the blind intruder test
- 22 and 25: label faithfulness, the blind gloss ratings, and the
  temporal honesty audit
- 9: the T4 collapse rule, and why the shipped levels don't use it
- 10: temporal matching, and why change isn't localised

Sections are numbered in the order things came up, not by topic:

- data and snapshots: 1, 2
- building the hierarchy: 3, 4, 6, 7, 8, 17
- collapse, output format and time: 9, 10, 11
- labels: 12, 25
- evaluation: 5 and 13 (coherence), 21 (blind intruder test), 20
  (stability intervals), 22 (faithfulness), 14 and 16 (extrinsic)
- decision rules written before each experiment: 15, 18, 25
- reproducibility: 19, 23, 24

## 1. Which node types get clustered

`io.py`, `CONCEPT_TYPES` and `build_snapshot`.

The graph has 12 node types. Two of them, article and author, aren't
concepts in the sense the task means: an author is a person and an article
is a container for ideas. So only the other ten (method, technique, task,
problem, dataset, metric, component, cited_work, future_topic, claim) go
into `concept_ids` and get clustered. Articles and authors stay in the
snapshot as hyperedge members but never join a super-node. In
`collapse.py` they pass through as singleton super-nodes, so the
hyperedge bookkeeping needs no special case.

I went back and forth on dropping claim and cited_work too. I kept them
because they're still text about a specific scientific thing, not metadata
about the corpus. It's a judgment call.

## 2. What counts as "present" in a snapshot

`io.py`, `_node_present_year` and `build_snapshot`.

A node's `first_seen_year` is when the corpus first mentions it.
`origin_year` is when the thing was introduced in the world, and it's
missing for 86% of nodes anyway. Temporal honesty is about what the corpus
had seen by t, so wherever the pipeline asks for a node's date it uses
`first_seen_year`.

Membership, though, follows the edge year. `build_snapshot` keeps every
hyperedge with `year <= cutoff` and pulls in the nodes those edges
reference, so a node first seen after the cutoff can get in. It gets a
quality note and stays. I first wrote that inclusion used first_seen_year
and that temporal honesty held by construction. Both were wrong, and a
review pass caught it by checking the claim against the code instead of
against this file.

How many: 22 concept nodes at 2020, 35 at 2022, 19 at 2024 and none at
2026, under 2% of each snapshot (`n_concept_nodes_first_seen_after_cutoff`
in `t1_snapshot_stats.json`). It does leak into the labels. The 2020 label
for `L1_S00048` ends with "plus one unrelated method, EquiformerV2", and
that node was first seen in 2024.

I left those nodes in and documented the count. Dropping them leaves
hyperedges that reference nodes outside the snapshot, and the collapse
rule would need a policy for half-present edges that I'd rather not invent
in a hurry. It would also change the clustering, so all 248 labels and
everything downstream would have to be redone. So the labeller's input is
temporally honest for about 98% of nodes, not by guarantee. Section 25
counts what reached the labels: 5 of 186 early labels name something
the corpus hadn't seen yet, all of them through this route.

Every node in this export is referenced by at least one edge (the 2026
snapshot holds all 5,798), so no node drops out of a snapshot unnoticed.
On another export that might not hold, and nothing checks for it.

## 3. Structural affinity weighting (weighted clique expansion)

`hypergraph.py`, `build_structural_affinity` and `clique_expansion`.

Counting 1.0 per co-occurring pair badly overweights big hyperedges. An
arity-65 edge gives C(65,2) = 2080 pairs against 1 for an arity-2 edge, so
one big table in one paper would dominate. Instead a hyperedge of arity n
adds 1/(n-1) to each of its pairs. Each member then gets a total of 1 from
every edge it's in, and the edge's total weight is n/2, linear in arity
rather than quadratic. On this data it matters: with unit weights, edges
of arity above 10 hold about 94% of the pair weight at 2026, against about
63% with 1/(n-1) (`t1_describe.py` prints both for every snapshot), and
80% of the hyperedges have arity above 2. `tests/test_hypergraph.py` pins
the weights.

This is a weighted clique expansion, so the clustering runs on a pairwise
projection of the hypergraph. I first credited the weighting to Zhou,
Huang and Schoelkopf (2006), which overstated it. Their normalised
hypergraph Laplacian works out to a clique expansion with w(e)/|e| per
pair, close to mine but not the same, and I use no Laplacian or spectral
step. Agarwal, Branson and Belongie (2006) showed that several hypergraph
Laplacians reduce to clique or star expansions anyway, so the honest claim
is "a projection with arity-aware weights". The parts that really keep
k-ary structure are the T4 collapse (section 9) and the retrieval
enrichment (section 14).

Ruggeri, Lonardi and De Bacco (arXiv:2312.00708) use a third
normalisation: each hyperedge gets total mass 1, spread as 2/(n(n-1)) per
pair. With mine a big table still counts for more than a two-concept edge,
just not quadratically more. I haven't compared the two on this data.

## 4. Naive projection comparison

`hypergraph.py`, `high_arity_weight_share`, printed by
`scripts/pipeline/t1_describe.py`.

The brief asks what a naive pairwise projection would lose. The only thing
I measured is how much of the pair weight comes from high-arity edges with
and without the weighting (the 94% against 63% in section 3). It doesn't
feed into clustering.

That doesn't really answer the question. Both sides are projections, and
the number describes how weight is spread, not what the clustering loses.
A real answer needs a hypergraph-native variant to compare against, scored
on something like how many hyperedges each clustering cuts. I haven't
built that.

## 5. Keeping the semantic signal separate from the coherence-check signal

`embeddings.py`, `encode_semantic` and `encode_lexical_tfidf`.

Checking coherence with the embedding that built the clusters is circular,
because the clusters were built to be similar under exactly that measure.
That's the trap T6 warns about. So clustering uses MPNet
(`encode_semantic`) and the coherence check uses plain TF-IDF
(`encode_lexical_tfidf`), a different family with no pretraining and no
notion of synonymy.

I first wrote that this made the check independent. It doesn't. TF-IDF
reads the same surface forms MPNet does, and on short strings a shared
word is most of what either model sees. Measured
(`scripts/experiments/signal_overlap.py`, `outputs/signal_overlap.json`):
69% of the MPNet k-NN pairs at 2026 share at least one TF-IDF term,
against 7% of random pairs, and a neighbour pair's mean TF-IDF cosine is
0.13 against 0.003 (57% against 8% at 2020). So TF-IDF is a second view of
the same signal, and a clustering built on MPNet neighbours scores well on
it partly by construction. The brief does allow "a different embedding
family", so I don't think the check is worthless, but the coherence
evidence I put first doesn't read through a text representation at all:
held-out hyperedges (section 13) and the blind intruder test (section 21).

## 6. Union k-NN graph, not mutual k-NN

`embeddings.py`, `semantic_knn_graph`.

Each node gets its top-k neighbours by cosine, and the graph keeps an edge
if either side listed the other (union), not only if both did (mutual).
Mutual k-NN graphs are sparser and fall apart into more pieces, and since
this graph feeds `sparse_upgma`, more pieces just mean more forced merges
later (section 8). Union is the permissive choice on purpose.

## 7. Combining structural and semantic signal into one graph

`cluster.py`, `combine_affinities` and `_normalize_affinity`.

This is my answer to P3, how to reconcile structure and meaning. Each
graph is normalised on its own (divided by its 99th percentile and clipped
to [0, 1], so one outlier edge doesn't squash the rest), and they're
combined as `alpha * structural + (1 - alpha) * semantic` into one sparse
graph. I picked this over a two-stage scheme (structure decides the coarse
level, semantics refines inside it) for two reasons. One graph makes alpha
a knob that can be swept. And with two stages, two topically identical
groups that share no hyperedge could never merge at the top. I didn't
build the two-stage version, so that part is argued, not measured.

alpha started at 0.5 as a placeholder. I swept 0.05 to 0.95 in steps of
0.05 (`scripts/experiments/alpha_sweep.py`), scored on coherence against
its null and on both stability measures, the T6 metrics that don't need
labels. 0.3 was the only value that beat 0.5 on every one of those at
every level, so it became the default, and labels, faithfulness and the
extrinsic eval were all redone on the alpha=0.3 clustering. Section 17
has two things I found later: alpha doesn't weight what it seems to, and
one of the sweep's criteria is confounded with it.

The blend is also global: every pair inside a hyperedge gets the same
structural weight whatever its two members mean. Ma et al.
(arXiv:2502.15564, AdE) make the expansion feature-aware, so pairs whose
members have similar features get more weight through a learned kernel.
That's supervised node classification with a GNN, so it doesn't transfer
directly, but an unsupervised version would be easy here: scale each
pair's 1/(n-1) share by the cosine of the two concepts. I haven't tried
it. My guess is it would pull structure toward the semantic graph and
shrink whatever independent information structure carries, which section
13 already says is small.

## 8. Sparse average-linkage clustering and forced merges

`cluster.py`, `sparse_upgma`.

The hierarchy is built by average linkage (UPGMA) directly on the sparse
combined graph, never densified. On a sparse graph some groups share no
edge at all, so merging can run out of edges before it gets down to 12
groups. When that happens the code joins the two smallest remaining groups
and marks the merge `forced`. Those merges have no evidence behind them
and exist only to meet the size budget. On the shipped snapshots it never
happens (0 forced merges out of 5,427 at 2026), but it's tracked rather
than assumed away.

The average-linkage update treats a missing edge as similarity 0 rather
than guessing a value. That's what keeps merge heights non-decreasing (no
inversions), which `fcluster` with `maxclust` needs to produce nested
cuts. Fill missing edges with anything more optimistic and a later merge
can land below an earlier one, which would break the nesting (P1) without
any error.

I checked the implementation against scipy instead of trusting it
(`tests/test_upgma_vs_scipy.py`). On random sparse graphs `sparse_upgma`
gives the same merge heights as scipy's dense average linkage, with a
missing edge as distance 1, to floating-point error, and the same cuts at
k = 2, 5 and 12. On graphs sparse enough to need forced merges the heights
still match and never decrease, and a super-node of weight w behaves like
w identical points, which is what the `sizes` argument is for. The test
fails against weighted average linkage (WPGMA), the easy variant to write
by accident, so it isn't passing trivially.

## 9. Hyperedge collapse rule (T4)

`collapse.py`, `collapse_hyperedge` and `build_coarse_hyperedges`.

When a hyperedge's members are mapped to their super-nodes there are three
cases. If they all land in one super-node, the edge is internal to it, so
it's dropped from the coarse hypergraph and counted in that super-node's
internal stats. If they land in exactly two, it becomes an ordinary
pairwise edge. If they land in three or more, it stays one hyperedge over
those groups instead of being exploded into every pair. That last case is
the one T4 calls out: exploding a k-group edge into C(k,2) pairs inflates
the coarse graph and loses the fact that one original statement asserted
all k together. At level 0 in 2026 the 1,429 edges become 592 coarse
edges (128 pairwise, 464 k-ary) and 7 internal ones (the `summary` under
`coarse_hyperedges_by_level` in hierarchy.json).

What the rule loses: original edges that collapse onto the same set of
super-nodes merge into one coarse edge with a count and a relation-type
breakdown, so you can see how many edges of which kinds contributed, but
not which original edge linked which underlying nodes.

Two limits. First, in the shipped pipeline the collapsed hypergraph is an
output only. `pipeline.py`, `build_hierarchy_json`, writes it into
hierarchy.json, but nothing reads it back, because all three levels are
cuts of one tree built on the fine-level affinity. The brief wants the
rule used by the method. I built that version (`pipeline.py`,
`coarsen_one_level`, and `collapse.py`, `coarse_structural_affinity`),
where each coarser level clusters the super-nodes below it on the
collapsed hypergraph plus centroid embeddings. It was clearly worse on
coherence and didn't ship (section 15), but it's still there as
`coarsening="multilevel"`. Second, articles and authors pass through as
singleton super-nodes, so an edge touching an article can never become
internal. That's why only 7 edges are internal at level 0, and it says
more about the node-type choice in section 1 than about the clustering.

On why the multilevel variant failed, one reading from the partitioning
literature. Multilevel hypergraph partitioners usually coarsen with local
heuristics, and Sajadinia, Aghdaei and Feng (SHyPar, arXiv:2410.10875)
argue that this loses global structure, so they coarsen with hyperedge
effective resistances and flow-based clustering instead. My variant
coarsens on exactly that kind of local signal and ended up with one
level-0 cluster holding about 40% of the nodes. I haven't tested whether
spectral coarsening fixes that, and their goal is minimum-cut
partitioning of circuits, not interpretable groups, so it's a plausible
explanation, not a diagnosis.

## 10. Temporal matching instead of warm-starting

`temporal.py`, the threshold constants, `match_snapshots` and
`classify_events`.

There are two ways to keep the hierarchy stable across snapshots:
warm-start the clustering at t+1 from t's result, or cluster each snapshot
on its own and match afterwards. I went with matching. Each snapshot stays
self-contained and reproducible, and you don't need the previous run to
interpret one. The cost is that stability is only measured after the fact
and nothing pushes the clustering toward it. Asgari, Cazabet and Borgnat
(arXiv:2310.02840), who benchmark dynamic community detection, call this
"No-Smoothing": a static algorithm per snapshot, then Jaccard matching.

Matching is Jaccard overlap between clusters at t and t+1, computed only
on the nodes present in both (`match_snapshots`, `common_ids`). The first
version used the full union, although these notes said otherwise, so every
new node counted against a match, and a cluster that kept all its members
and doubled in size scored at most 0.5. At level 1, 2022 to 2024, that
left 14 of 50 clusters unmatched, against 3 of 50 after the fix, and
deaths over all levels fell from 106 to 43. Only the matching changed: I
checked every snapshot's member sets were identical before and after, and
remapped the label files by member set where persistent ids moved. Grow
and shrink still use full membership, since new members are real growth,
and a cluster made only of new nodes is a birth.

Three thresholds decide how a match is read: STABLE_JACCARD=0.5,
MATCH_THRESHOLD=0.15, SIZE_CHANGE_RATIO=0.2. I picked them by feel. Every
event record keeps its raw Jaccard, so events can be filtered by
confidence later instead of trusting the label. The sweep
(`scripts/experiments/temporal_threshold_sweep.py`, one threshold at a
time on the existing clusterings) says two of them are harmless:
STABLE_JACCARD and SIZE_CHANGE_RATIO only move the stable/grow/shrink
boundary, gradually, and never touch merge, split, birth or death.
MATCH_THRESHOLD is not harmless. From 0.10 to 0.20, pooled over levels,
deaths go from 12 to 78 and splits from 192 to 84, because a higher bar
turns weak but real continuity into death-plus-birth pairs. At 0.05
splits jump to 296 instead. Nothing in the sweep points at a better value,
so 0.15 stays, but the event counts depend on it much more than I assumed.
Asgari et al. use 0.3, and I have no principled way to choose between
theirs and mine.

When a cluster splits, only the piece that was the parent's best match
keeps the persistent id. The other pieces get new ids (births) or whatever
else they matched. I think that's right, since something can't keep two
identities, but one reorganisation then shows up twice in the log: as a
split from the parent's side and as a birth or merge from the piece's.
The split event's `into` field lists every piece's persistent id so the
two can be joined.

**Warm start.** I built and tested it
(`scripts/experiments/warm_start_sweep.py`, rule in section 15). A third
affinity term, `A_prior(i,j) = 1` if i and j shared a level-2 cluster at
the previous snapshot, is blended in as `(1-gamma)*A_task +
gamma*A_prior` before UPGMA, which is Asgari et al.'s "Smoothed-Graph".
Cross-snapshot ARI rises with gamma, a lot at the fine level: level 2 goes
from 0.65 to 0.86-0.92 for gamma of 0.1 and up. Coherence falls, but
noisily. Averaged over the three snapshots a warm start can change (2020
has no prior), level 2 is 7% to 18% lower for gamma of 0.1 and up, and
single snapshot-and-level cells range from 41% lower to 64% higher, with
no CI. I first reported "level 2 loses 17-41%", which was the 2026
snapshot alone. Every gamma that raises ARI at all three levels (0.1, 0.2
and 0.5) loses coherence somewhere: level 0 by 12% at 0.1, level 2 by 17%
at 0.2, levels 1 and 2 by 13% and 18% at 0.5. So under the rule it doesn't
ship, though the rule never put a number on "meaningfully", which is a
weakness of the rule. My reading is that the stability comes from pulling
nodes back toward last snapshot's grouping even where their content moved.
It's still the closest candidate to replace plain matching, and the
obvious next version only trusts the prior where the current snapshot's
own signal roughly agrees with it. On Asgari et al.'s synthetic benchmarks
No-Smoothing was the least smooth option in most settings, which matches.
What a benchmark with planted communities can't show is the coherence
cost on real data.

**Is change localised?** P5 asks that change between snapshots be
localised to where the corpus changed. I tested it
(`scripts/pipeline/localisation.py`, rule in section 15) and it isn't, in
the sense I pre-registered. Per cluster, churn is 1 minus its best Jaccard
at the next snapshot on shared nodes, and exposure is the share of
hyperedges touching its members that are new. At no level does churn rise
with exposure. Pooled Spearman is -0.32 at level 0 (CI -0.63 to 0.02),
+0.03 at level 1 and -0.01 at level 2, and partialling out cluster size
doesn't change that. The clearest way to see it: 351 fine-level clusters
gained under 2% new edges, and their mean churn is 0.44, the same as
everyone else's.

After it failed I looked for another way new material could reach an old
cluster, so this part is post hoc. At alpha=0.3 the clustering is mostly
the semantic k-NN graph, and a new concept can enter an old concept's
neighbour list without sharing any hyperedge with it. About 30% of an old
node's k-NN neighbours at t+1 are new nodes, against 5 to 13% new edges.
Measured that way, churn does follow exposure at the finer levels:
Spearman +0.42 at level 1 (CI 0.27 to 0.55, 0.40 with size partialled
out) and +0.18 at level 2 (CI 0.10 to 0.27), and the most exposed quarter
of level-1 clusters churns 0.67 against 0.41 for the least. Level 0 shows
nothing either way. So there is some localisation, to where the corpus
changed in meaning rather than structure, and since I only found it after
the planned test failed, I treat it as a lead.

Even the least exposed clusters churn 0.36 to 0.41, so much of the change
isn't local on either measure. My guess, untested, is the fixed cut sizes.
Every snapshot is cut at exactly 12, 50 and 200 clusters, and the concept
set grows about fourfold from 2020 to 2026, so when a new region needs its
own cluster, clusters elsewhere have to merge to keep the count. Cutting
at a fixed merge height would avoid that, but then cluster counts would
drift between snapshots, which has its own cost for the report and the
labelling.

## 11. member_ids are raw node ids at every level, not child ids

`pipeline.py`, `build_hierarchy_json`; `validate_hierarchy.py`.

At first `member_ids` pointed at child super-node ids for the coarse
levels, with only the finest level holding node ids. That contradicts the
task's own definition: P0 through PK are all partitions of the same node
set V(t) at different granularity, not a tree of different objects. So
`member_ids` holds concept node ids at every level, and the tree lives in
`parent_id`: a super-node's children are the super-nodes one level down
with `parent_id` equal to its id. `validate_hierarchy.py` checks exactly
that the children sit one level down and split their parent's member_ids
with no gap and no overlap.

## 12. No LLM API for labelling

`labeling.py`; `eval/faithfulness.py`, `held_out_member_ids`.

Labelling needs something that reads a cluster's members and writes a
label and a gloss. An LLM API was the obvious route, but it costs money
outside my Claude subscription, so Claude Code sub-agents did it instead:
they read dumped prompts and wrote labels in the format an API response
would have had. `write_labeling_input` writes out exactly the prompt each
label came from. The catch is that this step isn't a script anyone can
rerun. Section 24 is how a reproducer makes and scores their own set.

What the labeller sees. `labeller_sample_ids` gives it 25 members plus the
node-type counts of all members. The first version took the first 25 ids
in sorted order, and ids start with their type (`cite_`, `claim_`,
`comp_`), so at 2026 the labeller's input was 40% cited works, 26% claims
and 23% components, while techniques and tasks, about 31% of the members,
barely appeared. The prompt also claimed to include the type counts and
didn't. Now the sample is random, seeded from the member list, so it's
reproducible and the faithfulness check can rebuild exactly what the
labeller saw. `sampling="first"` still reproduces the old input.

Relabelling. The first labels were written by sub-agents running inside
this repo, where `questions.csv` and `ground_truth.json` also live. The
labels feed the extrinsic routing, so I couldn't rule out leakage, and I
relabelled with four fresh agents, one per snapshot. Each got only the
path to a batch file in a scratch folder holding nothing else, and was
told to read that file and write one output. No tool was actually
blocked, so afterwards I read their tool logs: two reads of their own
batch and one write each, and no mention of the question files. That's
the strongest isolation I could get without an API. The first labels are
kept in `outputs/labels_v1/`.

Faithfulness had a worse version of the same problem. The NLI premise was
the first 15 sorted members, a strict subset of what the labeller had
seen, so each gloss was checked against its own input, while the code
comment and my report said the opposite. Now the premise is a seeded
sample of up to 15 members the labeller was NOT shown
(`held_out_member_ids`), and clusters with fewer than 5 such members are
skipped and counted instead of graded. That skips 33 of 62 at 2020, where
clusters are small, and 4 of 62 at 2026.

The numbers moved a lot. The first labels got 5-11% contradiction per
snapshot under the circular check and 14-21% against held-out members
(control, the same gloss against a random other cluster: 62-69%). The
current labels get 5-14% (control 45-63%), and not-entailed 60-79%
(control 90-97%). The old numbers are in
`outputs/labels_v1/faithfulness_v1.json`. The two sets were checked
against different held-out samples, so the comparison isn't exact. I read
the improvement as mostly the biased sample going away: a gloss written
from cited works and claims was being tested on techniques and tasks.

The not-entailed rate stays high, and I guessed that's because a list of
15 surface forms rarely entails a summary sentence even when the summary
is fair. The blind rating (section 22) confirmed it: the rater called 32
of the 36 real glosses NLI marks not entailed "accurate", so I no longer
report not-entailed as an over-claim rate. The over-claim rate comes from
the blind rating: 0 of 48 real glosses rated wrong, upper bound 7%.

## 13. Coherence measured with a signal clustering never saw

`eval/coherence.py`; `scripts/pipeline/hypergraph_shuffle_null.py`,
`scripts/pipeline/structural_holdout.py`,
`scripts/experiments/pair_overlap.py`.

Three checks here, from weakest to strongest. The blind intruder test in
section 21 is a fourth.

**TF-IDF against a random-labels null.** Coherence is the mean pairwise
TF-IDF cosine among a cluster's members, weighted by cluster size. The
number means nothing on its own, so it's compared with 30 random
clusterings with the same cluster sizes. At 2026 the real clusters score
4.3, 11 and 28 times the null mean by level, and none of the 30 draws
reaches them. `metrics.json` also has a z-score (709, 864 and 1,424),
but I don't quote it: with 30 draws all you can really say is that the
real value beats every one of them, and a z that size assumes a normal
tail far past anything the draws measured. Coherence rises from about 0.01 at
level 0 to 0.08 at level 2. That's expected, since small groups are easier
to keep lexically tight, and it only means levels shouldn't be compared
with each other, each only with its own null. Section 5 explains why
TF-IDF isn't independent of the clustering signal, which makes this the
weakest check.

**Hypergraph shuffle null.** The random-labels null only controls for
cluster size. So I shuffled the 2026 hypergraph by swapping members
between hyperedges in a way that exactly keeps each concept's hyperedge
degree and each edge's concept arity (checked on every shuffle, not
assumed), rebuilt the structural affinity on it, kept the real embeddings,
and clustered and scored the same way. The z-scores collapse to 0.30,
0.44 and 2.91. The real clustering's coherence (0.01292, 0.03418,
0.08333) is nearly the same as the shuffled hypergraph's (0.01279,
0.03395, 0.08225). So the clusters are lexically non-random, and two nulls
agree on that, but by this measure the semantic side does almost all of
the work. Section 17 explains why: at alpha=0.3 structure carries about
7% of the affinity mass.

**Held-out hyperedges.** The shuffle null says structure doesn't make
clusters more lexically coherent. It can't say whether structure carries
information of its own, so I test that directly (`heldout_edge_cohesion`):
hide 20% of the edges, cluster on the rest, and measure how often members
of a hidden edge share a cluster, against a random partition with the
same sizes (the lift). Run across alpha, this gives the structure/meaning
trade-off as a curve rather than one setting.

With random edges held out, structure predicts unseen edges well beyond
what meaning alone does. At level 0 the lift goes from 1.95 at alpha=0 to
2.42 at the shipped 0.3 and 3.53 at 0.5, while TF-IDF coherence over its
null goes from 4.25 to 4.14 to 3.95. Level 2 goes from 15 to 16 to 34.
With whole papers held out, most of that gain disappears: level 0 goes
from 1.98 to 2.11 to 2.54, and level 2 stays around 15 until alpha=0.7.
So the structure term mostly encodes co-occurrence inside a paper it has
already seen, and says much less about a paper it hasn't. That's the
honest limit of structure on this corpus.

Under the rule I wrote beforehand (section 15), structure earns its place
at alpha=0.3 only at level 0: the paired gain over alpha=0 is above zero
under both schemes there, under the paper scheme only barely (CI 0.01 to
0.25). At levels 1 and 2 the paper-level gain is indistinguishable from
zero. No alpha beats 0.3 on both axes. The curve does make 0.5 look like
a fair alternative, since under the paper scheme it gains held-out lift
at level 0 for almost no TF-IDF cost, but alpha was chosen on stability
and coherence and changing it means relabelling, so I left it. Structure
alone (alpha=1.0) isn't usable: the concept-only structural graph is too
sparse, about 1,100 merges are forced, and everything ends up in one
cluster.

**Why structure generalises weakly.** Ruggeri, Lonardi and De Bacco
(arXiv:2312.00708) find hypergraph communities easier to recover when
hyperedges overlap heavily on the same pairs of nodes. Here they rarely do
(`outputs/pair_overlap.json`). At 2026, 2.1% of concept pairs appear in
more than one hyperedge and 1.7% in more than one paper, which is 3.1% of
the structural weight. 2020 is sparser still (0.2% of pairs, 0.6% of the
weight). Almost every structural edge is one paper's claim, made once.
That fits the held-out result: a hidden edge's members are usually still
connected through that paper's other edges, so random-edge holdout leaves
a path between them, and holding out the whole paper removes it. Their
result is about a generative model, not this corpus, so I read it as a
consistent explanation, not a proof. It does suggest structure would
matter more on a bigger slice of the TKH, where the same concepts get
related by more than one paper.

## 14. Extrinsic evaluation: retrieval, drill-down and routing

`eval/extrinsic.py`, `build_retrieval_texts`, `hierarchy_drilldown`,
`routing_pool_recall`, `leave_one_out_select` and `paired_comparison`;
`scripts/pipeline/t6_extrinsic.py`, `scripts/experiments/label_routing.py`,
`scripts/experiments/rerank_sweep.py`.

**Retrieval text for short names.** The first drill-down run embedded each
candidate by its bare surface form, as clustering does, and got zero hits
on every question at k=20. That looked like a bug, so I checked one case.
The cosine between a question and "MACE", one of its answers, was 0.03,
while an unrelated "problem" node with a long descriptive surface form
scored 0.79. A four-letter acronym gives a sentence model almost nothing
to match against a paragraph-long question. So for surface forms of 20
characters or less, the retrieval text is the surface form plus the forms
it most often shares a hyperedge with ("MACE" pulls in ACE, AFLOW, Allegro
and so on), and the same pair went from 0.03 to 0.56. Longer forms stay
bare. Enriching everything was also far too slow on CPU: over 80 seconds
per 100 nodes with 40 context terms, because padding follows the longest
text in a batch. Enriching only the ~1,228 short ones, with 10 terms and a
64-token cap, takes about 4 minutes for the whole pool. Even so, recall is
modest. The questions are jargon-dense, and I report that rather than
tune until the number looks better.

**Drill-down against flat.** Flat ranks all 3,104 method-like nodes by
cosine to the question. Drill-down routes through label and gloss text,
the top b0 of the 12 level-0 groups and then the top b1 of their level-1
children, and ranks the nodes in the pool that leaves. Ranking a subset
with the same scoring function flat uses can at best tie flat, which is
why every branching value I tried, (3, 3) first, then (5, 5), then (8, 8)
after alpha moved, only ever approached flat's recall from below. So the
ranker also blends in the candidate's level-1 ancestor score,
`final = (1 - beta) * node_score + beta * ancestor_score`, which is
information flat doesn't have. beta=0.6 came from a sweep
(`rerank_sweep.py`).

Those settings were picked on the same 14 questions they were scored on,
and on the answer key from before the section 16 fix.
`outputs/rerank_sweep.json` keeps those numbers as the record of what the
choice was made on and wasn't regenerated. Per question, the whole "beats
flat" gain at the time came from one question (Q14). So the comparison I
report is leave-one-out: for each question, the setting with the best
recall on the others is picked from the full grid (five branching values
times eleven betas) and scored on the held-out one. The headline is the
paired difference against flat, with a bootstrap CI and a sign-flip test.

On the 12 questions that still have ground truth after the matcher fix,
leave-one-out drill-down gets 3.2% recall@20 against flat's 11.9%, a
paired difference of -8.7 points (CI -25.9 to +0.7, p = 0.37): 1 win, 3
losses, 8 ties. The chosen setting jumps around depending on which
question is held out, so the grid search was mostly fitting noise. The
shipped setting scored in-sample gets 0.121 against flat's 0.119, +0.2
points (CI -1.2 to +1.5). So at 12 questions there's no detectable
difference on recall@20, and the honest point estimate leans toward flat.

**Routing.** Where the hierarchy does show up is routing
(`routing_pool_recall`): the share of each question's ground-truth nodes
that survives into the routed pool, against a random pool of the same
size, with a bootstrap CI over questions. It's the main extrinsic metric
in metrics.json (`extrinsic.routing`). With the current labels,
label+gloss routing at (8, 8) keeps 58% of ground-truth nodes in 24% of
the candidates, lift 0.33, CI 0.04 to 0.54, and every budget from (2, 2)
up has a CI above zero. Routing on member-centroid embeddings gives no
lift at any budget (19% in a 12% pool at (8, 8), lift 0.06, CI -0.04 to
0.17), so the signal comes from the labels, not the grouping.

The first labels, written by agents that could see the question files,
score lift 0.48 (CI 0.34 to 0.57). `label_routing.py TAG LABELS_DIR`
applies an archived label set in memory, so both sets are scored under
the same matcher without touching the shipped hierarchy. The first set
still looks better, the CIs overlap, and I can't separate leakage from
wording.

The matcher fix cost margin, not direction. On the broken answer key the
clean labels scored lift 0.42 (CI 0.23 to 0.57). Now the (8, 8) CI nearly
touches zero, and the narrow budgets (3, 3) and (4, 4) carry the firmer
evidence. My read of the whole picture: hierarchy plus labels finds the
right region, and the per-node ranker inside it is the weak part.

## 15. Decision rules written before each follow-up experiment

The scripts named below; `t6_evaluate.py` and `blind_eval.py` for the last
three rules.

After the first draft I ran a series of follow-up experiments. For each
one I wrote the keep-or-discard rule down before running it, so I couldn't
pick the rule after seeing which result looked good. The first few were
written in a private working file and are copied here as written; the
later ones went straight into this file. The general rule: a change
replaces the shipped default only if it clearly improves the metric it's
about. A wash or a trade-off gets reported and the default stays. Each
rule is followed by its result in brief, and the section named has the
detail.

**Alpha** (`alpha_sweep.py`). Replace 0.5 only if some alpha beats it on
both mean coherence z-score (averaged over 3 levels and 4 snapshots) and
mean stability ARI (perturbation and cross-snapshot, averaged over
levels). *Result:* several values passed. I then added a stricter bar,
beating 0.5 on each of the nine per-level numbers separately, and 0.3 was
the only value that passed, so 0.3 shipped. The stricter bar came after
seeing the results, so it's a robustness check, not part of the rule.
Section 17 re-reads the sweep without the confounded perturbation measure.

**Level-0 size skew** (`level0_skew_check.py`). Track each level-0 cluster
through the same 5-seed perturbation, fit a stability-against-size trend
on the 11 non-largest clusters, and see where the largest falls. Clearly
below the trend means the big cluster is the problem and I'd try a
balance constraint. On or above means no special skew effect. *Result:*
0.8 standard deviations below, size against stability r = 0.058. Not
"clearly below", so nothing changed.

**Temporal thresholds** (`temporal_threshold_sweep.py`). Not a keep or
discard call. Sweep each threshold with the other two fixed and see
whether the event mix near the shipped value sits on a shallow or a steep
part of the curve. *Result:* STABLE_JACCARD and SIZE_CHANGE_RATIO shallow,
MATCH_THRESHOLD steep (section 10).

**Warm start** (`warm_start_sweep.py`). Promote only if cross-snapshot ARI
improves at every level and coherence against its null doesn't
meaningfully drop at any level. *Result:* ARI rises at every level for
gamma 0.1, 0.2 and 0.5, and each of those loses 12% to 18% of coherence at
some level, so not promoted. Section 10 has the corrected size of the
drop, which is smaller and noisier than I first reported.

**Hypergraph shuffle null** (`hypergraph_shuffle_null.py`). A robustness
check; nothing ships either way. First verify that degree and arity
sequences are exactly preserved. If z stays in the hundreds, the structure
carries real signal. If it collapses to single digits, report that as a
weakening of the coherence claim. *Result:* it collapsed (section 13).

**Rerank** (`rerank_sweep.py`). Keep the blend only if some beta gets mean
recall strictly above flat's 0.03246 with no more than the (8, 8) pool's
774 candidates. *Result:* passed at beta=0.6, on the old answer key.
Section 14 explains why that pass meant much less than it looked.

**Multilevel coarsening** (`coarsening_compare.py`). The alternative
builds level 2 exactly as now, collapses the hyperedges onto the level-2
super-nodes with the T4 rule, clusters those super-nodes on the collapsed
hypergraph plus centroid embeddings to get level 1, and repeats to get
level 0. Level 2 is the same in both, so only levels 0 and 1 are
compared. The brief requires the T4 rule to be used by the method, so I
lean toward shipping this. It replaces the single tree unless it's
clearly worse at level 0 or 1, averaged over the four snapshots, on any
of: TF-IDF coherence more than 10% below the tree's, mean perturbation ARI
below the tree's 95% CI lower bound, or mean cross-snapshot ARI more than
0.05 lower. Label-free routing recall gets reported but isn't part of the
rule, because the questions are what I'd tune on later and I don't want
them deciding the method.

**Multilevel, second attempt** (written after the first failed, before
running this one). The first variant failed: 40% lower coherence at level
0, worse on all three checks at level 1, and one level-0 cluster holding
39% of nodes. My guess was that coarse structural weight is a sum over
collapsed edges, so big super-nodes attract more weight and snowball. The
second variant divides the coarse weight between S and T by |S||T| and
runs average linkage weighted by member counts, the super-node version of
average linkage on the node graph. Same rule and thresholds. Since this
try was picked after seeing a failure, a pass has to be clear on all three
checks, not marginal, and if it fails too I stop and keep the tree.

*Result, both attempts:* both failed, and the pipeline stays on the single
tree. Averaged over the four snapshots, level-0 TF-IDF coherence was
0.0159 for the tree, 0.0096 for the first multilevel variant and 0.0107
for the size-normalised one. Level 1 was 0.0453, 0.0373 and 0.0368. Both
variants also had lower perturbation ARI at level 1 (0.58 and 0.61
against 0.78). The one place multilevel did better is cross-snapshot ARI
at level 0 (0.43 and 0.51 against 0.35), so it buys temporal stability at
the top, at the cost of coherence, the same trade the warm start made.
Size normalisation didn't touch the imbalance (one level-0 cluster held
42% of nodes, against 15% for the tree), so my guess at the cause was
wrong. My next guess is the semantic side: the centroid of a big mixed
super-node sits near the average of everything, so it looks similar to
every other big mixed super-node and they keep merging. Untested. Numbers
in `outputs/coarsening_compare.json`, whose label-free routing numbers
predate the section 16 matcher fix and weren't regenerated. The rule
doesn't use them.

**Relabelling** (`label_routing.py`, written before any new labels
existed). The first labels have two problems: the labeller saw the first
25 members by sorted id, which is biased by type, and the sub-agents that
wrote them could see the question files. The relabel uses a seeded random
25-member sample plus the full type breakdown, written by fresh agents
with no repo access and no knowledge of the questions. If label routing
with the new labels still clearly beats centroid routing (bootstrap CI on
the lift over chance above zero at the shipped (8, 8) budget), the labels
carry real information and the earlier routing result wasn't just
leakage. If the lift falls to within the centroid-routing CI, I treat the
earlier result as unexplained and possibly leaked, and the report says
so. For faithfulness, the new labels replace the old ones whatever the
rates turn out to be, because the old ones were written from a biased
sample. Both sets of rates get reported. No settings get retuned on the
new labels. *Result* (rerun after the section 16 fix): met. Lift 0.33 (CI
0.04 to 0.54) against centroid routing's 0.06 (CI -0.04 to 0.17). The new
labels replace the old ones.

**Extrinsic re-evaluation** (`t6_extrinsic.py`). The drill-down settings
had so far been picked on the same questions they're scored on. Now
they're picked by leave-one-out: for each question, the setting with the
best mean recall@20 on the others (ties go to fewer candidates, then
lower beta), scored on the held-out one. The grid is the five branching
values already swept times beta 0.0 to 1.0 in steps of 0.1. The headline
is the paired per-question difference between leave-one-out drill-down
and flat, with a bootstrap 95% CI and a sign-flip test. Drill-down beats
flat only if the CI excludes zero; otherwise the report says there's no
detectable difference, whatever the point estimate. Routing goes into
metrics.json as the main extrinsic metric, for label and centroid routing
at every budget, and counts as beating chance at a budget only if its CI
excludes zero. *Result* (rerun after the section 16 fix): -8.7 points, CI
-25.9 to +0.7, p = 0.37, so no detectable difference at 12 questions.
Label routing beats chance at every budget, centroid routing at none.

**Structural held-out coherence** (`structural_holdout.py`). Hide 20% of
the hyperedges that have at least two concept members, cluster on the
rest, and measure how often members of a hidden edge land in the same
cluster, divided by what a random partition with the same sizes gives
(the lift). 2026 snapshot, alpha in {0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}, 5
seeds, with TF-IDF coherence measured on the same clusterings so each
alpha gets a point on both axes. Two schemes: edge-level hides random
edges, paper-level hides every edge from a random 20% of papers, which is
stricter because edges from one paper are correlated. Structure earns its
place at a level if held-out lift at alpha=0.3 beats alpha=0 under both
schemes, with the 95% CI of the paired per-seed difference above zero. If
alpha=0.3 is dominated (another alpha better on both axes at every level,
CIs clear), I report it but don't change alpha here, since that would mean
relabelling. *Result:* level 0 only (edge +0.47, CI 0.31 to 0.64; paper
+0.13, CI 0.01 to 0.25). No gain at levels 1 and 2 under the paper
scheme. 0.3 isn't dominated and stays (section 13).

**Localisation of change** (`localisation.py`). For each cluster C at
snapshot t, at every level and every transition, churn is 1 minus the
best Jaccard between C and any cluster at t+1, on nodes present in both,
so a cluster that only gained members has churn 0. Exposure is the share
of hyperedges touching C's members at t+1 that are new at t+1. This uses
only the written hierarchies and the raw edges, so it doesn't depend on
the section 10 thresholds. Clusters under 3 members are skipped, since
their Jaccard only takes a few values. Change counts as localised at a
level if the Spearman correlation between exposure and churn, pooled over
the three transitions, is positive with a 95% bootstrap CI (resampling
clusters) above zero, and stays positive with cluster size partialled
out, since big clusters could get both more new edges and more churn. As
a descriptive second check, report mean churn for the quarter of clusters
with the least and the most exposure. If it fails, report that change
isn't localised and don't try to fix it here. *Result:* fails at every
level. Spearman -0.32 (CI -0.63 to 0.02), +0.03 (CI -0.14 to 0.19) and
-0.01 (CI -0.09 to 0.07), none positive with size partialled out. Least
and most exposed quarters churn 0.73 and 0.54 at level 0, 0.55 and 0.57
at level 1, 0.44 and 0.44 at level 2. The exploratory semantic-exposure
measure in section 10 came afterwards and doesn't change the verdict.

The last three rules were written in the review pass, before any rating
or provenance data existed (sections 21 and 22 describe the checks).

**Blind intruder test** (coherence). A rater with no repo access sees six
terms per item, five from one super-node and one type-matched node from a
different level-0 branch, and picks the odd one out. Chance is 1/6. 60
real items on the 2026 snapshot (all 12 level-0 super-nodes, 24 at level
1, 24 at level 2) and 20 null items where the five are random nodes.
Clusters are coherent to a blind reader at a level if the Wilson 95% lower
bound on detection is above 1/6. The test only counts as clean if
detection on the null items is NOT clearly above chance (Wilson lower
bound at or below 1/6). If the nulls are detected well above chance, the
rater is using surface cues and the real rate gets reported as
confounded. *Result:* clean, 1 of 20 nulls (5%). Real items 35 of 59
(one level-2 item couldn't be built): level 2 17 of 23 (74%, CI 54% to
88%), level 1 14 of 24 (58%, CI 39% to 76%), level 0 4 of 12 (33%, CI 14%
to 61%). Levels 1 and 2 pass, level 0 doesn't.

**Blind gloss rating** (faithfulness). A second rater with no repo access
rates 72 items, 12 real and 6 control per snapshot, as accurate, vague or
wrong, seeing the gloss and up to 15 members the labeller never saw. The
real "wrong" share becomes the headline over-claim rate if the rater
separates real from control (real "wrong" rate below control's, with
non-overlapping Wilson CIs). The NLI judge counts as validated for
over-claim only if the kappa between NLI contradiction and rater "wrong"
has a bootstrap CI above zero. And if the rater calls more than half of
the real glosses NLI marks not entailed "accurate", I stop reporting
not-entailed as an over-claim rate and say it measures the premise.
*Result:* 0 of 48 real glosses wrong (CI 0% to 7%) against 23 of 24
controls, so 0 of 48 is the headline. Kappa 0.46 (CI 0.24 to 0.68), so
NLI passes as a usable but noisy signal: it flags 6 real glosses as
contradicted and the rater calls none of them wrong. The rater calls 32
of the 36 not-entailed real glosses accurate (89%), which trips the third
rule.

**Provenance faithfulness** (`check_provenance_faithfulness`, run by
`t6_evaluate.py faithfulness`). The same NLI judge, with the titles of the
papers the held-out members came from as the premise, which neither the
labeller nor the clustering ever saw. Glosses count as consistent with
independent evidence if, pooled over the four snapshots, the real
contradiction rate is below the control rate with non-overlapping Wilson
CIs. If not, report that titles are too coarse a premise and don't lean
on the result. *Result:* 21% (39 of 186, CI 16% to 27%) against 31% (58
of 186, CI 25% to 38%). Right direction, overlapping intervals, so I
don't lean on it.

## 16. What counts as a ground-truth node for the extrinsic eval

`eval/extrinsic.py`, `match_ground_truth_methods`.

`ground_truth.json` gives each question a list of expected method names as
text, and the eval scores node ids, so the names have to be matched to
corpus nodes. Exact surface-form match handles most of them. For the rest
I allow a substring match in either direction, which catches "HamGNN"
inside the node "Universal HamGNN Hamiltonian model", and "DeepH" for the
expected "xDeepH".

The first version only required the expected name to be at least four
characters, and said nothing about the node's surface form. In a
materials corpus that's a disaster, because the node table contains
chemical elements. "N", "P", "S", "C" and "Si" are all substrings of
"physics-informed", so that term matched seven element nodes, and
"Hessian training" matched He, N, S and Si. Across the 14 type-A
questions, 47 of 147 ground-truth node ids were surface forms of two
characters or less: about a third of the answer key was chemical
elements. I found this in a review pass, not from the metric, which is the
uncomfortable part, since the numbers looked plausible the whole time. The
report had also said about a third of the expected names "don't match any
corpus node", when every one matched something, mostly junk.

The fix is one condition: both sides of a substring match must be at
least `min_substring_len` (4) characters. `tests/test_ground_truth_match.py`
pins it, including that "MACE-F" no longer matches the node "ACE" and
that an unmatchable term reports "none" instead of quietly matching an
element. Now, of 63 expected-method mentions (50 distinct names), 47 match
exactly, 7 by substring and 9 not at all, and the 9 really are
descriptive categories like "hybrid frameworks" and "GNN free energies".
Two questions (Q5 and Q11) are left with no ground-truth node and drop
out, so the extrinsic numbers are over 12 questions, not 14. I haven't
checked the 7 substring matches against the papers.

Routing, the main extrinsic metric, is weaker than before the fix
(section 14). Every conclusion kept its direction, which is luck as much
as anything.

## 17. What alpha actually weights

`cluster.py`, `combine_affinities` and `_normalize_affinity`;
`scripts/experiments/affinity_mass_share.py`,
`outputs/affinity_mass_share.json`; `eval/stability.py`,
`perturbation_stability`.

I used to describe alpha=0.3 as "structure gets 30%, semantics 70%". That
reading is wrong, and I only checked it after the shuffle null in section
13 said the structure term barely matters.

Each graph is normalised by its own 99th percentile, which puts both in
[0, 1] but not on the same scale in any useful sense. A semantic k-NN edge
is a cosine between neighbours, so after normalisation its typical value
is around 0.57. A structural entry is a sum of 1/(n-1) shares, and most
pairs co-occur in exactly one hyperedge of middling arity, so its typical
value is around 0.06, ten times lower. Multiplying one by 0.7 and the
other by 0.3 doesn't give a 70/30 split of anything. At 2026 the
structural term carries 6.6% of the total affinity mass at alpha=0.3 and
about 14% at 0.5, and the other snapshots land between 5% and 7% at 0.3.
So the shipped method is semantic k-NN clustering with the hypergraph as a
tie-breaker, and the shuffle null's collapse follows directly.

That doesn't make structure idle. Structural pairs outnumber semantic ones
at 2026 (65,679 against 58,856), and 94% of them are pairs the k-NN graph
never proposes, so structure mostly adds edges rather than re-weighting
existing ones. A small weight on an edge that would otherwise be absent
isn't the same as a small weight on one already there, and it's probably
why the held-out test in section 13 still finds something at level 0.

I'm leaving the parametrisation alone for this submission, because
changing it means re-sweeping alpha and relabelling, but it's a defect in
how the method is presented, not a subtlety. The clean fix is to normalise
the two graphs so a unit of alpha means a unit of influence, for instance
by rank or quantile transform, or at least to quote the realised mass
share next to alpha every time. What I don't think this justifies is
quietly raising alpha so structure "counts more", since section 13 already
measured what structure buys on held-out edges: level 0 only.

**How alpha was chosen, and what that does to the numbers.** alpha=0.3 was
picked by maximising the same coherence and stability numbers the report
then presents as results, so for the shipped setting those numbers are
in-sample. They still separate the method from its nulls, which is a
different question, but they aren't an independent estimate of how good
0.3 is.

One of the three criteria is also confounded with alpha itself. The
perturbation test removes 10% of hyperedges and leaves the semantic k-NN
graph untouched, so it only ever disturbs the structural term. A
clustering that leans less on structure is steadier under it
mechanically: perturbation ARI is 0.75, 0.92 and 0.98 by level at
alpha=0.05, against 0.58, 0.77 and 0.91 at 0.3 and 0.47, 0.66 and 0.81 at
0.5. As robustness that's close to meaningless, since it rewards ignoring
the hypergraph. So I re-read the sweep without it, on the six per-level
numbers left (coherence z-score and cross-snapshot ARI). alpha=0.3 still
beats 0.5 on all six and is still the only value that does, with 0.05,
0.15, 0.25, 0.4 and 0.45 at five of six. That's the version of the choice
I'd defend, with thin margins and no uncertainty on the z-scores. The fix
for the test itself is a semantic-side perturbation (drop or re-embed a
sample of nodes) so both signals are under the same stress. Untried.

## 18. Re-applying the pre-registered rules after the matcher fix

The ground-truth fix in section 16 landed after the rules in section 15
had been written and read. That's exactly where it would be easy to cheat
without noticing, so to be explicit: I didn't touch any rule. I reran the
experiments the fix could affect, the relabelling rule and the extrinsic
re-evaluation rule, both of which score against ground truth, and re-read
the same rules against the new numbers. Both still pass, less comfortably,
and section 14 keeps the old numbers next to the new ones. The rules that
don't involve ground truth (alpha, level-0 skew, temporal thresholds, warm
start, shuffle null, multilevel coarsening, structural holdout,
localisation) don't depend on the matcher and weren't rerun.

The risk I can't rule out: if the fix had flipped a rule from pass to
fail, I'd like to think I'd have reported the fail, but I didn't have to,
so that's a claim about myself, not evidence.

## 19. Reproducibility details

`embeddings.py` and `eval/faithfulness.py`, the pinned model revisions;
`labeling.py`, `apply_labels_to_hierarchy` and `unlabelled_super_nodes`;
`scripts/pipeline/t5_apply_labels.py`, `t6_evaluate.py` and
`t6_extrinsic.py`.

Both Hugging Face models load at a fixed commit (the revision hashes in
the two modules) rather than whatever the Hub serves on the day. The
cached copies these results came from are those commits, so pinning
changed nothing here. It stops a later upload from quietly changing the
embeddings, and with them every cluster, for someone rerunning this next
year.

`run_pipeline.py` writes every label as null, since it has no business
guessing which label goes with a freshly built cluster, so
`t5_apply_labels.py` has to run next. Applying labels is guarded: a label
goes onto a super-node only if the prompt rebuilt from that super-node's
current members is byte-identical to the prompt the label was written
from. Persistent ids are reused across reruns, so without the guard a
change to alpha or the thresholds would put old labels on new clusters
with no error. The guard accepts all 248 shipped labels and rejects two
clusters whose members are swapped under the same ids
(`tests/test_label_apply.py`). The two steps that read labels,
`t6_evaluate.py` (faithfulness) and `t6_extrinsic.py`, check for
unlabelled super-nodes up front and say which script to run. Before that,
following the README literally ended in a KeyError deep in the
faithfulness step.

On Linux the PyPI wheel for `torch==2.14.0` pulls in the CUDA 13 toolkit
and nvidia packages that `requirements.txt` doesn't pin, so the CPU wheel
from the PyTorch index is installed first there (`reproduce_all.py` does
this). I checked the dependency metadata and the index listing, but nobody
has run the Linux install end to end.

## 20. Confidence intervals on cross-snapshot stability

`eval/stability.py`, `cross_snapshot_stability` and `_half_sample_ari`.

The original CI was a t-interval over three numbers, one per transition,
with two degrees of freedom, and it answered the wrong question. The three
transitions aren't draws from one distribution: 2022 to 2024 is where the
edge count nearly doubles, and a t-interval treats that real difference
as noise.

The interval I report now holds each clustering fixed and asks how much
the ARI depends on which concepts happen to be in the corpus. It
resamples the shared nodes within each transition, and the interval for
the mean over transitions is built from the same resamples. It doesn't
cover variability of the clustering procedure itself; that's what the
perturbation measure is for.

I got the resampling wrong on the first try and worked through why with
the coding agent. An ordinary bootstrap resamples with replacement, and a
node drawn twice sits in the same cluster as its copy in both partitions,
so it adds an agreeing pair. I'd assumed that was negligible against the
total number of pairs. But ARI is driven by same-cluster pairs, not all
pairs, and at level 2 (200 clusters, averaging 7 members at 2020 and 27 at
2026) the duplicate pairs are a sizeable share of those. The level-2
interval came out as 0.651 to 0.682 around an estimate of 0.647, entirely
above it. Resampling half the nodes without replacement has no
duplicates. Its mean matches the full-data ARI to within 0.001 in the four
cases I checked, and for smooth statistics it has roughly the variance of
the ordinary bootstrap. `tests/test_stability_ci.py` includes the
many-small-clusters case, which the with-replacement version fails.

Results. The means are 0.35, 0.45 and 0.65 by level, with intervals of
0.34 to 0.37, 0.44 to 0.47 and 0.63 to 0.66, much narrower than the
t-intervals were. The more interesting part is per transition. At level
1, 2020 to 2022 is 0.52 (0.49 to 0.56), 2022 to 2024 is 0.37 (0.35 to
0.40) and 2024 to 2026 is 0.47 (0.45 to 0.49). Those don't overlap, so the
middle transition really is less stable, not just noisier, and it's the
one where the corpus changed most. The per-transition numbers are the
honest unit, and the mean over three is a summary, not an estimate of
anything.

## 21. Blind intruder test for coherence

`eval/blind.py`, `make_intruder_items` and `score_intruder`;
`scripts/pipeline/blind_eval.py`; `outputs/blind_eval/`. Rule and full
result in section 15.

The brief lists blind LLM or human judgement as one way to measure
coherence independently, and TF-IDF turned out less independent than I'd
claimed (section 5), so I added the intruder test from the topic-model
literature (Chang et al. 2009). A rater sees five members of a super-node
plus one node from a different level-0 branch, shuffled, and picks the odd
one out. If the groups mean something, the intruder stands out. Chance is
1 in 6.

Two design choices carry the weight. The intruder has the same type as
one of the shown members, so a cluster of short technique names can't give
away a long claim by its length or shape. And a quarter of the items (20
of 79) are nulls, five random nodes plus an intruder picked the same way.
A rater using surface cues rather than meaning would beat chance on those
too.

The rater saw only item ids and six strings. The answer key stayed in a
separate folder until the ratings were in. The rater was a fresh Claude
Code subagent on Opus, a different model from the Sonnet agents that
wrote the labels, though the same family, so they may share blind spots.
Its transcript shows one read of its own item file,
then its answer. It's one LLM rater, with no second rater and no human
pass, so there's no inter-rater agreement to report. That's the obvious
next step if this mattered more.

The result: 74% at level 2 and 58% at level 1, both clearly above chance,
and 5% on the nulls, so the rater wasn't reading surface cues. Level 0 got
4 of 12, which isn't distinguishable from chance, and with only 12
level-0 super-nodes there's no bigger sample to be had on this snapshot.
That fits everything else about level 0: it's the least stable level, and
its labels are the ones that read as mixed topics. This is the first
coherence result in the project that doesn't pass through the text
representation the clustering used, and it says the finer levels hold
together for a blind reader and the coarsest may not.

## 22. Blind gloss rating, provenance check and CIs on faithfulness

`eval/blind.py`, `make_gloss_items` and `score_gloss_ratings`;
`eval/faithfulness.py`, `premise_member_forms` and
`check_provenance_faithfulness`; `eval/stats.py`. Rules and full results
in section 15.

Three additions to the faithfulness evaluation, all from the same worry:
the NLI judge had never been checked against anything.

Wilson intervals. Every faithfulness rate has one now
(`real_contradiction_ci95` and so on in metrics.json). They're wide
because the samples are small: 29 checkable glosses at 2020 means a 14%
contradiction rate comes with an interval of about 6% to 31%.

Blind rating. 72 items, 12 real and 6 control per snapshot. A real item
shows a gloss plus up to 15 of its super-node's members that the labeller
never saw, and a control shows the same gloss with another super-node's
members. The rater marks each accurate, vague or wrong. The members come
from the same function the NLI premise uses (`premise_member_forms`), and
NLI is run on exactly those members, so the rater and NLI grade identical
inputs. Isolation and rater are as in section 21.

This changed the headline. The rater rated 0 of 48 real glosses wrong and
23 of 24 controls wrong, and called 32 of the 36 real glosses NLI marks
not entailed accurate. So the old 60-79% not-entailed figure was measuring
the premise, and the honest over-claim rate is 0 of 48 with an upper
bound of 7%. NLI contradiction does track the rater (kappa 0.46), but it
raises false alarms on real glosses, so I read the 5-14% NLI contradiction
rate as a noisy upper bound, not an over-claim rate. Two caveats: one LLM
rater is not a panel, and "wrong" is a high bar, since a vague gloss is
never wrong, which is partly why real items score so cleanly. The five
real glosses rated vague are the weak spots to look at.

Provenance. This is the brief's circularity again. The labeller's only
input was surface forms, and the clusters were built from surface forms,
so grading glosses against more surface forms only goes so far. Every
node records which papers it came from, and `collection10_articles.csv`
has their titles, which neither the labeller nor the clustering ever saw.
So each gloss is also graded against the titles of its held-out members'
source papers, with the same NLI judge and the same random-other-cluster
control. Real glosses are contradicted less often than controls, 21%
against 31%, but the intervals overlap, so under the rule I set I read it
as titles being too coarse to grade a one-sentence gloss, not as evidence
either way, and keep it in the metrics without leaning on it.

## 23. Publishing model outputs so a rerun can be checked

`model_outputs.py`; `embeddings.py`, `encode_semantic`;
`eval/faithfulness.py`, `nli_labels`; `scripts/reproduce_all.py`,
`scripts/release/export_release.py`, `scripts/verify_release.py`.

Nothing in this project is trained, so there are no weights to publish.
The method is two frozen models pinned to exact Hub commits (section 19)
plus deterministic code, so a rerun on another machine can only differ in
one place: the numbers the two models hand back.

I checked how fragile those are first. On this machine, with the same
model and the same library versions, the same text doesn't always embed
to the same bits. Embedding nodes snapshot by snapshot, 514 of the 5,428
vectors differ from a single pass over all of them, by at most 1.1e-7, and
one text embedded alone differs from the same text inside a batch by
7.5e-8. It's float rounding in batched matrix products. Here it doesn't
change any cluster (the zero-perturbation rebuild in the stability check
gives ARI 1.0), but a different CPU or maths library will move the
embeddings a little, and a near-tie in someone's neighbour lists could
then change a cluster and every number downstream of it.

So the release publishes every model output my run produced, and
`TKH_MODEL_OUTPUTS=replay:DIR` makes the code use those instead of the
models. That separates the two possible causes of a difference. If the
replay reproduces my outputs, my results follow from my model outputs and
the code is fine, and any difference in the reproducer's own run comes
from their model outputs. `verify_release.py compare-embeddings` then
shows how far theirs are from mine and whether any nearest neighbour
changed. Outputs are keyed by the whole call (model revision, texts in
order, batch size, sequence length), not by text, because of the batch
effect above: a text-keyed store couldn't reproduce both passes that embed
the same node.

Building it turned up two problems. Loading a pinned model still sent a
request to the Hub every time, and when the network dropped mid-run each
load sat through five retries before using the cache, so both models now
load from the local cache first and only go online if files are missing.
And `localisation.json` wasn't reproducible. Its per-cluster semantic
exposure was a mean over a Python set of node ids, and set order depends
on string-hash randomisation, so the floats were summed in a different
order each run and differed in the 16th digit. The summary statistics are
rank-based and never moved, which is why every earlier check of
`metrics.json` passed. The checksum over all 45 output files caught it.
`localisation.py`, `cluster_rows`, now sums in sorted order, and the file
is byte-identical under two different hash seeds.

The checksums cover `outputs/` except `outputs/figures/`: matplotlib's
output bytes change between versions, and the figure PDFs aren't in the
repo at all (`export_release.py` leaves the whole folder out).

What the release doesn't cover. The LLM-written labels and the blind
ratings aren't pipeline model outputs. They're checked-in data that no one
can regenerate bit for bit (section 24 is how to make your own). And
replay only checks the code after the models. Whether my model outputs are
what the pinned models really produce is what `compare-embeddings` is for,
run on the reproducer's machine.

## 24. Label and rating sets: the shipped samples, or your own

`labeling.py`, `validate_template`, `label_set_template`,
`write_labeller_request` and `apply_labels_to_hierarchy`; `replies.py`,
`extract_json`; `eval/blind.py`, `rating_reply_problems` and
`glosses_not_applied`; `scripts/pipeline/t5_dump_labeling_input.py`,
`t5_import_labels.py`, `t5_apply_labels.py`, `blind_eval.py` and
`reproduce_all.py`.

The labels and the blind ratings are the two parts no one can rerun bit
for bit, because an LLM wrote them. Section 23 handles the models inside
the pipeline, and this handles the ones outside it. My labels and ratings
are checked in and a plain run uses them. A reproducer can also make
their own set with any LLM or by hand, with their own prompt if they like,
and run the whole evaluation on it with two flags. My numbers come from
one labelling run, two counting the replaced first set, so I don't know
how much they move between labellers, and someone else's set is the most
direct test of that I can offer.

A set is a directory. A label set holds `<year>/labeling_output.json`
and, if it was written from a custom prompt, `prompt_template.txt`. A
rating set holds the packets, their keys and the ratings. The shipped sets
already had that shape (`outputs/snapshots` and `outputs/blind_eval`), so
nothing about them changed, and I checked that: the default prompts, the
rebuilt rating packets (with a fresh NLI run) and every output file are
byte-identical to the published release.

Three design choices. First, the staleness guard from section 19 compares
each label's stored prompt with the prompt rebuilt from the cluster's
current members, which only works if the rebuild uses the template the set
was written from. So the set carries its template and
`t5_apply_labels.py --labels DIR` uses it. The test shows the same labels
apply cleanly with their own template and come up stale against the
default one, so a custom prompt doesn't weaken the guard.

Second, the template can be changed but the number of members the
labeller sees can't. The faithfulness check holds out exactly the members
the labeller didn't see (`held_out_member_ids`), and a different sample
size would have to be passed through to it. If it weren't, the held-out
set would silently overlap the labeller's input, which is the circularity
from section 12. So it's fixed at 25 rather than a flag.

Third, both imports are all or nothing, and scoring checks the ratings
belong to the labels in use. Testing the workflow found a real bug here: a
rating import that failed on one reply still rewrote the other reply's
file and overwrote the rater description. Now nothing is written unless
every reply checks out. `blind_eval.py score` refuses a rating set whose
glosses aren't among the labels currently applied, because gloss ratings
are about specific glosses and scoring them against different labels
would give numbers that mean nothing.

`--replay` doesn't work with a custom label set. The recorded NLI outputs
only cover my glosses, and replay stops with an error on the first call it
has no recording for rather than guess.

I tested the workflow end to end with a stand-in labeller and stand-in
raters that answer every item in the required format, one reply wrapped in
a code fence the way chat models often send it. That tests the plumbing
(custom template, 248 labels imported and applied, packets rebuilt for
them, ratings imported and scored, both refusals), not the quality of
anyone's labels.

## 25. Auditing the labels for temporal honesty

`scripts/experiments/temporal_honesty_audit.py`, `main`.

Section 2 says the labeller's input is temporally honest for about 98% of
nodes, and backs that with one example I happened to notice. That's a
count of inputs, not of outputs, and one example isn't a rate. So I
audited every level-0 and level-1 label at 2020, 2022 and 2024, 186 in
all. 2026 can't fail, since nothing in the export is first seen after it.

A label is flagged if its label or gloss names, as a whole word and
ignoring case, the surface form of something the corpus first saw after
the snapshot year. I check two sources. The first is the labeller's own
input: sampled members whose `first_seen_year` is after t. The second is
the whole export: any surface form whose earliest `first_seen_year`, over
every node that carries it, is after t. That one catches a labeller
bringing in a name from its own knowledge that the corpus only meets
later. Surface forms under four characters are skipped, for the same
reason as the matcher fix in section 16. Every flag gets read by hand
and marked a real leak (the label says something about a thing the corpus
hadn't seen yet) or a false alarm (a generic word that happens to match).

I wrote the rule down before running it. If real leaks are at most 2% of
the 186 labels, I report the measured count in place of the single
example and change nothing else. If there are more, the report says
temporal honesty fails measurably, and the fix (drop future-seen members
from the labeller's sample and relabel) becomes the first thing on the
four-week list. Either way I also report how many labeller inputs held a
future-seen node at all, so the leak rate can be read against the
exposure.

What the audit can't see is a paraphrase: a gloss that describes a later
idea without naming it. It's a floor on the leak rate, not the rate.

*Result* (`outputs/temporal_honesty_audit.json`). 94 of the 186 labels
were flagged, on 154 matches. 76 of those matches were backed by a
sampled member the corpus had already seen, a longer surface form
containing the term, so the corpus did know the thing by t. The other
78 were read one by one, and the verdicts are listed in the script
(`REAL_LEAKS`). Nearly all are field vocabulary whose standalone node happens
to be dated late ("materials" is first a node of its own in 2026), or
names the corpus had seen earlier under another spelling (MPNN from 2017,
ACE, LightGBM as "Light Gradient-Boosting Machine").

Five labels are real leaks, 2.7% (Wilson 95% interval 1.2% to 6.1%):
DeePMD-kit and OC20 at 2020, the EquiformerV2 one I already knew about,
OC20/OC22 at 2022, and Togo Database with the CSP Blind Test at 2022.
None are at 2024. That's over the 2% I set, so by my own rule temporal
honesty fails measurably, and I say so in the report. The fix goes first
on the four-week list.

All five come from the same place: a node first seen after t that sat in
the labeller's sample, e.g. "DeePMD-kit v2" (first seen 2023) behind the
2020 label's "DeePMD-kit". I found no case of the labeller bringing a
later name in from its own knowledge, though the check can't see
paraphrase. 32 labeller inputs held a future-seen node, and 5 of those
labels leaked it (16%, 7% to 32%). So the problem is the snapshot
membership from section 2, not the labeller disobeying the prompt, and
dropping future-seen members from the labeller's sample would fix it
without touching the clustering, though every early label would have to
be written again. The prompt did tell the labeller not to use anything
unknown by the snapshot year, but it had no way to know which members
were from later.

The whole-export source is noisy. I kept it because it's the only thing
that found three of the five. The "labeller input" source only matches
a future node's whole surface form, and "DeePMD-kit v2" never appears
whole in a gloss.
