# Design notes

This file holds the reasoning behind choices in the code that aren't
obvious from reading the code itself. Each section names the file (and
usually the function) it's about. Comments in the code point back here
instead of repeating the reasoning inline, so the source stays readable
without needing the full justification inline every time.

## 1. Which node types get clustered

`io.py`, `CONCEPT_TYPES` and `build_snapshot`.

The graph has 12 node types but two of them, article and author, aren't
really "concepts" the way the task means it. An author isn't a scientific
idea, it's a person who wrote something, and an article is a container
for ideas, not an idea itself. So only the other ten types (method,
technique, task, problem, dataset, metric, component, cited_work,
future_topic, claim) go into `concept_ids` and get clustered. Articles and
authors still exist in the snapshot and still show up as hyperedge
members, they just never become a super-node member on their own. In
collapse.py they get treated as singleton "super-nodes" so the hyperedge
math still works without a special case.

I went back and forth on whether to drop claim and cited_work too, since
they're a bit different from method/task/dataset. Decided to keep them in
because they're still text describing a specific scientific thing, not
metadata about the corpus itself. Could be wrong about this, it's a
judgment call more than a fact.

## 2. What counts as "present" in a snapshot

`io.py`, `_node_present_year` and `build_snapshot`.

A node's `first_seen_year` is when the corpus first mentions it, which is
not the same as `origin_year` (when the method/dataset/whatever was
actually introduced in the world). For a snapshot at year t we want to
know what the corpus had SEEN by t, not what existed by t, because the
whole point of the temporal-honesty requirement is that a label shouldn't
claim knowledge the corpus didn't have yet. So where this pipeline asks
about a node's date at all, it asks `first_seen_year` and never
`origin_year`.

What actually decides membership, though, is the edge year. `build_snapshot`
keeps every hyperedge with `year <= cutoff` and then pulls in whatever
nodes those edges reference. `first_seen_year` is only checked afterwards,
and a node whose first_seen_year is later than the cutoff gets a quality
note and is kept anyway. An earlier version of this section said inclusion
"uses first_seen_year", and `report.md` claimed temporal honesty for the
labeller's input held by construction. Both were wrong, and I only caught
it when a reviewer pass checked the claim against the code rather than
against this file.

The size of the problem: 22 concept nodes at 2020, 35 at 2022, 19 at 2024
and none at 2026, so under 2% of each snapshot
(`n_concept_nodes_first_seen_after_cutoff` in `t1_snapshot_stats.json`,
from `describe_snapshot`). Small, but not zero, and it does leak into the
labels. The 2020 label for `L1_S00048` ends with "plus one unrelated
method, EquiformerV2", and that node's first_seen_year is 2024. So a 2020
super-node really does name a method the corpus hadn't met in 2020.

I left the nodes in and documented the count rather than filtering them
out. Two reasons, one good and one honest. The good one is that dropping
them means a hyperedge can reference a node that isn't in the snapshot,
and then the collapse rule and the arity bookkeeping need a policy for
half-present edges, which is a real design question I'd rather not answer
badly in a hurry. The honest one is that filtering changes the clustering,
which means the 248 labels and every label-dependent number would have to
be redone, and I'd rather ship a documented 2% than a rushed rebuild. It's
a real limitation of P6 as shipped, not a technicality: temporal honesty
of the labeller's input is empirical here, at about 98%, not guaranteed.

A node only makes it into a snapshot if some included hyperedge
references it. I didn't add a separate check for "orphan" nodes with no
edges at all, because in this dataset every node turned out to be
referenced by at least one edge (checked this by comparing the snapshot's
total node count against the full 5798 at cutoff 2026, they match
exactly). If this pipeline were ever run on a different export where that
isn't true, a node could disappear from a snapshot without any warning,
which would be worth fixing then.

## 3. Structural affinity weighting (weighted clique expansion)

`hypergraph.py`, `build_structural_affinity`, the `share` weight formula.

For two concept nodes that co-occur in the same hyperedge, how much
"structural affinity" should that count for? The obvious naive answer is
1.0 per co-occurring pair, but that badly overweights big hyperedges: an
arity-65 hyperedge produces C(65,2) = 2080 pairs, versus 1 pair for an
arity-2 edge, so one big table in one paper would dominate the whole
structural signal. Instead each hyperedge of arity n contributes
1/(n-1) to each of its C(n,2) pairs. Each member then gets a total of 1
from every hyperedge it's in, split evenly over its co-members, and the
hyperedge's total weight is n/2, linear in arity instead of quadratic.

I want to be straight about what this is. It's a weighted clique
expansion, so the clustering runs on a pairwise projection of the
hypergraph, not on the hypergraph itself. An earlier version of these
notes said this weighting "comes from" Zhou, Huang and Schoelkopf (2006).
That overstated it. Their normalized hypergraph Laplacian works out to a
clique expansion with weight w(e)/|e| per pair, which is close to mine but
not the same, and I don't use a Laplacian or any spectral step at all.
Agarwal, Branson and Belongie (2006) showed that several hypergraph
Laplacians reduce to clique or star expansions anyway, so the honest claim
is "a projection with arity-aware weights", not "hypergraph-native". The
parts of the pipeline that really do keep k-ary structure are the T4
collapse (section 9) and the retrieval enrichment (section 14).

Ruggeri, Lonardi and De Bacco (arXiv:2312.00708), in their analysis of how
much information a hypergraph carries beyond its clique expansion, use a
third normalisation: each hyperedge gets total mass 1, spread as
2/(n(n-1)) per pair. Mine gives each hyperedge mass n/2, so a big table
still counts for more than a two-concept edge, just not quadratically more.
I haven't compared the two on this data. Their main result turned out to
be more useful to me than the weighting (section 13).

I checked this actually matters on the real data (see section 4 below):
under the naive unweighted version, edges of arity > 10 end up holding
about 94% of total pairwise weight, versus about 63% under the 1/(n-1)
weighting. So this isn't just a theoretical nicety for this corpus, since
80% of the hyperedges here have arity > 2.

## 4. Naive projection comparison

`hypergraph.py`, `projection_loss_report`.

The assignment specifically asks to quantify what a naive pairwise
projection would lose compared to a hypergraph-native method, so
`projection_loss_report` builds both versions (weighted=True and
weighted=False in `build_structural_affinity`) and compares how much of
the total pairwise "mass" comes from high-arity edges under each. This
function doesn't feed into clustering at all, it only exists to produce
that comparison number for the report.

This doesn't really answer the question the brief asks. Both sides are
projections (section 3), and the number describes how weight is spread,
not what the clustering loses. A real answer needs a hypergraph-native
variant to compare against, scored on something like how many hyperedges
each clustering cuts. I haven't built that yet.

## 5. Keeping the semantic signal separate from the coherence-check signal

`embeddings.py`, the split between `encode_semantic` and
`encode_lexical_tfidf`.

If clustering uses embedding X and then we check "coherence" by measuring
similarity under the same embedding X, high coherence is basically
guaranteed and proves nothing, since the clusters were built to be
similar under exactly that measure. This is the circularity problem T6
warns about. To avoid it, `encode_semantic` (the MPNet embeddings that
drive clustering) and `encode_lexical_tfidf` (plain bag-of-words TF-IDF,
used only for the coherence check) need to come from different
signal families, not just two different neural embedding models that
were both trained the same way. TF-IDF has no notion of synonymy or
pretraining at all, so it can't just be rediscovering what MPNet already
encoded.

That last sentence overstated it. TF-IDF has no pretraining, but it reads
the same surface_form text MPNet does, and on short strings a shared word
is most of what either model sees. Measured (`scripts/signal_overlap.py`,
`outputs/signal_overlap.json`), 69% of the MPNet k-NN pairs at 2026 share
at least one TF-IDF term against 7% of random pairs, and a neighbour
pair's mean TF-IDF cosine is 0.13 against 0.003 (57% against 8% at 2020).
So TF-IDF is a different representation of the same signal rather than an
independent one, and a clustering built on MPNet neighbours scores well on
it partly by construction. The brief does allow "a different embedding
family", so the check isn't worthless, but the coherence evidence I'd now
put first doesn't read through a text representation at all: held-out
hyperedges (section 13) and a blind intruder test (section 21).

## 6. Union k-NN graph, not mutual k-NN

`embeddings.py`, `semantic_knn_graph`.

Building the semantic graph, I compute each node's top-k neighbors by
cosine similarity, then symmetrize by keeping an edge if EITHER side
listed the other in its top-k (union), not requiring BOTH sides to list
each other (mutual/intersection). Mutual k-NN graphs are usually sparser
and can fragment into more disconnected pieces, and since this semantic
graph gets unioned with the structural graph and then fed into
`sparse_upgma`, more disconnection just means more forced merges (see
section 8) later, which is the thing I'm trying to avoid. Union k-NN is
the more permissive choice on purpose.

## 7. Combining structural and semantic signal into one graph

`cluster.py`, `combine_affinities` and `_normalize_affinity`.

This is the actual answer to the "how do you reconcile structure and
meaning" question (P3) that the task treats as the central design
decision. The two affinity graphs get normalized separately (each one's
values divided by its own 99th percentile and clipped to [0,1], so one
outlier edge doesn't compress everything else toward zero) and then
combined as `alpha * structural + (1-alpha) * semantic` into one sparse
graph. I picked this over a two-stage approach (structure decides the
coarse level, semantics only refines within it) mainly because a single
combined graph lets alpha be a literal, sweepable knob that shows up in
the eval section, rather than a qualitative statement about which stage
matters more.

Alpha started at 0.5 as a placeholder. I swept it from 0.05 to 0.95 in
steps of 0.05 (`scripts/alpha_sweep.py`), scored by coherence-vs-null and
both stability measures. Those are the T6 metrics that don't need labels,
so the sweep didn't need 19 rounds of relabelling. alpha=0.3 was the only
value that beat 0.5 on every one of those metrics at every level, so it's
the shipped default now. Labels, faithfulness and the extrinsic eval were
all redone against the alpha=0.3 clustering afterwards.

One thing the sweep can't tell me is whether the structural term matters.
The shuffle null in section 13 says that at alpha=0.3 it barely does,
at least by lexical coherence.

The blend is also global. Every pair inside a hyperedge gets the same
structural weight whatever its two members mean. Ma et al. (arXiv:2502.15564,
AdE) make the expansion itself feature-aware: pairs inside a hyperedge
whose members have similar features get a larger weight, through a learned
distance kernel. Their setting is supervised node classification with a
GNN on top, so it doesn't transfer directly, but an unsupervised version
would be easy here: scale each pair's 1/(n-1) share by the cosine
similarity of the two concepts. I haven't tried it. My guess is it would
pull the structural graph toward the semantic one and shrink whatever
independent information structure still carries, which section 13 already
says is small, so I don't think it's an obvious improvement.

## 8. Sparse average-linkage clustering and forced merges

`cluster.py`, `sparse_upgma`.

The hierarchy is built by average-linkage (UPGMA) directly on the sparse
combined graph, not by converting to a dense matrix first. Doing this on
a sparse graph means some pairs of nodes never share any edge at all
(no shared hyperedge, not in each other's k-NN), so eventually the merge
process can run out of real edges before reaching the target number of
top-level clusters (12). When that happens, the code force-merges the two
smallest remaining groups anyway, and marks that merge `forced=True`.
These forced merges have zero evidence behind them, they exist purely
because the report needs to show at most ~15 top-level groups no matter
what. On the actual 2026 snapshot this never happened (0 forced merges
out of 5427), but it's the kind of thing that could show up on a smaller
snapshot, so it's tracked rather than assumed away.

The Lance-Williams update inside the merge loop treats a missing edge as
similarity 0 rather than trying to guess a value. I checked that this
specific choice is the one that keeps the dendrogram's merge distances
non-decreasing (no "inversions"), which matters because `fcluster` with
criterion=maxclust assumes that property to produce properly nested cuts.
If a missing edge got filled with something more optimistic than 0, a
later merge could end up with a smaller distance than an earlier one,
which would break the nesting guarantee (P1) silently.

I also checked the implementation against scipy instead of trusting it
(`tests/test_upgma_vs_scipy.py`). On random sparse graphs, `sparse_upgma`
gives the same merge heights as scipy's dense average linkage, where a
missing edge is distance 1, down to floating-point error, and the same
cuts at k = 2, 5 and 12. On graphs sparse enough to need forced merges
the heights still match and never decrease, and a super-node with weight
w behaves like w identical copies of one point, which is what the
`sizes` argument is for. The test does fail against weighted average
linkage (WPGMA), the easy variant to implement by accident, so it isn't
passing trivially.

## 9. Hyperedge collapse rule (T4)

`collapse.py`, `collapse_hyperedge` for the rule,
`clique_explosion_comparison` for the comparison number.

When a hyperedge's endpoints land in super-nodes after clustering, there
are three cases. If all endpoints land in the same super-node, the edge
is now entirely internal to that one group, so it gets dropped from the
coarse hypergraph and just counted as an internal-density number instead
(it's not a relationship BETWEEN groups anymore). If endpoints land in
exactly two groups, it becomes an ordinary pairwise edge between them. If
endpoints land in three or more groups, it stays a genuine hyperedge
between those groups, rather than getting expanded into every pairwise
combination between them. That last part is the one the task calls out
specifically (T4), because exploding a k-group hyperedge into C(k,2)
pairwise edges instead of keeping one k-ary edge would inflate the
apparent density of the coarse graph and lose the fact that all k groups
were asserted together by one original fact. `clique_explosion_comparison`
computes exactly how much bigger the pairwise version would have been, so
that's not just an assertion, there's a number behind it once the
pipeline runs.

What this rule does lose: once several original hyperedges collapse onto
the same set of super-nodes, they get merged into one coarse edge with a
weight count and a relation-type breakdown, so you can see how many
edges and what kinds contributed, but not which original edge asserted
which specific pairing of underlying nodes.

Two limits I need to state. First, in the shipped pipeline the collapsed
hypergraph is an output only. `pipeline.py`, `build_hierarchy_json`,
writes it into hierarchy.json but nothing reads it back, because all
three levels are cuts of one dendrogram built on the fine-level
affinity. The brief wants the rule to be used by the method. I built that
version (`pipeline.py`, `coarsen_one_level`, and
`collapse.py`, `coarse_structural_affinity`): each coarser level clusters
the super-nodes below it on the T4-collapsed hypergraph plus centroid
embeddings. It was clearly worse on coherence and didn't ship (section
15). It's still in the code as `coarsening="multilevel"`. Second, articles and
authors pass through as their own singleton super-nodes, so any edge
touching an article can never become fully internal. That's why only 7 of
1429 edges end up internal at level 0. The coarse hypergraph is mostly
article-centred edges, which says more about the node-type choice in
section 1 than about the clustering.

On why the multilevel variant failed, one reading from the partitioning
literature. Multilevel hypergraph partitioners usually coarsen with local
heuristics like vertex or hyperedge similarity, and Sajadinia, Aghdaei and
Feng (SHyPar, arXiv:2410.10875) argue that this loses the hypergraph's
global structure. They replace it with coarsening driven by hyperedge
effective resistances and flow-based local clustering. My multilevel
variant coarsens on exactly that kind of local signal (collapsed edge
weights plus centroid similarity), and it ended up with one level-0
cluster holding about 40% of the nodes. I haven't tested whether spectral
coarsening would fix that, and their goal is minimum-cut partitioning of
circuit netlists, not interpretable groups, so I'd call it a plausible
explanation, not a diagnosis.

## 10. Temporal matching instead of warm-starting

`temporal.py`, whole file, especially the threshold constants near the
top and `classify_events`.

Two ways to make the hierarchy stable across snapshots: warm-start the
clustering at t+1 using t's result as a prior, or cluster each snapshot
independently and match the results afterward. I went with matching,
mainly because it's simpler to reason about and keeps each snapshot's
result fully self-contained and reproducible on its own, you don't need
the previous snapshot's clustering run to interpret any given snapshot.
The real cost is that this only MEASURES stability after the fact, it
doesn't actively push the clustering toward being stable while it runs.
A warm-started or regularized version would probably produce a smoother
hierarchy at the cost of being harder to verify independently.

Update: built and tested that version (`scripts/warm_start_sweep.py`,
decision rule in section 15). Added a
third affinity term, `A_prior(i,j) = 1` iff i and j were in the same
level-2 cluster in the previous snapshot (0 for nodes that didn't exist
yet), blended in as `(1-gamma)*A_task + gamma*A_prior` before UPGMA runs.
Cross-snapshot ARI does rise with gamma, as expected, and by a lot at
the finer levels: level 2 goes from 0.65 to between 0.86 and 0.92 for
gamma of 0.1 or more. Coherence (TF-IDF against a random-labels null)
goes down, but less cleanly than I first wrote. I had said it "drops at
nearly every gamma tried" and that level 2 "loses 17-41%". The 17-41% is
the 2026 snapshot alone. A review pass recomputed it per snapshot. At 2022
and 2024 the level-2 effect is mixed, from a 16% rise to an 11% drop, and
averaged over the three snapshots a warm start can change (2020 has no
prior), level 2 is 7% to 18% lower for gamma of 0.1 or more. Single
snapshot-and-level cells swing from 41% lower to 64% higher, so this is a
noisy measure, and it has no CI.

The verdict under the rule in section 15 still holds, but by less than I
made it sound. Every gamma that raises ARI at all three levels (0.1, 0.2
and 0.5) also loses coherence somewhere: level 0 by 12% at 0.1, level 2 by
17% at 0.2, and levels 1 and 2 by 13% and 18% at 0.5. The rule said
"doesn't meaningfully drop at any level" without putting a number on
"meaningfully", which is a weakness of the rule, not a detail. My reading
is still that the stability is bought by pulling nodes back toward last
snapshot's grouping even where their content has moved, which is what
"measure, don't force" was meant to avoid. But the warm start is the
closest candidate to replacing plain matching, and the next version is
the obvious one: only trust the prior where the current snapshot's own
signal roughly agrees with it. Untried here.

The dynamic community detection literature has names for these options.
Asgari, Cazabet and Borgnat (arXiv:2310.02840) benchmark four approaches.
What I ship is their "No-Smoothing": a static algorithm on each snapshot,
then Jaccard matching of the most similar communities. My warm start is
closest to their "Smoothed-Graph", which blends a same-community-last-time
term into the adjacency matrix with a weight, and that's what `A_prior`
does. On their synthetic benchmarks No-Smoothing was the least smooth in
most settings, which matches the direction my sweep found. What a
benchmark with planted communities can't show is the cost my sweep found
on real data, that the smoother hierarchy was less coherent. Their matching
threshold is 0.3, against my 0.15. I don't have a principled way to pick
either, and the sweep below shows the event counts move a lot with it.

The matching itself is Jaccard overlap between a snapshot's clusters and
the next one's, computed only on node ids that exist in both snapshots
(`match_snapshots`, the `common_ids` argument). An earlier version of
these notes said that too, but the code actually used the full union, so
every new node at t+1 counted against the match. A cluster that kept all
its old members and doubled with new ones scored at most 0.5 when it
should score 1.0. I measured it before fixing it: at level 1, 2022 to
2024, 14 of 50 clusters had no match above MATCH_THRESHOLD under the old
code and 3 of 50 under the shared-node version. After the fix, death
events at the shipped thresholds went from 106 to 43 across all levels.
The clusters themselves didn't change, only the matching, and I checked
that every snapshot's member sets were identical before and after. Some
persistent ids did get reassigned, so I remapped the label files by
member set rather than relabelling. Sizes for grow/shrink still use full
membership, since new members are real growth. A cluster made only of
new nodes now has zero overlap with everything and is logged as a birth,
which is what it is.

Three thresholds control how it's read: STABLE_JACCARD=0.5,
MATCH_THRESHOLD=0.15, SIZE_CHANGE_RATIO=0.2. These are picked by feel,
not fit to anything. In particular MATCH_THRESHOLD=0.15 is fairly
permissive, so a weak 1-1 match (say Jaccard 0.16) still gets labeled as
the SAME entity growing or shrinking rather than flagged as a shaky
match. The raw Jaccard value is saved on every event record, though, so
this can be filtered or re-weighted downstream by confidence instead of
trusting the discrete label blindly.

Update: ran that sensitivity check (`scripts/temporal_threshold_sweep.py`).
Reran `classify_events` over the already-computed hierarchy.json
clusterings with each threshold swept individually (other two held at
shipped), no need to touch embeddings or reclustering since T3 only
rematches existing partitions. The numbers below are from the rerun after
the shared-node Jaccard fix above. STABLE_JACCARD and SIZE_CHANGE_RATIO
turned out fine: sweeping either one only moves the stable/shrink or
stable/grow boundary, gradually and monotonically, and never touches
merge, split, birth, or death counts at all. The picked-by-feel values
weren't doing anything fragile.

MATCH_THRESHOLD is a different story. It's sensitive, not
gradual: going from 0.10 to 0.20 (a small move either side of the shipped
0.15), pooled across all 3 levels, death events go from 12 to 78 and
split from 192 to 84, because raising the bar for "adequate match"
reclassifies a lot of weak-but-real continuity as clean death+birth pairs
instead. At the low end (0.05) splits explode instead (296) because almost
any nonzero overlap counts as a match, so one real cluster can look like
it split against several unrelated small ones. I'd hoped the Jaccard fix
would take most of this sensitivity away, since part of it looked like
new-node dilution pushing real matches under the line. It lowered the
counts but the shape is the same, so the sensitivity is real.
There's no single value here that's obviously more correct than 0.15,
it's picking where to draw a fuzzy line, but the line's position matters a
lot more than I assumed when I picked it by feel. Left the shipped value
alone (this was a characterization pass, not an optimization one, and
nothing in the sweep points at a specific better number), but this is now
a documented real limitation rather than an unchecked worry.

One more thing worth writing down: when a group splits into several
pieces (the split check inside `classify_events`), only the piece that
was the parent's best match keeps the original persistent id, and the
other pieces get new ids (logged as births) or ids from whatever else
they matched. I think that's the right way to think about a split
(something can't keep two identities at once), but it does mean a single
real-world reorganization can show up in the event log from two
different angles: as a "split" from the parent's point of view and
separately as a "birth" or "merge" for the piece that didn't keep the
name. The split event's `into` field lists the persistent ids of every
piece, so the two views can be joined. It used to store local cluster
labels, which mean nothing outside one run.

P5 also asks that change be localised to where the corpus changed. I
tested that (`scripts/localisation.py`, rule in section 15) and it isn't,
at least not in the sense I pre-registered. For each cluster I took its
churn (1 minus its best Jaccard match at the next snapshot, on nodes
present in both) and its exposure (the share of hyperedges touching its
members that are new). At no level does churn rise with exposure. The
pooled Spearman correlation is -0.32 at level 0 (CI -0.63 to 0.02), +0.03
at level 1 and -0.01 at level 2, with CIs around zero, and partialling out
cluster size doesn't change that. The clearest way to see it: 351
fine-level clusters gained under 2% new edges, and their mean churn is
0.44, the same as everyone else's. Structurally untouched regions get
reorganised as much as the ones where new papers landed.

After it failed I looked for another route by which new material could
reach an old cluster, and this part is exploratory, not pre-registered.
At alpha=0.3 the clustering is 70% semantic k-NN graph, and a new concept
can enter an old concept's neighbour list without sharing any hyperedge
with it. On average about 30% of an old node's k-NN neighbours at t+1
are new nodes, against only 5 to 13% new edges. Measured that way, churn
does follow exposure at the two finer levels: Spearman +0.42 at level 1
(CI 0.27 to 0.55, 0.40 with size partialled out) and +0.18 at level 2
(CI 0.10 to 0.27), and the quarter of level-1 clusters with the most
semantic exposure churn 0.67 against 0.41 for the least. Level 0 shows
nothing either way. So there is some localisation, but it's to where the
corpus changed in meaning, not in structure, and I only found it by
looking after the first test failed, so I'd treat it as a lead.

Even the least exposed clusters churn 0.36 to 0.41, so a large part of
the change isn't local on either measure. My guess, untested, is the
fixed cut sizes. Every snapshot is cut at exactly 12, 50 and 200
clusters, and the concept set grows about fourfold from 2020 to 2026, so
when a new region needs its own cluster, some other clusters elsewhere
have to merge to keep the count fixed. Cutting at a fixed merge height
instead of a fixed count would avoid that, but then the number of
clusters per level would drift between snapshots, which has its own cost
for the report and the labelling. I haven't tried it.

## 11. member_ids are raw node ids at every level, not child ids

`pipeline.py`, `build_hierarchy_json`.

Earlier version of this had `member_ids` point at child super-node ids
for the coarse levels (level 0's members were level 1's ids, and so on),
with only the finest level holding actual node ids. Changed this after
noticing it contradicts how the task itself defines the partitions: P0
through PK are all partitions of the same node set V(t), just at
different granularity, not a tree of different kinds of objects. So now
`member_ids` holds raw concept node ids at every level, and the tree
structure lives entirely in `parent_id`. A level's children can always be
found by filtering for `parent_id == this_id` at the next level down.
`validate_hierarchy.py` checks this holds exactly (a super-node's
children partition its member_ids with no gap and no overlap) rather than
just assuming the construction got it right.

## 12. No LLM API for labeling

`labeling.py`, whole file.

Labeling needs something that reads a cluster's member list and writes a
label and gloss. Calling out to an LLM API was the obvious way to do it,
but that costs money that isn't covered by a Claude Pro subscription (API
billing is separate). So I had the coding agent do it instead: Claude Code
sub-agents, one per snapshot, read the dumped prompts and wrote labels
back into the same file format an API response would have produced.
`write_labeling_input` writes out exactly the prompt each label was
written from, so you can see what information went into it. The catch is
that this step isn't a script anyone can rerun, and the first set of
labels was written by sub-agents running inside this repo, where
`questions.csv` and `ground_truth.json` also live. I couldn't rule out
that they'd seen those, and the labels feed the extrinsic routing
(section 14), so I relabelled (below).

The labeller sees `labeller_sample_ids` in `labeling.py`: the first 25
member ids in sorted order. I only noticed later that ids are
type-prefixed (`cite_...`, `claim_...`, `comp_...`), so sorted order is
not a neutral sample. On the 2026 snapshot the labeller's input was 40%
cited works, 26% claims and 23% components, while techniques and tasks,
about 31% of the actual members, barely showed up. Now the default is a
random sample seeded from the member list itself, so it's reproducible
and the faithfulness check can rebuild exactly what the labeller saw. The
prompt also lists the node-type counts of all members, which it had
claimed to include but never did. `sampling="first"` still reproduces the
old input, and the first labels are kept in `outputs/labels_v1/`.

The relabel (`scripts/t5_import_labels.py` merges and checks a reply) was
done by four fresh agents, one per snapshot. Each got only the path to a
batch file in a scratch directory holding nothing else, told to read that
one file and write one output file. No tool was actually blocked, so
afterwards I read their tool logs: each made two reads of its own batch
and one write, and none of the logs mentions the question files. That's
the strongest isolation I could get without an API call.

Faithfulness (`eval/faithfulness.py`) had a worse version of the same
problem. The NLI premise was built from the first 15 sorted members, a
strict subset of what the labeller had seen, so each gloss was being
checked against its own input. The code comment and my report both said
the opposite. Now the premise is a seeded random sample of up to 15
members the labeller was NOT shown (`held_out_member_ids`), and clusters
with fewer than 5 such members are skipped and counted instead of
graded. That skips a lot at 2020 (33 of 62, since early clusters are
small) and only 4 of 62 at 2026.

The numbers moved a lot. For the first labels, the circular version gave
a contradiction rate of 5-11% per snapshot, and against held-out members
it was 14-21% (control, same gloss against a random other cluster:
62-69%), with not-entailed at 84-93% against 95-98% for control. The
relabelled set does clearly better: contradiction 5-14% (control 45-63%)
and not-entailed 60-79% (control 90-97%), so both rates now separate
real from random. The old numbers are in
`outputs/labels_v1/faithfulness_v1.json`. The two sets were checked against
different held-out samples, so the comparison isn't exact. I read the
improvement as mostly the biased sample going away: a gloss written from
cited works and claims was being tested on techniques and tasks. The
not-entailed rate is still high, and I think that's mostly because a list
of 15 surface forms rarely entails a summary sentence even when the
summary is fair. I report both rates.

That last guess turned out to be checkable, and it held. A blind rater
(section 22) called 32 of the 36 real glosses NLI marks not entailed
"accurate", so the not-entailed rate is a property of the premise, and I
no longer report it as an over-claim rate. The over-claim rate now comes
from the blind rating: 0 of 48 real glosses rated wrong, upper bound 7%.

## 13. Coherence measured with a signal clustering never saw

`src/tkh/eval/coherence.py`.

The obvious way to check "are these clusters coherent" is to look at how
similar the members are under whatever signal built the clusters. That's
circular: of course they look similar under that signal, that's what the
clustering optimized for. So the coherence check here uses TF-IDF
(`encode_lexical_tfidf` in embeddings.py) instead of the MPNet embeddings
that actually drove clustering. TF-IDF is plain bag-of-words with no
neural pretraining behind it, so it isn't just a second opinion from a
similar model, it's a different way of representing the text.

Coherence itself is the mean pairwise cosine similarity among a cluster's
members under TF-IDF, weighted by cluster size when averaging across
clusters. A raw coherence number doesn't mean anything on its own though,
so it gets compared against a null model: reshuffle which nodes are in
which cluster at random, keeping the exact same size distribution, and
recompute the same weighted-mean coherence on that random partition, 30
times. Report the observed value against the mean and standard deviation
of those 30 null trials as a z-score. On the 2026 snapshot this came out
extremely high (z in the several-hundreds range at every level), so
whatever else is uncertain about the method, the clusters are not lexically
coherent by chance.

One thing to flag: coherence goes up noticeably from the coarse level to
the fine level (roughly 0.01 at level 0 up to 0.08 at level 2 on the 2026
snapshot). That's expected, smaller groups are easier to keep lexically
tight, so this isn't evidence the coarse level is "worse," just that
coherence at different granularities isn't directly comparable to each
other, only each level's number against its own null.

Update: built the harder null the random-labels one doesn't cover
(`scripts/hypergraph_shuffle_null.py`, result in metrics.json under
`coherence_hypergraph_shuffle_null`). Random-labels only controls for cluster size, it
says nothing about whether the specific hyperedge structure matters versus
some other clustering built from a structurally-similar random hypergraph.
So I shuffled the 2026 hypergraph via bipartite double-edge-swaps that
exactly preserve each concept node's hyperedge-degree and each qualifying
edge's concept-arity (checked programmatically, not assumed), rebuilt
structural affinity on the shuffled hypergraph, kept the real semantic
embeddings unchanged, clustered the same way, and scored that clustering's
coherence the same way. The z-scores collapse hard under this null: 0.30
(level 0), 0.44 (level 1), 2.91 (level 2), against 709/864/1424 under
random-labels. The real clustering's coherence (0.01292, 0.03418, 0.08333)
is nearly identical to a random-structure hypergraph's clustering
(0.01279, 0.03395, 0.08225) at the same degree/arity statistics, run
through the exact same shipped alpha=0.3 pipeline.

That's not the method failing, it's a sharper diagnosis than the first
null could give: at alpha=0.3, semantic gets 70% of the affinity weight,
so a clustering built on the real semantic embeddings looks about equally
TF-IDF-coherent whether the structural 30% comes from the real hypergraph
or a degree/arity-matched random one. The coherence check was always
measuring "is this partition better than chance," and it still says yes
against random-labels, but it can't actually tell you SEMANTIC vs.
STRUCTURE is doing that work, and it turns out it's overwhelmingly
semantic, at least by this lexical measure. Net effect: I
believe the clusters are lexically non-random more confidently than
before (two independent nulls agree on that), but I believe less than I
did that the hypergraph structure specifically, as opposed to the
semantic embeddings, is what's driving that lexical coherence, at least
at the coarse and mid levels. `report.md` section 5.1 carries this,
it's a real qualification, not a footnote to bury.

The shuffle null can only say whether structure makes clusters more
lexically coherent, and it doesn't. It can't say whether structure carries
information of its own. So I tested the structure side directly with
held-out hyperedges (`heldout_edge_cohesion` in `eval/coherence.py`,
`scripts/structural_holdout.py`): hide 20% of the edges, cluster on the
rest, and measure how often members of a hidden edge share a cluster,
against a random partition with the same sizes. Running it across alpha
gives the structure/meaning trade-off as a curve instead of a single knob
setting.

With random edges held out, structure predicts unseen edges well beyond
what meaning alone does. At level 0 the lift goes from 1.95 at alpha=0 to
2.42 at the shipped 0.3 and 3.53 at 0.5, while TF-IDF coherence over its
null goes from 4.25 to 4.14 to 3.95. Level 2 goes from 15 to 16 to 34. With
whole papers held out, most of that gain disappears: level 0 goes from
1.98 to 2.11 to 2.54, and level 2 stays around 15 until alpha=0.7. So the
structural signal is mostly co-occurrence inside a paper. It predicts
another edge from a paper it has already seen, but says much less about
a paper it hasn't. That's the honest limit of the structure term on this
corpus.

Ruggeri, Lonardi and De Bacco (arXiv:2312.00708) give a reason to expect
this. In their model, communities in a hypergraph are easier to recover
when hyperedges overlap heavily on the same pairs of nodes, i.e. when a
pair shows up in several hyperedges. So I counted how often that happens
here (`scripts/pair_overlap.py`, `outputs/pair_overlap.json`). In 2026,
2.1% of concept pairs appear in more than one hyperedge and only 1.7%
appear in more than one paper. Weighted by the 1/(n-1) affinity, that's
3.1% of the structural weight coming from pairs that recur across papers.
2020 is even sparser (0.2% of pairs, 0.6% of the weight). Almost every
structural edge is one paper's claim, made once. That fits the held-out
result. The exact pair rarely repeats even inside one paper, but a hidden
edge's members are usually still tied together through that paper's other
edges, so random-edge holdout leaves a path between them. Holding out the
whole paper removes the path. Their result is about a specific generative model,
not this corpus, so I read it as a consistent explanation, not a proof.
It does suggest the structure term would matter more on a bigger slice of
the TKH, where the same concepts get related by more than one paper.

One caveat on everything TF-IDF in this section: it isn't independent of
the clustering signal (section 5 has the overlap numbers). The two checks
here that don't lean on it are the held-out hyperedge test above and the
blind intruder test in section 21, which is where I'd look first for
whether the clusters mean something.

At the shipped alpha=0.3, structure only clears the bar I set beforehand
at level 0 (paired gain over alpha=0 above zero under both schemes; under
the paper scheme only barely, CI 0.01 to 0.25). At levels 1 and 2 the
paper-level gain is indistinguishable from zero. No alpha beats 0.3 on
both axes. The curve does suggest 0.5 is a reasonable alternative, since
under the paper scheme it gains held-out lift at level 0 for almost no
TF-IDF cost, but the alpha sweep chose 0.3 on stability and coherence, and
changing it means relabelling, so I've left it. Structure alone
(alpha=1.0) isn't a usable point: the concept-only structural graph is too
sparse, about 1,100 of the merges are forced and everything ends up in
one cluster.

## 14. Fixing embedding-based retrieval for short method names

`src/tkh/eval/extrinsic.py`, `build_retrieval_texts`.

First version of the extrinsic drill-down eval embedded each candidate
node by its bare surface_form, same as clustering does, and got zero hits
across every single question at k=20. That's suspicious enough on its own
to be a bug, not a real result, so I checked it directly: the cosine
similarity between a real question and the embedding of "MACE" (a
ground-truth answer for that question) was 0.03, while an unrelated
"problem" node with a long descriptive surface_form scored 0.79 against
the same question, purely because it shared more surface vocabulary with
the question text. A four-character acronym just doesn't give a sentence
embedding model enough to work with against a full paragraph-length
question.

Fix: for short surface forms (20 characters or under, which covers almost
all the bare acronym/proper-noun method names), build the embedding text
from the node's surface_form PLUS the surface forms it most frequently
co-occurs with across its own hyperedges. Concretely, for "MACE" that
pulls in things like "ACE", "AFLOW", "Allegro", other papers and methods
it's actually discussed alongside. Checked this on the same example
before committing to it: cosine similarity went from 0.03 to 0.56.
Already-descriptive surface forms (over 20 characters) are left as their
bare text, they don't need the help and enriching them too turned out to
be prohibitively slow (tested it: encoding all 3104 method-like nodes with
40 context terms each took over 80 seconds per 100 nodes on CPU, mostly
because sequence-length padding in a batch is driven by the longest text
in it, so a handful of long texts drag the whole batch down; restricting
enrichment to just the ~1228 short ones with 10 context terms and a
capped 64-token sequence length brought the whole candidate pool down to
about 4 minutes).

Even after the fix, absolute recall on the extrinsic task is modest, a
handful of questions get real hits at k=20 and most don't. I'm reporting
that as it is rather than tuning parameters until the number looks better,
the honest story is that this is a hard retrieval problem given how
jargon-dense the questions are relative to what a general-purpose sentence
embedding model can resolve, not that the method secretly works great
under different settings.

The drill-down branching factor (b0, b1 in `hierarchy_drilldown`) went
through the same honesty check: the first value I tried (3, 3) cut recall
roughly in half compared to the flat baseline, which would have made the
hierarchy look strictly worse at the one thing it's supposed to help
with. Swept a few wider values instead of reporting that number, and
(5, 5) matched the flat baseline's recall and precision exactly while
still only scoring about 16% of the candidate pool (after the alpha
ablation moved alpha to 0.3, this specific pair stopped matching and got
re-swept to (8, 8); see section 7). Going wider than that
doesn't help further, recall saturates at the flat baseline's level once
the branching is wide enough to not exclude the right answer up front.

Update, beating flat rather than matching it (`scripts/rerank_sweep.py`,
decision rule in section 15): every
branching-factor value above has the same ceiling built in, and I didn't
notice it until I asked myself why nothing had ever beaten flat, only tied
it. `hierarchy_drilldown` only ever ranks a *subset* of flat's candidate
pool, using the exact same scoring function flat uses on the full pool
(cosine similarity to the node's own embedding). A subset ranked by the
same function as the full set can tie the full set at best, it can't
systematically do better, because it has strictly less information, not
different information. Widening the branch just approaches that ceiling
from below, which is exactly the saturation pattern above.

So I gave the ranker something flat doesn't have: each
candidate's level-1 ancestor's label+gloss score, the same score already
used to pick which branches to descend into, now also blended into the
final per-node ranking instead of being thrown away after the branching
decision. `final = (1 - beta) * node_score + beta * ancestor_score`.
Swept beta from 0.0 (current shipped behavior, exact reproduction was the
smoke test) to 1.0 in steps of 0.1, at the shipped (8, 8) pool. beta=0.6
and 0.7 tie for the best result: mean recall 0.0423 against flat's 0.0325
and mean precision 0.0286 against flat's 0.0143, both clearly above flat,
still only scoring the same 774 candidates (25% of the pool) as before.
Checked this wasn't a pool-restriction artifact by rerunning the same beta
sweep against the full unrestricted 3,104-candidate pool: beta=0.6 still
beats flat there too, so the gain is really coming from the extra
information in the blend, not from which nodes the branching happens to
keep. Picked beta=0.6 over beta=1.0 (which scored marginally higher recall
but visibly worse precision and throws away the node's own embedding
signal entirely) and over beta=0.9 (a single-point dip that looks like
noise from averaging over only 14 questions, not a real effect) because it
keeps both signals in play and sits at the start of a plateau
rather than a single lucky point.

Looking at it per question, I overstated this. In 10 of the 14 questions
both flat and drill-down retrieve zero ground-truth nodes. Of the other
four, three tie, and the whole gap comes from one question (Q14, 1 hit
for flat against 5 for drill-down). Beta and the branching factor were also
tuned on these same 14 questions, and there's no confidence interval. So
"beats flat" is one question after tuning on the test set. It needs
held-out tuning and a paired test before I claim it.

After relabelling (section 12) and with the same settings, not retuned,
drill-down got 0.0341 recall against flat's 0.0325. Q14's gain shrank
from 4 extra hits to 1, and one question went flat's way. So drill-down
ties flat on recall@20 at a quarter of the candidates, and that's the
claim I can make. Every number in the three paragraphs above, from the
beta sweep on, predates the ground-truth matcher fix in section 16 and is
kept here as the record of what the decisions were made on. Under the
corrected answer key the same settings give drill-down 0.121 against
flat's 0.119 on 12 questions, which is the same tie read at a different
scale, and the leave-one-out paragraph below is the rerun.

Routing is where the hierarchy shows up (`routing_pool_recall` in
`eval/extrinsic.py`, `scripts/label_routing.py`): the share of each
question's ground-truth nodes that survives into the routed pool, against
a random pool of the same size, with a bootstrap CI over questions.

The numbers below are the rerun after the ground-truth matcher fix in
section 16, on the 12 questions that still have ground truth. Both label
sets were re-scored under the fixed matcher so the comparison is
like-for-like (`label_routing.py TAG LABELS_DIR` applies an archived label
set in memory, so `outputs/labels_v1` can be scored without disturbing the
shipped hierarchy). With the clean relabel, label+gloss routing at (8, 8)
keeps 58% of ground-truth nodes in 24% of candidates, lift 0.33, CI 0.04
to 0.54, and every budget from (2, 2) up has a CI above zero, which meets
the rule I wrote down beforehand (section 15). Routing on member-centroid
embeddings gives no lift at any budget: 19% in a 12% pool at (8, 8), lift
0.06, CI -0.04 to 0.17. So the signal comes from the labels rather than
from the grouping. The first labels, the ones written by agents that could
see the question files, score 73% in 25%, lift 0.48, CI 0.34 to 0.57. They
still look better than the clean set, the CIs still overlap, and I still
can't separate leakage from wording.

What the matcher fix cost here is margin, not direction. Under the broken
ground truth the clean labels scored lift 0.42 with a CI of 0.23 to 0.57;
now the CI at (8, 8) nearly touches zero and the narrow budgets (3, 3) and
(4, 4) carry the firmer evidence. My read of the whole picture is
unchanged: hierarchy plus labels finds the right region, and the per-node
ranker inside it is the weak part.

The drill-down settings had always been picked on the same 14 questions
they were scored on, so I redid the comparison with leave-one-out
(`leave_one_out_select` and `paired_comparison` in `eval/extrinsic.py`,
run from `scripts/t6_patch_extrinsic.py`). For each question, the setting
with the best recall on the other 13 is chosen from the full grid (five
branching values times eleven betas) and scored on the held-out one.
Leave-one-out drill-down gets 3.2% recall against flat's 11.9%, a paired
difference of -8.7 points with a bootstrap CI of -25.9 to +0.7 and a
sign-flip p of 0.37. It wins 1 question, loses 3, and ties 8. The chosen
setting also jumps around depending on which question is held out, which
says the grid search was mostly fitting noise. For comparison, the shipped
setting scored in-sample is +0.2 points, CI -1.2 to +1.5. So at 12
questions there's no detectable difference between drill-down and flat on
recall@20, and the honest point estimate leans toward flat. (These are the
post-fix numbers too. Before the matcher fix the same comparison read
-1.5 points, CI -6.3 to +2.9, p = 0.59, on 14 questions whose answer key
was a third chemical elements. Same verdict, worse answer key.)

Routing is now the main extrinsic metric in metrics.json, under
`extrinsic.routing`. Label routing beats chance at all five budgets, with
every CI above zero, and centroid routing doesn't beat chance at any of
them.

## 15. Decision rules I wrote down before each follow-up experiment

The scripts named here: `alpha_sweep.py`, `level0_skew_check.py`,
`temporal_threshold_sweep.py`, `warm_start_sweep.py`,
`hypergraph_shuffle_null.py`, `rerank_sweep.py`.

After the first draft I ran a series of follow-up experiments. For each
one I wrote the keep-or-discard rule into a private working file before
running it, so I couldn't pick the rule after seeing which result looked
good. That file isn't part of the repo, so the rules are copied here as
they were written. The general rule was that a change only replaces the
shipped default if it clearly improves the metric it's about. A wash or a
trade-off gets reported and the default stays.

Alpha (`alpha_sweep.py`). Replace 0.5 only if some alpha beats it on both
mean coherence z-score (averaged over 3 levels and 4 snapshots) and mean
stability ARI (perturbation and cross-snapshot, averaged over levels).
Several values passed that. After the sweep I also checked a stricter
bar, beating 0.5 on each of the nine per-level numbers separately, and
0.3 was the only one that passed. That stricter bar was added after
seeing results, so it's a robustness check, not part of the
pre-registered rule.

Level-0 size skew (`level0_skew_check.py`). Track each level-0 cluster
through the same 5-seed perturbation, fit a stability-vs-size trend on
the 11 non-largest clusters, and see where the largest falls. Clearly
below the trend means the big cluster is the problem and I'd try a
balance constraint. On or above means no special skew effect. Result: 0.8
standard deviations below, size vs stability r = 0.058, not "clearly
below", so nothing changed.

Temporal thresholds (`temporal_threshold_sweep.py`). Not a keep/discard
call. Sweep each threshold with the other two fixed and check whether the
event mix near the shipped value is on a shallow part of the curve or a
steep one. Result: STABLE_JACCARD and SIZE_CHANGE_RATIO shallow,
MATCH_THRESHOLD steep (section 10).

Warm start (`warm_start_sweep.py`). Promote only if cross-snapshot ARI
improves at every level and coherence-vs-null doesn't meaningfully drop
at any level. Result: ARI up at every level for gamma 0.1, 0.2 and 0.5,
and at each of those a coherence drop of 12% to 18% at some level, so not
promoted (section 10, where the size of that drop is corrected: it is
smaller and noisier than I first reported).

Hypergraph shuffle null (`hypergraph_shuffle_null.py`). A robustness
check, nothing ships either way. Verify that degree and arity sequences
are exactly preserved first. If z stays in the hundreds, the structure
carries real signal. If z collapses to single digits, report it as a
weakening of the coherence claim. Result: it collapsed (section 13).

Rerank (`rerank_sweep.py`). Keep the blend only if some beta gets mean
recall strictly above flat's 0.03246 at no more than the (8, 8) pool's
774 candidates. Result: passed at beta=0.6, but see the end of section 14
on why that pass is weaker than it looks.

Multilevel coarsening (`coarsening_compare.py`, written before running).
The alternative builds level 2 exactly as now, then collapses the
hyperedges onto level-2 super-nodes with the T4 rule and clusters the
super-nodes on that collapsed hypergraph plus centroid embeddings to get
level 1, and repeats from level 1 to get level 0. Level 2 is the same in
both variants, so only levels 0 and 1 are compared. The brief requires the
T4 rule to be used by the method, so I lean toward shipping this. It
replaces the single-dendrogram version unless it's clearly worse at
levels 0 or 1, averaged over the four snapshots, on any of these:
TF-IDF coherence more than 10% below the dendrogram's, mean perturbation
ARI below the dendrogram's 95% CI lower bound, or mean cross-snapshot ARI
more than 0.05 lower. Label-free routing recall on the extrinsic questions
gets reported too, but it isn't part of the rule. The questions are what
I'd tune on later, so I don't want them deciding the method.

Multilevel, second attempt (written after the first attempt failed, before
running this one). The first multilevel variant failed the rule above: 40%
lower coherence at level 0, worse on all three checks at level 1, and one
level-0 cluster holding 39% of nodes. My guess at the cause is that coarse
structural weight is a sum over collapsed edges, so big super-nodes attract
more weight and snowball. The second variant divides the coarse structural
weight between S and T by |S||T| and runs average linkage weighted by
member counts, which makes the coarse step the super-node analogue of
average linkage on the node graph. Same rule, same thresholds. This is a
second try picked after seeing a failure, so a pass needs to be clear on
all three checks, not marginal, and if it fails too I stop and keep the
dendrogram.

Multilevel result (both attempts). Both failed and the shipped pipeline
stays on the single dendrogram. Averaged over the four snapshots, level-0
TF-IDF coherence was 0.0159 for the dendrogram, 0.0096 for the first
multilevel variant and 0.0107 for the size-normalised one. Level 1 was
0.0453, 0.0373 and 0.0368. Both multilevel variants also had lower
perturbation ARI at level 1 (0.58 and 0.61 against 0.78). The one place
multilevel did better is cross-snapshot ARI at level 0 (0.43 and 0.51
against 0.35), so it does buy temporal stability at the top, but at the
cost of coherence, which is the same trade the warm start made. Size
normalisation didn't touch the imbalance (one level-0 cluster held 42% of
nodes, against 15% for the dendrogram), so my guess at the cause was wrong.
My next guess is the semantic side: the centroid of a big mixed super-node
sits near the average of everything, so it looks similar to every other
big mixed super-node and they keep merging. I haven't tested that.
Numbers are in `outputs/coarsening_compare.json`.

Relabelling (`label_routing.py`, written before any new labels existed).
The first labels have two problems: the labeller only saw the first 25
members by sorted id, which is type-biased, and the sub-agents that wrote
them could see the question files. The relabel uses a seeded random
25-member sample plus the full type breakdown in the prompt. It's done by
fresh agents that get the prompts pasted in, with no repo access and no
knowledge of the questions. The first labels are kept in
`outputs/labels_v1/` for comparison. What I'll read from it, decided now:
if label routing with the new labels still clearly beats centroid routing
(bootstrap CI on the lift over chance above zero at the shipped (8, 8)
budget), the labels carry real information and the earlier routing
result wasn't just leakage. If the lift falls to within the centroid-routing
CI, I treat the earlier result as unexplained and possibly leaked, and the
report says so. For faithfulness, the new labels replace the old ones
whatever the rates turn out to be, because the old ones were written from
a biased sample. Both sets of rates get reported. No settings (beta,
branching) get retuned on the new labels in this step.

Relabelling result, as rerun after the section 16 matcher fix. The rule
above is met. Label routing with the clean labels at (8, 8) keeps 58% of
ground-truth nodes in 24% of candidates (lift 0.33, CI 0.04 to 0.54), clear
of centroid routing (lift 0.06, CI -0.04 to 0.17). The first labels scored
0.48 (CI 0.34 to 0.57) under the same matcher. The new labels replace the
old ones. Recall@20 with the new labels and unchanged settings: drill-down
0.121, flat 0.119. (The original reading of this rule, on the broken answer
key, was 65% in 23%, lift 0.42, CI 0.23 to 0.57 against the first labels'
0.55. It passed then and it passes now.)

Extrinsic re-evaluation (`t6_patch_extrinsic.py`, written before running).
The drill-down settings (branching and beta) have so far been picked on the
same 14 questions they're scored on. Now they're picked by leave-one-out:
for each question, choose the setting with the best mean recall@20 on the
other 13 (ties go to fewer candidates, then lower beta), and score it on
the held-out one. The grid is the five branching values already swept
times beta 0.0 to 1.0 in steps of 0.1. The headline number is the paired
difference between leave-one-out drill-down and flat, per question, with a
bootstrap 95% CI and a sign-flip test. I'll say drill-down beats flat only
if the CI excludes zero. Otherwise the report says there's no detectable
difference at 14 questions, whatever the point estimate is. Routing (share
of ground truth kept in the routed pool, against a same-size random pool)
goes in metrics.json as the main extrinsic metric, for label and centroid
routing at every budget, and it counts as beating chance at a budget only
if its CI excludes zero. None of this changes the labels or the hierarchy.

Extrinsic re-evaluation result, rerun after the section 16 matcher fix.
Leave-one-out drill-down against flat: -8.7 points of recall@20, CI -25.9
to +0.7, p = 0.37. The CI includes zero, so per the rule the report says no
detectable difference, now at 12 questions. Label routing beats chance at
every budget and centroid routing at none. (On the broken answer key this
read -1.5 points, CI -6.3 to +2.9, p = 0.59. Same verdict.)

Structural held-out coherence (`structural_holdout.py`, written before
running). TF-IDF coherence only checks the meaning side. This checks the
structure side with edges the clustering never saw: hide 20% of the
hyperedges that have at least two concept members, cluster on the rest,
and measure how often members of a hidden edge land in the same cluster,
divided by what a random partition with the same cluster sizes would give
(the lift). It runs on the 2026 snapshot, for alpha in {0, 0.1, 0.2, 0.3,
0.5, 0.7, 1.0}, with 5 seeds, and TF-IDF coherence is measured on the
same clusterings, so each alpha gets a point on both axes. There are two
holdout schemes. Edge-level hides random edges. Paper-level hides every
edge from a random 20% of papers, which is stricter because edges from
the same paper are correlated. Structure counts as earning its place at a
level if held-out lift at alpha=0.3 beats alpha=0 under both schemes,
with the 95% CI of the paired per-seed difference above zero. If alpha=0.3
turns out to be dominated (some other alpha better on both axes at every
level, CIs clear), I report that but don't change alpha here, since it
would mean relabelling again. That becomes its own decision.

Structural held-out result. At alpha=0.3 the paired held-out lift over
alpha=0 is above zero under both schemes at level 0 only (edge +0.47, CI
0.31 to 0.64; paper +0.13, CI 0.01 to 0.25). At levels 1 and 2 the paper
scheme shows no gain. alpha=0.3 isn't dominated by any other alpha. Alpha
stays at 0.3.

Localisation of change (`localisation.py`, written before running). P5
asks that change between snapshots be localised to where the corpus
changed. For each cluster C at snapshot t (every level, every
transition 2020 to 2022, 2022 to 2024, 2024 to 2026), churn is 1 minus
the best Jaccard between C and any cluster at t+1, computed on nodes
present in both snapshots, so a cluster that only gained new members
has churn 0. Exposure is the share of hyperedges touching C's members at
t+1 that are new at t+1 (not in snapshot t). This uses only the
hierarchies already written and the raw edges, so it doesn't depend on
the event thresholds in section 10. Clusters under 3 members are
skipped, since their Jaccard only takes a few values. Change counts as
localised at a level if the Spearman correlation between exposure and
churn, pooled over the three transitions, is positive with a 95%
bootstrap CI (resampling clusters) above zero, and stays positive when
cluster size is partialled out, since big clusters could plausibly get
both more new edges and more churn. As a second, descriptive check I
report mean churn for the quarter of clusters with the least and most
exposure. If it fails, I report that change isn't localised and don't
try to fix it here.

Localisation result. Fails at every level. Pooled Spearman between
new-edge exposure and churn: level 0 -0.32 (CI -0.63 to 0.02), level 1
+0.03 (CI -0.14 to 0.19), level 2 -0.01 (CI -0.09 to 0.07), and none
turn positive with size partialled out. Mean churn in the least and most
exposed quarters is 0.73 and 0.54 at level 0, 0.55 and 0.57 at level 1,
0.44 and 0.44 at level 2. Change is not localised in the pre-registered
sense. An exploratory measure I added afterwards, the share of new
nodes among an old node's k-NN neighbours, does correlate with churn at
levels 1 and 2 (section 10). That's post hoc, and it doesn't change the
verdict.

Three more rules, written before any of these were run, for the checks
added in the review pass (sections 21 and 22 describe the checks).

Blind intruder test (`blind_eval.py`, coherence). A rater with no repo
access sees six terms per item, five from one super-node and one
type-matched node from a different level-0 branch, and picks the odd one
out. Chance is 1/6. There are 60 real items on the 2026 snapshot (all 12
level-0 super-nodes, 24 at level 1, 24 at level 2) and 20 null items where
the five are random nodes. Clusters count as coherent to a blind reader at
a level if the Wilson 95% lower bound on intruder detection is above 1/6.
The test only counts as clean if detection on the null items is NOT
clearly above chance (Wilson lower bound at or below 1/6); if the nulls
are detected well above chance, the rater is using surface cues and the
real-item rate gets reported as confounded, not as coherence.

Blind gloss rating (`blind_eval.py`, faithfulness). A second rater with no
repo access rates 72 items, 12 real and 6 control per snapshot, as
accurate, vague or wrong, seeing the gloss and up to 15 members the
labeller never saw. The rater's "wrong" share on real glosses becomes the
headline over-claim rate if the rater separates real from control (real
"wrong" rate below control "wrong" rate with non-overlapping Wilson CIs).
The NLI judge counts as validated for over-claim only if the kappa between
NLI contradiction and rater "wrong" has a bootstrap CI above zero. And if
the rater calls more than half of the real glosses that NLI marks not
entailed "accurate", I stop reporting the not-entailed rate as an
over-claim rate and say it measures the premise instead.

Provenance faithfulness (`check_provenance_faithfulness`, in
`t6_evaluate.py faithfulness`). The same NLI judge, but the premise is the
titles of the papers the held-out members were extracted from, which
neither the labeller nor the clustering ever saw. Glosses count as
consistent with independent evidence if, pooled over the four snapshots,
the real contradiction rate is below the control contradiction rate with
non-overlapping Wilson CIs. If not, I report that paper titles are too
coarse a premise to grade a gloss and don't lean on the result.

Blind intruder result. The test is clean: the rater found the intruder
in 1 of 20 null items (5%, CI 1% to 24%), so it wasn't reading type or
length. On real items it found 35 of 59 (59%, CI 47% to 71%) against a
chance rate of 17%. By level that is 17 of 23 at level 2 (74%, CI 54% to
88%) and 14 of 24 at level 1 (58%, CI 39% to 76%), both clearly above
chance, and 4 of 12 at level 0 (33%, CI 14% to 61%), which does not clear
the bar. So levels 1 and 2 are coherent to a blind reader and level 0 is
not shown to be. With only 12 level-0 super-nodes there is no way to get a
bigger sample at that level on this snapshot.

Blind gloss rating result. The rater separates real from control almost
perfectly: 0 of 48 real glosses rated wrong (CI 0% to 7%) against 23 of 24
controls (96%). So the headline over-claim rate is now 0 of 48, upper
bound 7%. NLI contradiction agrees with the rater's "wrong" at kappa 0.46
(CI 0.24 to 0.68), so the NLI judge passes as a usable over-claim signal,
though a noisy one: it flags 6 of the 48 real glosses as contradicted and
the rater calls none of those 6 wrong. And the rater calls 32 of the 36
real glosses that NLI marks not entailed "accurate" (89%), which trips the
third rule: the not-entailed rate measures how weak a list of 15 terms is
as a premise, not over-claim, and I've stopped reporting it as one.

Provenance result. Pooled over the four snapshots, real glosses are
contradicted by their source-paper titles 21% of the time (39 of 186, CI
16% to 27%) against 31% for controls (58 of 186, CI 25% to 38%). The
direction is right but the intervals overlap, so under the rule I don't
lean on it: titles are too coarse a premise to grade a one-sentence gloss.

## 16. What counts as a ground-truth node for the extrinsic eval

`src/tkh/eval/extrinsic.py`, `match_ground_truth_methods`.

`ground_truth.json` gives each question a list of expected method names as
text. The extrinsic eval scores node ids, so those names have to be
resolved against the corpus. Exact surface-form match handles most of
them. For the rest I allowed a substring match in either direction, which
is what catches "HamGNN" inside the node "Universal HamGNN Hamiltonian
model", and "DeepH" for the expected "xDeepH".

The first version of that only required the expected NAME to be at least
four characters, and said nothing about the node's surface form. In a
materials corpus that is a disaster, because the node table contains
chemical elements. "N", "P", "S", "C" and "Si" are all substrings of
"physics-informed", so the ground truth for that term became seven element
nodes. "Hessian training" resolved to He, N, S and Si. Across the 14
type-A questions, 47 of 147 ground-truth node ids were surface forms of
two characters or less, i.e. about a third of the answer key was chemical
elements that have nothing to do with the question.

I found this in a review pass, not by looking at the metric, which is the
uncomfortable part: the numbers looked plausible the whole time. It also
means the report had been describing behaviour the code didn't have. I'd
written that about a third of the expected method names "don't match any
corpus node" and are descriptive categories, when in fact every single
term matched something, mostly junk.

The fix is one condition: both sides of a substring match must be at least
`min_substring_len` (4) characters. `tests/test_ground_truth_match.py`
pins it, including that "MACE-F" no longer resolves to the node "ACE" and
that an unmatchable term reports "none" rather than quietly matching an
element. After the fix, of 63 expected-method mentions (50 distinct names)
47 match exactly, 7 by substring and 9 not at all, and the 9 really are
descriptive categories like "hybrid frameworks" and "GNN free energies".
Two of the 14 questions (Q5 and Q11) are left with no ground-truth node at
all and drop out of the eval, so the extrinsic numbers are over 12
questions now, not 14.

What it cost: routing, which is the main extrinsic metric, is weaker than
it was. Section 14 has the new numbers. The direction of every conclusion
held, which is luck as much as anything.

## 17. What alpha actually weights

`cluster.py`, `combine_affinities` and `_normalize_affinity`;
`scripts/affinity_mass_share.py`, `outputs/affinity_mass_share.json`.

Section 7 describes alpha as the knob that trades structure against
meaning, and both this file and the report talked about alpha=0.3 as
"structure gets 30%, semantics 70%". That reading is wrong, and I only
checked it after the shuffle null in section 13 came back saying the
structure term barely matters.

The two graphs are each normalised by their own 99th percentile, which
puts both in [0, 1] but not on the same scale in any useful sense. A
semantic k-NN edge is a cosine similarity between neighbours, so after
normalisation the typical value sits around 0.57. A structural entry is a
sum of 1/(n-1) shares, and most pairs co-occur in exactly one hyperedge of
middling arity, so the typical value sits around 0.06, an order of
magnitude down. Multiplying the first by 0.7 and the second by 0.3 does
not give a 70/30 split of anything.

Measured on the 2026 snapshot, the structural term carries about 7% of the
total affinity mass at alpha=0.3 and about 14% at alpha=0.5, and the other
three snapshots land between 5% and 7% at the shipped alpha. So the
shipped method is semantic k-NN clustering with the hypergraph acting as a
tie-breaker, and the shuffle null's collapse (section 13) stops being a
surprise and starts being the obvious consequence.

I'm leaving the parametrisation alone for this submission, because
changing it means re-sweeping alpha and relabelling, but I want to be
clear that it's a defect in the method's presentation rather than a
subtlety. The clean version is to normalise the two graphs so a unit of
alpha means a unit of influence, for instance by rank or quantile
transform, or at least to report the realised mass share next to alpha
every time it's quoted. What I do NOT think this justifies is quietly
raising alpha so structure "counts more", since section 13 already
measured what structure buys on held-out hyperedges and the answer was
level 0 only.

One thing the numbers do NOT say is that structure is idle. Structural
pairs outnumber semantic ones at 2026 (65,679 against 58,856), and 94% of
them are pairs the k-NN graph never proposes at all, so structure is
mostly adding edges rather than re-weighting existing ones. A small weight
on an edge that would otherwise be absent is not the same thing as a small
weight on an edge that is already there, and it is probably why the
held-out hyperedge test in section 13 still finds something at level 0
when the mass share looks negligible.

How alpha was chosen, and what that does to the numbers
(`scripts/alpha_sweep.py`; `eval/stability.py`, `perturbation_stability`).
alpha=0.3 was picked by maximising the same coherence and stability
numbers the report then presents as results, so for the shipped setting
those numbers are in-sample. They still separate the method from its
nulls, which is a different question, but they are not an independent
estimate of how good 0.3 is.

One of the three criteria is also confounded with alpha itself. The
perturbation test removes 10% of hyperedges and computes the semantic
k-NN graph once, untouched, so it only ever disturbs the structural term.
A clustering that leans less on structure is steadier under it
mechanically: perturbation ARI is 0.75, 0.92 and 0.98 by level at
alpha=0.05, against 0.58, 0.78 and 0.91 at 0.3 and 0.47, 0.66 and 0.81 at
0.5. As robustness that is close to meaningless, since it rewards
ignoring the hypergraph. So I re-read the sweep without it, on the six
per-level numbers left (coherence z-score and cross-snapshot ARI).
alpha=0.3 still beats 0.5 on all six and is still the only value that
does, with 0.05, 0.15, 0.25, 0.4 and 0.45 at five of six. That's the
version of the choice I'd defend, with thin margins and no uncertainty on
the z-scores. The fix for the test itself is a semantic-side perturbation
(drop or re-embed a sample of nodes) so both signals are under the same
stress. Untried.

## 18. Re-applying the pre-registered rules after the matcher fix

The ground-truth fix in section 16 landed after the decision rules in
section 15 had already been written down and read. That's exactly the
situation where it would be easy to cheat without noticing, so to be
explicit about what I did: I did not touch any rule, I reran the
experiments the fix could affect (the relabelling rule and the extrinsic
re-evaluation rule, both of which score against ground truth) and re-read
the same rules against the new numbers. Both still pass, and both pass
less comfortably than before, which is written into the results above
rather than smoothed over. The rules that don't involve ground truth
(alpha, level-0 skew, temporal thresholds, warm start, shuffle null,
multilevel coarsening, structural holdout, localisation) don't depend on
the matcher and weren't rerun.

The honest risk here is the one I can't fully rule out: if the fix had
flipped a rule from pass to fail, I'd like to think I'd have reported the
fail, but I didn't have to, so that's a claim about myself and not
evidence. The old numbers are kept next to the new ones throughout so a
reader can check the direction of every change.

## 19. Reproducibility details

`embeddings.py` and `eval/faithfulness.py`, the pinned model revisions;
`labeling.py`, `apply_labels_to_hierarchy` and `unlabelled_super_nodes`;
`scripts/t5_apply_labels.py`, `scripts/t6_evaluate.py`.

Both Hugging Face models are now loaded at a fixed commit (the revision
hashes in the two modules) rather than whatever the Hub serves on the day.
The cached copies these results came from are those commits, so pinning
changed nothing here. It only stops a later upload from quietly changing
the embeddings, and with them every cluster, for someone rerunning this
next year.

`run_pipeline.py` writes every label as null, since it has no business
guessing which label goes with a freshly built cluster. The README didn't
say `t5_apply_labels.py` had to run next, and if you skipped it,
`t6_evaluate.py` died with a KeyError deep in the faithfulness log line. A
reviewer following the README literally would have hit exactly that.
Three changes. The README now lists the step in the main path.
`t6_evaluate.py` checks for unlabelled super-nodes up front and says which
script to run. And applying labels is now guarded. A label only goes onto a
super-node if the prompt rebuilt from that super-node's current members is
byte-identical to the prompt the label was written from. Persistent ids
are reused across reruns, so without this, a change to alpha or the
thresholds would put old labels on new clusters without any error. The
guard accepts all 248 shipped labels, and it rejects two clusters whose
members are swapped under the same ids (`tests/test_label_apply.py`).

Platforms. On Linux, the PyPI wheel for `torch==2.14.0` depends on the
CUDA 13 toolkit and a set of nvidia packages, none of which are in
`requirements.txt`. So "the full pinned set, CPU-only" was only true on
Windows and macOS. A `2.14.0+cpu` wheel exists on the PyTorch CPU index,
and the README now says to install that first on Linux. I checked the
dependency metadata and the index listing, but I haven't run the Linux
install end to end.

## 20. Confidence intervals on cross-snapshot stability

`eval/stability.py`, `cross_snapshot_stability` and `_half_sample_ari`.

The original CI was a t-interval over three numbers, one per transition,
with two degrees of freedom. It was also answering the wrong question.
The three transitions aren't draws from one distribution: 2022 to 2024 is
the phase where the edge count nearly doubles. A t-interval treats that
real difference as noise.

The interval I report now holds each clustering fixed and asks how much
the ARI depends on which concepts happen to be in the corpus. It
resamples the shared nodes within each transition, and the interval for
the mean over transitions is built from the same resamples. It does not
cover variability of the clustering procedure itself. The perturbation
measure is for that.

I got the resampling wrong on the first try, and I worked through why with
the coding agent. An ordinary bootstrap resamples with replacement, and a
node drawn twice sits in the same cluster as its copy in both partitions,
so it adds an agreeing pair. I'd assumed that effect was negligible
because it's small against the total number of pairs. But ARI is driven
by same-cluster pairs, not all pairs, and at level 2 (200 clusters,
averaging 7 members at 2020 and 27 at 2026) the duplicate pairs are a sizeable fraction of
those. The level-2 interval came out as 0.651 to 0.682 around an estimate
of 0.647, entirely above it. Resampling half the nodes without
replacement has no duplicates. Its mean matches the full-data ARI to
within 0.001 in the four cases I checked, and for smooth statistics it has
roughly the variance of the ordinary bootstrap.
`tests/test_stability_ci.py` includes the many-small-clusters case, which
the with-replacement version fails.

Results. The means are unchanged (0.35, 0.45 and 0.65 by level). The
intervals are much narrower than the t-intervals: 0.34 to 0.37, 0.44 to
0.47 and 0.63 to 0.66. The more interesting thing is per transition. At
level 1, 2020 to 2022 is 0.52 (0.49 to 0.56), 2022 to 2024 is 0.37 (0.35
to 0.40) and 2024 to 2026 is 0.47 (0.45 to 0.49). Those don't overlap, so
the middle transition really is less stable, not just noisier. It's also
the one where the corpus changed most. So the per-transition numbers are
the honest unit, and the mean over three is a summary, not an estimate of
anything.

## 21. Blind intruder test for coherence

`eval/blind.py`, `make_intruder_items` and `score_intruder`;
`scripts/blind_eval.py`; `outputs/blind_eval/`; rule and result in
section 15.

The brief lists blind LLM or human judgement as one way to measure
coherence independently. TF-IDF turned out to be less independent than I'd
claimed (section 5), so I added the classic intruder test from the
topic-model literature (Chang et al. 2009). Show a rater five members of a
super-node plus one node from a different level-0 branch, shuffled, and
see whether the rater can pick the odd one out. If the groups mean
something, the intruder stands out. Chance is 1 in 6.

Two design choices carry the weight. First, the intruder is type-matched to
one of the shown members, so a cluster of short technique names can't give
away a long claim just by its length or shape. Second, a quarter of the items
(20 of 79) are nulls: five random nodes plus an intruder picked the same way. A rater
using surface cues rather than meaning would beat chance on those too.
The rater saw only item ids and six strings. The answer key never left a
separate folder until the ratings were in. The rater was a fresh Claude
Code subagent on Opus, a different model from the Sonnet agents that
wrote the labels. I checked its transcript afterwards: one read of its
own item file, then its answer. It's one LLM rater, with no second rater
and no human pass, so there is no inter-rater agreement to report. That's
the obvious next thing if this mattered more.

Result, by the rule written beforehand: 74% at level 2 and 58% at level 1,
both clearly above chance, and 5% on the nulls. Level 0 was 4 of 12, which
isn't distinguishable from chance. That fits everything else about level
0: it's the least stable level, and its labels are the ones that read as
mixed topics. The intruder test is the first coherence result in this
project that doesn't pass through the same text representation the
clustering used, and it says the finer levels hold together for a blind
reader and the coarsest one may not.

## 22. Blind gloss rating, provenance check and CIs on faithfulness

`eval/blind.py`, `make_gloss_items` and `score_gloss_ratings`;
`eval/faithfulness.py`, `premise_member_forms`,
`check_provenance_faithfulness`; `eval/stats.py`.

Three additions to the faithfulness evaluation, all driven by the same
worry: the NLI judge had never been checked against anything.

Wilson intervals. Every faithfulness rate now has one
(`real_contradiction_ci95` and so on in metrics.json). They're wide
because the samples are small: 29 checkable glosses at 2020 means a 14%
contradiction rate comes with an interval of about 6% to 31%.

Blind rating. There are 72 items, 12 real and 6 control per snapshot. A
real item shows a gloss plus up to 15 of its super-node's members that the
labeller never saw. A control item shows the same gloss with another
super-node's members. The rater marks each item accurate, vague or wrong.
The members are drawn with the same function the NLI premise uses
(`premise_member_forms`), and the NLI judge is run on exactly those
members, so the rater and NLI grade identical inputs. The same isolation
and the same kind of rater as section 21 apply.

The result (section 15) changes the headline. The rater rated 0 of 48 real
glosses wrong and 23 of 24 controls wrong. Of the 36 real glosses NLI
calls not entailed, the rater calls 32 accurate. So the old 60-79%
"not-entailed" figure was measuring the premise, and the honest over-claim
rate is 0 of 48 with an upper bound of 7%. NLI contradiction does track
the rater (kappa 0.46), but it raises false alarms on real glosses. I'd
now read the 5-14% NLI contradiction rate as an upper bound with noise in
it, not as an over-claim rate.

Two caveats. One LLM rater is not a panel. And "wrong" is a high bar, since
a vague gloss is never wrong, which is partly why real items score so
cleanly. The five real glosses rated vague are the ones to look at if you
want the weak spots.

Provenance. This is the brief's own version of the circularity. The
labeller's only input was surface forms, and the clusters were built from
surface forms, so grading glosses against more surface forms can only go
so far. Every node records which papers it was extracted from, and
`collection10_articles.csv` has their titles. Neither the labeller nor the
clustering ever saw a title. So I graded each gloss against the titles of
its held-out members' source papers, using the same NLI judge and the same
random-other-cluster control. The rule was set beforehand, and the result
(section 15) doesn't pass it: real glosses are contradicted less often
than controls, 21% against 31%, but the intervals overlap. I read it as
titles being too coarse to grade a one-sentence gloss, not as evidence
either way, and I've kept it in the metrics without leaning on it.

## 23. Publishing model outputs so a rerun can be checked

`model_outputs.py`; `embeddings.py`, `encode_semantic`;
`eval/faithfulness.py`, `nli_labels`; `scripts/reproduce_all.py`,
`scripts/export_release.py`, `scripts/verify_release.py`.

Nothing in this project is trained, so there are no weights to publish.
The method is two frozen models, pinned to exact Hub commits (section 19),
plus deterministic code. That means a rerun on another machine can only
differ in one place: the numbers the two models hand back.

I checked how fragile those are before designing anything around them.
On this machine, with the same model and the same library versions, the
same text doesn't always embed to the same bits. When the pipeline embeds
nodes snapshot by snapshot, 514 of the 5,428 vectors differ from a single
pass over all of them, by at most 1.1e-7. One text embedded alone differs
from the same text inside a batch by 7.5e-8. It's float rounding in
batched matrix products, and here it happens not to change any cluster
(the zero-perturbation rebuild in the stability check gives ARI 1.0). But
it means a different CPU or maths library will certainly move the
embeddings a little, and a near-tie in someone's nearest-neighbour lists
could then change a cluster, and every number downstream of it.

So the release publishes every model output my run produced, and
`TKH_MODEL_OUTPUTS=replay:DIR` makes the code use those instead of the
models. A reproducer can then separate the two possible causes of a
difference. If the replay reproduces my outputs, my results follow from
my model outputs and the code is fine; any difference in their own run
comes from their model outputs, and `verify_release.py
compare-embeddings` shows how far theirs are from mine and whether any
nearest neighbour changed. Outputs are keyed by the whole call (model
revision, texts in order, batch size, sequence length), not by text,
because of the batch effect above. A text-keyed store couldn't reproduce
both passes that embed the same node.

Building it turned up two problems. First, loading a pinned model still
sent a request to the Hub every time, and when the network dropped
mid-run each load sat through five retries before using the cache. Both
models now load from the local cache first and only go online if the
files are missing. Second, and more interesting, `localisation.json`
wasn't reproducible. Its per-cluster semantic exposure is a mean over a
Python set of node ids, and set order depends on string-hash
randomisation, so the floats were summed in a different order each run
and came out different in the 16th digit. The summary statistics are
rank-based and never moved, which is why every earlier check of
`metrics.json` passed. The byte-level checksum over all 45 output files is
what caught it. `localisation.py`, `cluster_rows`, now sums in sorted order,
and the file is byte-identical under two different hash seeds.

What the release does not cover. The LLM-written labels and the blind
ratings aren't model outputs of the pipeline. They are checked-in data,
and they can't be regenerated bit for bit by anyone. And replay only
checks the code after the models. Whether my model outputs are what the
pinned models really produce is what `compare-embeddings` is for, run on
the reproducer's machine.

## 24. Label and rating sets: the shipped samples, or your own

`labeling.py`, `validate_template`, `label_set_template`,
`write_labeller_request`, `apply_labels_to_hierarchy`; `replies.py`,
`extract_json`; `eval/blind.py`, `rating_reply_problems` and
`glosses_not_applied`; `scripts/t5_dump_labeling_input.py`,
`t5_import_labels.py`, `t5_apply_labels.py`, `blind_eval.py`,
`reproduce_all.py`.

The labels and the blind ratings are the two parts of this project that
no one can rerun bit for bit, because an LLM wrote them. Section 23
handles the models inside the pipeline; this handles the ones outside it.
The labels and ratings I used are checked in, and a plain run uses them.
What I wanted on top was for a reproducer to make their own set with any
LLM or by hand, with their own prompt if they like, and run the whole
evaluation on it with two flags. My numbers come from one labelling run,
two if you count the superseded first set, so I don't know how much they
move between labellers. Someone else's set is the most direct test of
that I can offer.

A set is just a directory. A label set holds `<year>/labeling_output.json`
and, if it was written from a custom prompt, `prompt_template.txt`. A
rating set holds the packets, their keys and the ratings. The shipped
sets already had that shape (`outputs/snapshots` and `outputs/blind_eval`),
so nothing about them changed, and I checked that before building on it:
the default prompts, the rebuilt rating packets, including a fresh NLI
run, and every output file are byte-identical to what's in the published
release.

Three design choices matter.

The staleness guard from section 19 compares each label's stored prompt
with the prompt rebuilt from the cluster's current members. With custom
templates, that only works if the rebuild uses the template the set was
written from. So the set carries its template, and `t5_apply_labels.py
--labels DIR` uses it. The test shows the same labels apply cleanly with
their own template and come up stale against the default one, so a
custom prompt doesn't weaken the guard.

The template is customisable, but the number of members the labeller sees
is not. The faithfulness check holds out exactly the members the
labeller didn't see (`held_out_member_ids`), and a different sample size
would have to be threaded through to it. If it wasn't, the held-out set
would silently overlap the labeller's input, which is the circularity
section 12 was about. So I fixed it at 25 rather than make it a flag.

Both imports are all or nothing, and scoring checks that the ratings
belong to the labels in use. Testing the workflow found a real bug here.
A rating import that failed on one reply still rewrote the other reply's
file, and overwrote the rater description. Now nothing is written unless
every reply checks out. `blind_eval.py score` refuses a rating set whose
glosses aren't among the labels currently applied, because gloss ratings
are about specific glosses, and scoring them against different labels
would produce numbers that mean nothing.

`--replay` doesn't combine with a custom label set. The recorded NLI
outputs cover my glosses only, and replay stops with an error on the
first call it has no recording for rather than guess.

I tested the workflow end to end with a stand-in labeller and stand-in
raters that answer every item in the required format, one reply wrapped
in a code fence the way chat models often send it. That tests the
plumbing (custom template, 248 labels imported and applied, packets rebuilt
for them, ratings imported and scored, both refusals), not the quality of
anyone's labels.
