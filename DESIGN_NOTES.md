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
claim knowledge the corpus didn't have yet. So snapshot inclusion uses
first_seen_year, not origin_year.

A node only makes it into a snapshot if some included hyperedge
references it. I didn't add a separate check for "orphan" nodes with no
edges at all, because in this dataset every node turned out to be
referenced by at least one edge (checked this by comparing the snapshot's
total node count against the full 5798 at cutoff 2026, they match
exactly). If this pipeline were ever run on a different export where that
isn't true, a node could disappear from a snapshot without any warning,
which would be worth fixing then.

## 3. Structural affinity weighting (Zhou et al. hypergraph cut)

`hypergraph.py`, `build_structural_affinity`, the `share` weight formula.

For two concept nodes that co-occur in the same hyperedge, how much
"structural affinity" should that count for? The obvious naive answer is
1.0 per co-occurring pair, but that badly overweights big hyperedges: an
arity-65 hyperedge produces C(65,2) = 2080 pairs, versus 1 pair for an
arity-2 edge, so one big table in one paper would dominate the whole
structural signal. Instead each hyperedge of arity n contributes
1/(n-1) to each of its C(n,2) pairs. This comes from Zhou, Huang and
Schoelkopf's 2006 paper on hypergraph spectral clustering, and the effect
is that the TOTAL weight a hyperedge contributes across all its pairs
stays bounded regardless of arity, instead of growing quadratically.

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

## 5. Keeping the semantic signal separate from the coherence-check signal

`embeddings.py`, the split between `encode_semantic` and
`encode_lexical_tfidf`.

If clustering uses embedding X and then we check "coherence" by measuring
similarity under the same embedding X, high coherence is basically
guaranteed and proves nothing, since the clusters were built to be
similar under exactly that measure. This is the circularity problem T6
warns about. To avoid it, `encode_semantic` (the MPNet embeddings that
drive clustering) and `encode_lexical_tfidf` (plain bag-of-words TF-IDF,
used only for the coherence check) need to come from genuinely different
signal families, not just two different neural embedding models that
were both trained the same way. TF-IDF has no notion of synonymy or
pretraining at all, so it can't just be rediscovering what MPNet already
encoded.

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
matters more. Alpha stayed at the default 0.5 for the final run, I never
found time to run a proper sweep of it, that's a real gap not a checked
answer.

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
hierarchy at the cost of being harder to verify independently. Didn't
build that version, noting it as a real alternative rather than pretending
matching was the obviously correct choice.

The matching itself is Jaccard overlap between a snapshot's clusters and
the next one's, restricted to whatever node ids happen to exist in both
snapshots. Three thresholds control how it's read: STABLE_JACCARD=0.5,
MATCH_THRESHOLD=0.15, SIZE_CHANGE_RATIO=0.2. These are picked by feel,
not fit to anything. In particular MATCH_THRESHOLD=0.15 is fairly
permissive, so a weak 1-1 match (say Jaccard 0.16) still gets labeled as
the SAME entity growing or shrinking rather than flagged as a shaky
match. The raw Jaccard value is saved on every event record, though, so
this can be filtered or re-weighted downstream by confidence instead of
trusting the discrete label blindly. Never got to a real sensitivity
check on these three numbers, that's a gap, not something I verified.

One more thing worth writing down: when a group splits into several
pieces (the split check inside `classify_events`), only the biggest
piece keeps the original persistent id, the smaller pieces get treated
as new births even though the event log records that they came from a
split. I think this is actually the right way to think about a split
(something can't keep two identities at once), but it does mean a single
real-world reorganization can show up in the event log from two
different angles: as a "split" from the parent's point of view and
separately as a "birth" or "merge" for the piece that didn't keep the
name.

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
billing is separate), so instead the same role gets filled by whoever is
working on this repo (me, writing this) reading the dumped member lists
directly and writing labels back into the same file format an API
response would have produced. `write_labeling_input` writes out exactly
the prompt an API call would have gotten, so the process is still
auditable, you can see precisely what information went into each label.

## 13. Coherence measured with a signal clustering never saw

`src/tkh/eval/coherence.py`.

The obvious way to check "are these clusters coherent" is to look at how
similar the members are under whatever signal built the clusters. That's
circular: of course they look similar under that signal, that's what the
clustering optimized for. So the coherence check here uses TF-IDF
(`encode_lexical_tfidf` in embeddings.py) instead of the MPNet embeddings
that actually drove clustering. TF-IDF is plain bag-of-words with no
neural pretraining behind it, so it isn't just a second opinion from a
similar model, it's a genuinely different way of representing the text.

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
(5, 5) matches the flat baseline's recall and precision exactly while
still only scoring about 16% of the candidate pool. Going wider than that
doesn't help further, recall saturates at the flat baseline's level once
the branching is wide enough to not exclude the right answer up front.
