---
license: other
pretty_name: TKH multi-resolution abstraction, model outputs and checksums
---

# TKH multi-resolution abstraction: model outputs and checksums

These are the files behind the results of my multi-resolution semantic
abstraction over the TKH hypergraph (Constructor Knowledge Labs
assessment). They let anyone who reruns the code check whether they got
exactly what I got, and if not, where the difference comes from.

This is not a model. Nothing in the project is trained or fine-tuned.
The method uses two frozen models, pinned to exact commits:

- embeddings: `{embedding_model}`
- NLI judge: `{nli_model}`

Everything after those two models is deterministic code. So these files
hold every output the two models produced in my run, plus checksums of
every result.

## Files

| file | what it is |
|---|---|
| `model_outputs/` | every model call my run made: {n_embed} embedding calls and {n_nli} NLI calls, one `.npy` each, keyed by a hash of the whole call |
| `node_embeddings.npz` | the embedding of each of the {n_nodes} concept nodes, exactly as clustering used it (`node_ids`, `embeddings`) |
| `retrieval_embeddings_2026.npz` | the {n_method} enriched method-node embeddings used by the extrinsic evaluation |
| `linkage_<year>.npz` | the average-linkage merge tree per snapshot (`node_ids`, scipy-style `Z`, `forced`) |
| `environment.json` | where and how this was produced |
| `SHA256SUMS.txt` | a SHA-256 hash for every output in `outputs/` and every file here |

Produced on {platform}, Python {python}, torch {torch}, from code commit
`{git_commit}`.

## Checking your results against mine

Put these files in a `release/` folder at the root of the code
repository, then:

```
# 1. Do my results follow from my model outputs? Replays the whole
#    pipeline on the recorded outputs (no model download), then compares.
python scripts/reproduce_all.py --replay release/model_outputs
python scripts/verify_release.py checksums

# 2. Do your model outputs match mine? Recomputes the node embeddings
#    with the real model on your machine and compares them to mine.
python scripts/verify_release.py compare-embeddings
```

If step 1 matches and your own full run doesn't, the difference is in
the model outputs, not the code. The same text can embed differently in
the last float bits depending on hardware, libraries and even batch
composition. Step 2 shows how big that difference is, and whether it
changes any node's nearest neighbours, which is where it would start to
change the clusters.

## Data

Derived from the TKH export (collection 10) that Constructor Knowledge
Labs provided for the assessment, which ships with the code repository.
The files contain node ids and vectors, no text.
