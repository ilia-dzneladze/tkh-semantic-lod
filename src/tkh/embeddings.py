"""Semantic signal(s) over node surface_form text.

encode_semantic drives clustering. encode_lexical_tfidf is reserved for
the T6 coherence check only, never feed it into clustering.
See DESIGN_NOTES.md section 5.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from tkh import model_outputs

_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# pinned Hub commit so a later upload can't change the embeddings:
# DESIGN_NOTES.md section 19
_MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        try:
            # a pinned commit that's already cached needs no network
            _model = SentenceTransformer(_MODEL_NAME, revision=_MODEL_REVISION, local_files_only=True)
        except OSError:
            _model = SentenceTransformer(_MODEL_NAME, revision=_MODEL_REVISION)
    return _model


def encode_semantic(texts, batch_size=64, show_progress_bar=False, max_seq_length=None):
    """max_seq_length temporarily overrides the model's default (384) for
    this call, then restores it. Used by the T6 extrinsic eval, where
    hyperedge-neighborhood-enriched texts are long enough that leaving the
    default causes multi-minute batches from padding to the longest
    sequence, see extrinsic.py's build_retrieval_texts.

    Goes through model_outputs, so a whole run can be recorded or replayed
    without the model (DESIGN_NOTES.md section 23)."""
    texts = list(texts)
    payload = {"model": f"{_MODEL_NAME}@{_MODEL_REVISION}", "texts": texts,
               "batch_size": batch_size, "max_seq_length": max_seq_length}
    return model_outputs.cached_array(
        "embed", payload, lambda: _encode(texts, batch_size, show_progress_bar, max_seq_length))


def _encode(texts, batch_size, show_progress_bar, max_seq_length):
    model = _get_model()
    prev = model.max_seq_length
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    try:
        emb = model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar,
            normalize_embeddings=True,  # so cosine sim == dot product
        )
    finally:
        model.max_seq_length = prev
    return np.asarray(emb)


def encode_lexical_tfidf(texts):
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    return X  # sparse, cosine sim via sklearn.metrics.pairwise.cosine_similarity


def semantic_knn_graph(embeddings, k=15):
    """Sparse cosine-similarity k-NN graph (embeddings assumed unit-normalized).

    Symmetrized as a union (edge kept if either side lists the other in
    its top-k), not mutual/intersection kNN. See DESIGN_NOTES.md section 6.
    """
    n = embeddings.shape[0]
    k_eff = min(k + 1, n)  # +1 because a point is its own nearest neighbor
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn.fit(embeddings)
    dist, ind = nn.kneighbors(embeddings)
    # cosine distance = 1 - cosine similarity for normalized vectors
    import scipy.sparse as sp
    rows, cols, vals = [], [], []
    for i in range(n):
        for d, j in zip(dist[i], ind[i]):
            if j == i:
                continue
            rows.append(i)
            cols.append(j)
            vals.append(1.0 - d)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)  # symmetrize: keep max similarity if asymmetric in k-NN
    return A
