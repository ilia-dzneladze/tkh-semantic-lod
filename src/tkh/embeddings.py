"""Text signals over node surface forms.

encode_semantic (MPNet) drives clustering. encode_lexical_tfidf is only
for the T6 coherence check and must never feed clustering.
DESIGN_NOTES.md section 5.
"""
import numpy as np
import scipy.sparse as sp
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
    """Unit-length embeddings, one row per text. max_seq_length overrides
    the model's default (384) for this call only; the extrinsic eval uses
    64 for its long enriched texts. Goes through model_outputs, so a run
    can be recorded or replayed without the model (DESIGN_NOTES.md
    section 23)."""
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
    """Sparse TF-IDF rows (unigrams and bigrams), L2-normalised."""
    return TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform(texts)


def semantic_knn_graph(embeddings, k=15):
    """Sparse k-NN graph with cosine similarity as edge weight. An edge is
    kept if either node has the other in its top k (union, not mutual
    k-NN): DESIGN_NOTES.md section 6."""
    n = embeddings.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")  # +1: each point finds itself
    nn.fit(embeddings)
    dist, ind = nn.kneighbors(embeddings)
    rows, cols, vals = [], [], []
    for i in range(n):
        for d, j in zip(dist[i], ind[i]):
            if j == i:
                continue
            rows.append(i)
            cols.append(j)
            vals.append(1.0 - d)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A.maximum(A.T)
