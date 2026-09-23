import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh import model_outputs, embeddings  # noqa: E402
from tkh.eval import faithfulness  # noqa: E402


def _fake_encode(calls):
    def encode(texts, batch_size, show_progress_bar, max_seq_length):
        calls.append(list(texts))
        rng = np.random.default_rng(len(texts) * 7 + batch_size)
        return rng.standard_normal((len(texts), 4)).astype(np.float32)
    return encode


def test_unset_env_just_computes(monkeypatch):
    monkeypatch.delenv(model_outputs.ENV_VAR, raising=False)
    calls = []
    monkeypatch.setattr(embeddings, "_encode", _fake_encode(calls))
    embeddings.encode_semantic(["a", "b"])
    assert calls == [["a", "b"]]


def test_replay_returns_recorded_bits_without_calling_the_model(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(embeddings, "_encode", _fake_encode(calls))
    monkeypatch.setenv(model_outputs.ENV_VAR, f"record:{tmp_path}")
    recorded = embeddings.encode_semantic(["MACE", "SchNet", "GAP"], batch_size=32, max_seq_length=64)

    def refuse(*a, **k):
        raise AssertionError("replay must not run the model")
    monkeypatch.setattr(embeddings, "_encode", refuse)
    monkeypatch.setattr(embeddings, "_get_model", refuse)
    monkeypatch.setenv(model_outputs.ENV_VAR, f"replay:{tmp_path}")
    replayed = embeddings.encode_semantic(["MACE", "SchNet", "GAP"], batch_size=32, max_seq_length=64)
    assert replayed.dtype == recorded.dtype and np.array_equal(replayed, recorded)


def test_key_covers_everything_that_changes_the_floats(monkeypatch, tmp_path):
    # same texts, but a different batch size or order is a different call
    monkeypatch.setattr(embeddings, "_encode", _fake_encode([]))
    monkeypatch.setenv(model_outputs.ENV_VAR, f"record:{tmp_path}")
    embeddings.encode_semantic(["a", "b"], batch_size=64)
    monkeypatch.setenv(model_outputs.ENV_VAR, f"replay:{tmp_path}")
    with pytest.raises(KeyError):
        embeddings.encode_semantic(["a", "b"], batch_size=32)
    with pytest.raises(KeyError):
        embeddings.encode_semantic(["b", "a"], batch_size=64)


def test_nli_replay_needs_no_model(monkeypatch, tmp_path):
    class FakeNLI:
        class config:
            id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

        def predict(self, pairs, show_progress_bar=False):
            return np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])

    monkeypatch.setattr(faithfulness, "_get_model", lambda: FakeNLI())
    monkeypatch.setenv(model_outputs.ENV_VAR, f"record:{tmp_path}")
    pairs = [["The cluster includes MACE.", "Interatomic potentials."], ["x", "y"]]
    assert faithfulness.nli_labels(pairs) == ["entailment", "contradiction"]

    def refuse():
        raise AssertionError("replay must not load the NLI model")
    monkeypatch.setattr(faithfulness, "_get_model", refuse)
    monkeypatch.setenv(model_outputs.ENV_VAR, f"replay:{tmp_path}")
    assert faithfulness.nli_labels(pairs) == ["entailment", "contradiction"]


def test_bad_env_value_is_rejected(monkeypatch):
    monkeypatch.setenv(model_outputs.ENV_VAR, "replay")
    with pytest.raises(ValueError):
        model_outputs.mode()
