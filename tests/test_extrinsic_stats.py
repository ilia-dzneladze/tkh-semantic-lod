import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tkh.eval.extrinsic import leave_one_out_select, paired_comparison


def test_loo_never_uses_the_held_out_question():
    # config "a" is best only on question 0; "b" is best on the rest
    scores = {"a": np.array([1.0, 0.0, 0.0, 0.0]), "b": np.array([0.0, 0.1, 0.1, 0.1])}
    held_out, chosen = leave_one_out_select(scores, {"a": (1,), "b": (2,)})
    assert chosen[0] == "b" and held_out[0] == 0.0  # can't pick "a" using q0's own score
    assert chosen[1] == "a"  # with q0 among the others, "a" has the higher mean


def test_loo_tie_goes_to_lower_cost():
    scores = {"cheap": np.array([0.5, 0.5]), "dear": np.array([0.5, 0.5])}
    _, chosen = leave_one_out_select(scores, {"cheap": (10,), "dear": (20,)})
    assert chosen == ["cheap", "cheap"]


def test_paired_comparison_identical_inputs():
    rng = np.random.default_rng(0)
    r = paired_comparison([0.1, 0.2, 0.0], [0.1, 0.2, 0.0], rng, n_boot=500)
    assert r["mean_diff"] == 0.0 and r["n_tied"] == 3 and r["p_sign_flip"] == 1.0


def test_paired_comparison_clear_difference():
    rng = np.random.default_rng(0)
    r = paired_comparison(np.full(12, 0.5), np.zeros(12), rng, n_boot=2000)
    assert r["ci95"][0] > 0 and r["p_sign_flip"] < 0.01 and r["n_a_better"] == 12
