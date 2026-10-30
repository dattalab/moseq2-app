"""The syllable statistics plot's error bars.

The analysis notebook's statistics widget draws a 95% confidence interval
around each group's mean for each syllable, bootstrapped by
``get_ci_vect_vectorized`` from the per-session values. A session in which a
syllable never occurs has no value for statistics such as its duration.
"""

from __future__ import annotations

import numpy as np

from moseq2_app.stat.view import get_ci_vect_vectorized


def test_sessions_without_a_value_are_left_out_of_the_bootstrap(monkeypatch):
    """The interval is the one the observed values give on their own."""
    monkeypatch.setattr(np.random, "choice", np.random.RandomState(17).choice)
    observed = get_ci_vect_vectorized(np.array([1.0, 2.0, 3.0]), n_boots=200)
    monkeypatch.setattr(np.random, "choice", np.random.RandomState(17).choice)
    with_missing = get_ci_vect_vectorized(np.array([1.0, np.nan, 2.0, 3.0]), n_boots=200)

    np.testing.assert_array_equal(with_missing, observed)


def test_no_observed_values_give_no_interval():
    assert np.isnan(get_ci_vect_vectorized(np.array([np.nan]), n_boots=10)).all()
