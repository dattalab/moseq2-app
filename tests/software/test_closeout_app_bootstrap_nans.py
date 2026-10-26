"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py





def test_missing_observations_do_not_poison_bootstrap_limits(monkeypatch):
    from moseq2_app.stat.view import get_ci_vect_vectorized
    monkeypatch.setattr(np.random, 'choice', np.random.RandomState(17).choice)
    expected = get_ci_vect_vectorized(np.array([1., 2., 3.]), n_boots=200)
    monkeypatch.setattr(np.random, 'choice', np.random.RandomState(17).choice)
    actual = get_ci_vect_vectorized(np.array([1., np.nan, 2., 3.]), n_boots=200)
    np.testing.assert_array_equal(actual, expected)
    assert np.isnan(get_ci_vect_vectorized(np.array([np.nan]), n_boots=10)).all()
