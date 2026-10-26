"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


def test_empty_usage_normalization_is_finite():
    from moseq2_viz.model.util import normalize_usages
    actual = normalize_usages({0: 0., 1: 0.})
    assert all(np.isfinite(list(actual.values())))
    assert sum(actual.values()) == 0
