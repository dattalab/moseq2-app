"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


def test_entropy_rate_default_matches_documented_row_normalization():
    from moseq2_viz.info.util import entropy_rate
    labels = [np.tile([0, 0, 1, 1, 2, 2, 0, 0], 5)]
    np.testing.assert_allclose(entropy_rate(labels), entropy_rate(labels, normalize="rows"))
