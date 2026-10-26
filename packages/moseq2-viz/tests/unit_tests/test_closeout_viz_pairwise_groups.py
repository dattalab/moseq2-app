"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


def _stats(groups):
    return pd.DataFrame([{"group": g, "uuid": g + str(i), "syllable": s,
                          "usage": base + .01 * i + .001 * s}
                         for g, base in groups.items() for i in range(4) for s in range(2)])


def test_pairwise_mann_whitney_ignores_unrequested_group():
    from moseq2_viz.model.stat import mann_whitney
    ab = _stats({"a": .1, "b": .3})
    abc = _stats({"a": .1, "b": .3, "c": .9})
    expected = mann_whitney(ab, "a", "b", max_syllable=2)["pvalue"]
    actual = mann_whitney(abc, "a", "b", max_syllable=2)["pvalue"]
    np.testing.assert_allclose(actual, expected)
