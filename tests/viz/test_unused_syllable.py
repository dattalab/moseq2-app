"""Group comparisons when a syllable has the same value in every session.

A syllable neither group uses has a usage of 0 in every session of both,
for example one only a third group uses. There is no difference to test, so
the z-test, t-test and Mann-Whitney each give it p = 1. Before multiple
comparison correction a single undefined p-value would leave every syllable
unmarked, and Mann-Whitney would refuse identical samples outright.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moseq2_viz.model.stat import mann_whitney, ttest, ztest, ztest_vect


def stats():
    """Syllable 0 is used more by b; syllable 1 by neither group."""
    rng = np.random.default_rng(0)
    rows = []
    for group, shift in (("a", 0.0), ("b", 0.05)):
        for i in range(10):
            rows.append({"group": group, "uuid": f"{group}{i}", "syllable": 0,
                         "usage": 0.10 + shift + rng.normal(0, 0.01)})
            rows.append({"group": group, "uuid": f"{group}{i}", "syllable": 1, "usage": 0.0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("test", [ztest, ttest, mann_whitney], ids=["z-test", "t-test", "Mann-Whitney"])
def test_an_unused_syllable_leaves_the_others_testable(test):
    """The unused syllable gets p = 1 and the real difference is still marked."""
    result = test(stats(), "a", "b", max_syllable=2)
    assert result.loc[1, "pvalue"] == 1.0
    assert bool(result.loc[0, "is_sig"])


def test_the_z_test_of_two_equal_constants_is_one():
    np.testing.assert_array_equal(ztest_vect(np.zeros((20, 2)), np.zeros((20, 2))), [1.0, 1.0])


def test_the_z_test_of_two_different_constants_is_zero():
    np.testing.assert_array_equal(ztest_vect(np.zeros((20, 1)), np.ones((20, 1))), [0.0])
