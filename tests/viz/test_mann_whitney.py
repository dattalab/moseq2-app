"""Mann-Whitney between the two groups a user picks in the syllable-stats cell.

``mann_whitney`` builds one row per session and tests the sessions of the
two named groups against each other, syllable by syllable. Which sessions
go into each sample is decided by the group names alone, so other groups in
the table and the order the two are named in leave the result unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moseq2_viz.model.stat import mann_whitney


def stats(groups):
    """A per-session syllable table from ``{group: [each session's usage]}``, two syllables."""
    return pd.DataFrame(
        [
            {"group": g, "uuid": f"{g}{i}", "syllable": s, "usage": usage + 0.001 * s}
            for g, usages in groups.items()
            for i, usage in enumerate(usages)
            for s in range(2)
        ]
    )


def pvalues(df, group1, group2):
    return mann_whitney(df, group1, group2, max_syllable=2)["pvalue"].to_numpy()


A = [0.10, 0.12, 0.14, 0.16]
B = [0.30, 0.11, 0.31, 0.13, 0.32]


def test_another_group_in_the_table_leaves_the_comparison_alone():
    """Sessions from a third group are in neither sample."""
    two = stats({"a": A, "b": B})
    three = stats({"a": A, "b": B, "c": [0.9, 0.8, 0.7]})
    np.testing.assert_allclose(pvalues(three, "a", "b"), pvalues(two, "a", "b"))


def test_groups_of_different_sizes_compare_the_same_in_either_order():
    """Naming the larger, alphabetically later group first changes nothing."""
    df = stats({"a": A, "b": B})
    np.testing.assert_allclose(pvalues(df, "b", "a"), pvalues(df, "a", "b"))
