"""Permutation p-values count every permutation that ties the data.

The Kruskal-Wallis and Dunn's tests behind the stats widget's "KW & Dunn's"
option take their p-values from permutations: the share of label shuffles
whose statistic is at least as large as the one observed. Ties count, and with
groups of different sizes there are many. The observed statistic and the
permuted ones are computed through different sums, so a permutation that ties
in exact arithmetic can land a few units in the last place below the observed
value. Those still count, as they do in ``scipy.stats.permutation_test``.

The designs here are small enough to count by hand. Two sessions in ``a`` with
values 0 and 2 against three in ``b`` with 2, 4 and 2 rank as 1, 3 | 3, 5, 3.
Six of the ten ways to choose two sessions for ``a`` put its rank sum as far
from the expected 6 as the observed 4 does: the three that pair rank 1 with a
3, and the three that pair rank 5 with a 3. So the permutation p-value is 0.6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moseq2_viz.model.stat import compute_pvalues_for_group_pairs, run_kruskal

N_PERM = 10000


def table(syllables):
    """Per-session rows; ``syllables[s][group]`` lists each session's duration, NaN if unused."""
    rows = []
    for s, groups in enumerate(syllables):
        for group, durations in groups.items():
            for i, duration in enumerate(durations):
                used = not np.isnan(duration)
                rows.append((group, f"{group}{i}", s, duration, float(used)))
    return pd.DataFrame(rows, columns=["group", "uuid", "syllable", "duration", "usage"])


def test_kruskal_wallis_counts_a_tie_reached_through_a_different_sum():
    """The mirror of the observed split ties it and is counted: p = 0.6, not 0.3."""
    two_against_three = table([{"a": [0.0, 2.0], "b": [2.0, 4.0, 2.0]}])

    df_k, _, _ = run_kruskal(two_against_three, statistic="duration", max_syllable=1, n_perm=N_PERM)

    assert abs(df_k.loc[0, "p_adj"] - 0.6) < 0.02, df_k.loc[0, "p_adj"]


def test_kruskal_wallis_counts_ties_among_the_sessions_that_used_a_syllable():
    """The same design, with a third session in each syllable that never used it.

    Each syllable is tested on the five sessions that used it, and both come
    out at 0.6.
    """
    nan = np.nan
    with_unused_sessions = table([
        {"a": [0.0, 2.0, nan], "b": [2.0, 4.0, 2.0]},
        {"a": [nan, 0.0, 2.0], "b": [2.0, 4.0, 2.0]},
    ])

    df_k, _, _ = run_kruskal(with_unused_sessions, statistic="duration", max_syllable=2, n_perm=N_PERM)

    assert (abs(df_k["p_adj"] - 0.6) < 0.02).all(), df_k["p_adj"].tolist()


def test_dunn_counts_a_tie_reached_through_a_different_sum():
    """Dunn's statistic for one session against six, at rank 6 and at rank 2 of 1..6 with three 6s.

    The mean rank of all seven is 4, so a session at rank 6 and one at rank 2
    are equally far from the rest. Computed the way Dunn's test computes them,
    the two differ in the last place.
    """
    ranks = np.array([6.0, 2.0, 1.0, 3.0, 4.0, 6.0, 6.0])
    observed = np.abs(ranks[None, :1].mean(1) - ranks[None, 1:].mean(1))
    mirror = np.abs(ranks[None, 1:2].mean(1) - ranks[None, [0, 2, 3, 4, 5, 6]].mean(1))
    assert mirror[0] < observed[0]

    p, _ = compute_pvalues_for_group_pairs(
        {("a", "b"): observed}, {("a", "b"): np.tile(mirror, (99, 1))},
        pd.DataFrame({"is_sig": [True]}), ["a", "b"], n_perm=99,
    )

    assert p[("a", "b")].iloc[0] == 1.0


def test_a_permutation_genuinely_below_the_data_is_not_a_tie():
    """The allowance is for rounding only: 1e-12 below the observed value is below it."""
    p, _ = compute_pvalues_for_group_pairs(
        {("a", "b"): np.array([1.0])}, {("a", "b"): np.array([[1.0 - 1e-15], [1.0 - 1e-12]])},
        pd.DataFrame({"is_sig": [True]}), ["a", "b"], n_perm=2,
    )

    assert p[("a", "b")].iloc[0] == (1 + 1) / (2 + 1)
