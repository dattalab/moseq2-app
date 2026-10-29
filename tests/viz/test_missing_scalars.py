"""Comparing groups on a scalar when some sessions never used a syllable.

A session that never performed a syllable has no duration, velocity or
height for it: the value is missing, not 0. Every group comparison leaves
those sessions out of that syllable, so each syllable is tested on the
sessions that used it. A syllable a test cannot evaluate (a group with no
session that used it, or one value everywhere) gets p = 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moseq2_viz.model.stat import bootstrap_group_means, mann_whitney, ttest, ztest

SESSIONS = 10


def stats(used, durations):
    """Per-session rows for two groups ``a`` and ``b``.

    ``used[(group, syllable)]`` lists the sessions that used the syllable and
    ``durations[(group, syllable)]`` is the mean duration they used it for.
    """
    rng = np.random.default_rng(0)
    rows = []
    syllables = sorted({s for _, s in used} | {s for _, s in durations})
    for group in ("a", "b"):
        for i in range(SESSIONS):
            for s in syllables:
                if i in used.get((group, s), ()):
                    rows.append((group, f"{group}{i}", s, durations[(group, s)] + rng.normal(0, 0.02), 1.0))
                else:
                    rows.append((group, f"{group}{i}", s, np.nan, 0.0))
    return pd.DataFrame(rows, columns=["group", "uuid", "syllable", "duration", "usage"])


def five_syllables():
    """Syllable by syllable:

    0: everyone used it; b's last longer.
    1: same duration, but half of b never used it.
    2: one b session used it, no a session.
    3: no one used it.
    4: half of b never used it, and the half that did last longer.
    """
    everyone, half = range(SESSIONS), range(5)
    return stats(
        used={("a", 0): everyone, ("b", 0): everyone, ("a", 1): everyone, ("b", 1): half,
              ("b", 2): [0], ("a", 4): everyone, ("b", 4): half},
        durations={("a", 0): 0.5, ("b", 0): 0.6, ("a", 1): 0.5, ("b", 1): 0.5, ("b", 2): 0.5,
                   ("a", 3): 0.5, ("a", 4): 0.5, ("b", 4): 0.7},
    )


@pytest.mark.parametrize("test", [ztest, ttest, mann_whitney], ids=["z-test", "t-test", "Mann-Whitney"])
def test_sessions_that_never_used_a_syllable_do_not_count_as_zero(test):
    result = test(five_syllables(), "a", "b", statistic="duration", max_syllable=5)
    assert bool(result.loc[0, "is_sig"]), "the real difference is missed"
    assert not bool(result.loc[1, "is_sig"]), "b's unused sessions were counted as 0 s"
    assert result.loc[2, "pvalue"] == 1.0 and result.loc[3, "pvalue"] == 1.0
    assert bool(result.loc[4, "is_sig"]), "a difference among the sessions that used it is missed"
    assert result["pvalue"].notna().all()


def test_a_group_with_one_observed_session_bootstraps_without_nan():
    """Syllable 1, used by everyone, keeps all ten b sessions in the table."""
    everyone = range(SESSIONS)
    table = stats(used={("a", 0): everyone, ("b", 0): [0], ("a", 1): everyone, ("b", 1): everyone},
                  durations={("a", 0): 0.5, ("b", 0): 0.6, ("a", 1): 0.5, ("b", 1): 0.5})
    boots = bootstrap_group_means(table, "a", "b", statistic="duration", max_syllable=2)
    assert np.isfinite(boots["b"]).all()


def test_one_observed_session_against_many_is_tested():
    """A t-test of one session's duration against ten has a p-value."""
    everyone = range(SESSIONS)
    table = stats(used={("a", 0): everyone, ("b", 0): [0], ("a", 1): everyone, ("b", 1): everyone},
                  durations={("a", 0): 0.5, ("b", 0): 0.9, ("a", 1): 0.5, ("b", 1): 0.5})
    result = ttest(table, "a", "b", statistic="duration", max_syllable=2)
    assert 0 < result.loc[0, "pvalue"] < 1
