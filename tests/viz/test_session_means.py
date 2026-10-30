"""The session x syllable table every group comparison starts from.

``get_session_mean_df`` turns the per-session syllable statistics into one
row per session and one column per syllable. A session that never used a
syllable used it zero times, so its usage is 0; its duration, velocity and
other scalars were never measured, so they stay missing. Every syllable up to
``max_syllable`` gets a column even when no session has a value for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moseq2_viz.model.stat import get_session_mean_df


def stats(rows):
    """Rows of (group, uuid, syllable, duration, usage), as compute_behavioral_statistics writes them."""
    return pd.DataFrame(rows, columns=["group", "uuid", "syllable", "duration", "usage"])


def two_sessions():
    """Session a used only syllable 0 and session b only syllable 1."""
    return stats([
        ("g1", "a", 0, 0.5, 1.0), ("g1", "a", 1, np.nan, 0.0),
        ("g2", "b", 0, np.nan, 0.0), ("g2", "b", 1, 0.6, 1.0),
    ])


def test_a_scalar_of_an_unused_syllable_is_missing():
    result = get_session_mean_df(two_sessions(), statistic="duration", max_syllable=2)
    assert np.isnan(result.loc[("g1", "a"), 1]) and np.isnan(result.loc[("g2", "b"), 0])
    assert result.loc[("g1", "a"), 0] == 0.5


def test_usage_of_an_unused_syllable_is_zero():
    result = get_session_mean_df(two_sessions(), statistic="usage", max_syllable=2)
    assert result.loc[("g1", "a"), 1] == 0


def test_every_syllable_is_a_column_even_with_no_values():
    """Syllable 2 has no duration in any session."""
    table = pd.concat([two_sessions(), stats([("g1", "a", 2, np.nan, 0.0), ("g2", "b", 2, np.nan, 0.0)])])
    result = get_session_mean_df(table, statistic="duration", max_syllable=3)
    assert list(result.columns) == [0, 1, 2]
    assert result[2].isna().all()


def test_only_real_sessions_get_a_row():
    """No group/session pair is made up from the levels of the index."""
    for statistic in ("usage", "duration"):
        result = get_session_mean_df(two_sessions(), statistic=statistic, max_syllable=2)
        assert list(result.index) == [("g1", "a"), ("g2", "b")]
