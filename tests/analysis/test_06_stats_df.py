"""Cell 6: collapse the frames into syllable statistics.

    count = 'usage'
    syllable_key = 'labels (usage sort)'   # either 'labels (usage sort)', 'labels', or 'labels (frames sort)'
    groupby = ['group', 'uuid']
    usage_normalization = True
    stats_df = compute_behavioral_statistics(moseq_df, count=count, groupby=groupby,
                                             usage_normalization=usage_normalization,
                                             syllable_key=syllable_key)

Cell 4's table has a row per frame, which is not a unit anyone compares.
This cell turns it into the one they do: a session and a syllable. How
much of that session was this syllable, how long its bouts ran, and what
the mouse's speed, size and posture were while it was performing it.

Every remaining cell reads this table. The dendrogram clusters its
scalars, the fingerprint plots them, the hypothesis tests in cell 11
compare its usages between groups, and cells 11 and 14 cache it to
parquet.

``count`` is the one real choice. ``usage`` counts how often a syllable
*started*, ``frames`` how many frames it occupied, and they rank syllables
differently: a rare syllable that runs long scores low on one and high on
the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from moseq2_viz.model.util import compute_behavioral_statistics

from tests.conftest import STATISTICS

#: The columns that identify a row, and the two statistics the cell exists
#: to produce.
CORE = ("group", "uuid", "syllable", "usage", "duration")

#: Extraction's frame rate. ``compute_behavioral_statistics`` divides bout
#: lengths by it, which is what puts ``duration`` in seconds.
FPS = 30


def test_there_is_a_row_for_every_session_and_every_syllable(moseq_df, stats_df):
    """1. A complete grid, including the syllables a session never used.

    The group comparisons in cell 11 line sessions up syllable by
    syllable. A session that simply had no row for a syllable it never
    performed would be dropped from that syllable's comparison, quietly
    shrinking the sample for exactly the syllables where a group
    difference is most likely.
    """
    missing = [c for c in CORE if c not in stats_df.columns]
    assert not missing, f"the table is missing {missing}"

    sessions = moseq_df["uuid"].nunique()
    syllables = stats_df["syllable"].nunique()
    assert len(stats_df) == sessions * syllables

    counts = stats_df.groupby("uuid")["syllable"].nunique()
    assert (counts == syllables).all(), "some sessions are missing syllables"


def test_the_unlabelled_frames_are_gone(stats_df):
    """2. The padding at the start of each recording is discarded.

    ``compute_behavioral_statistics`` drops it with ``labels (original)
    >= 0``. Left in, it would appear as an extra syllable, present in
    every session and used by none of them.
    """
    assert stats_df["syllable"].min() >= 0


def test_usage_is_a_share_of_the_session(stats_df):
    """3. What ``usage_normalization`` buys.

    Recordings differ in length, so a raw count says as much about how
    long the camera ran as about the mouse. Normalized, each session's
    usages are a distribution over syllables and two sessions can be
    compared directly.
    """
    totals = stats_df.groupby("uuid")["usage"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-9)
    assert (stats_df["usage"] >= 0).all()


def test_counting_onsets_and_counting_frames_are_different_questions(moseq_df, stats_df):
    """4. The ``count`` parameter is a real choice, not a formatting one.

    ``usage`` counts bouts, ``frames`` counts frames. A syllable the mouse
    enters rarely but stays in for a long time is uncommon by the first
    measure and common by the second, so the two rank syllables
    differently and a user picking one is choosing what "used" means.
    """
    by_frames = compute_behavioral_statistics(moseq_df, **{**STATISTICS, "count": "frames"})

    totals = by_frames.groupby("uuid")["usage"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-9)

    merged = stats_df.merge(by_frames, on=["uuid", "syllable"], suffixes=("_onsets", "_frames"))
    assert len(merged) == len(stats_df)
    assert not np.allclose(
        merged["usage_onsets"].to_numpy(), merged["usage_frames"].to_numpy()
    ), "counting onsets and counting frames gave the same answer"


def test_an_unsupported_count_is_refused(moseq_df):
    """5. A typo in the one parameter with two legal values.

    The cell's comment says usage or frames. Anything else raises here
    rather than being silently treated as one of them.
    """
    with pytest.raises(ValueError, match="usage"):
        compute_behavioral_statistics(moseq_df, **{**STATISTICS, "count": "Usage"})


def test_durations_are_in_seconds(stats_df):
    """6. The unit a user reads off the dendrogram and the stats widget.

    Bout lengths are counted in frames and divided by the frame rate, so
    the shortest possible bout is one frame, a thirtieth of a second. A
    duration column still in frames would be thirty times too large
    everywhere and nothing would flag it.
    """
    durations = stats_df["duration"].dropna()
    assert len(durations) > 0
    assert durations.min() == pytest.approx(1 / FPS)
    assert durations.max() < 60, "a syllable lasting a minute suggests frames, not seconds"


def test_a_syllable_a_session_never_performed_has_no_duration(stats_df):
    """7. Where the gaps in this table are, and why.

    The grid is completed by filling missing usages with zero. A duration
    cannot be filled the same way: a syllable that never occurred has no
    bout to measure, and a zero there would read as a bout of no length
    and drag every average down. It is left missing instead, so anything
    averaging durations has to skip it.
    """
    never_used = stats_df["usage"] == 0
    assert never_used.any(), "every session used every syllable; nothing to check"
    assert stats_df.loc[never_used, "duration"].isna().all()
    assert stats_df.loc[~never_used, "duration"].notna().all()


def test_every_scalar_arrives_summarised_four_ways(moseq_df, stats_df):
    """8. What makes a syllable describable rather than just countable.

    Each of cell 4's numeric columns is reduced within each session and
    syllable to its mean, spread and range. This is what cell 8's
    fingerprint plots and what a user reads to find out what a syllable
    physically is: this one is fast, this one is a rear.
    """
    numeric = [
        c
        for c in moseq_df.columns
        if moseq_df[c].dtype in (np.float32, np.float64)
    ]
    assert numeric, "cell 4 produced no numeric columns"

    for column in numeric:
        for statistic in ("mean", "std", "min", "max"):
            assert f"{column}_{statistic}" in stats_df.columns, (
                f"{column} was not summarised by {statistic}"
            )


def test_the_table_records_which_labelling_it_was_built_from(stats_df):
    """9. ``syllable`` is a renamed column, and the name is kept.

    The cell renames whichever label column it was given to ``syllable``,
    so by itself the result does not say whether syllable 0 is the most
    used, the most frequent by frames, or the model's own numbering.
    Cells 9, 11 and 14 cache this table to parquet and read it back, which
    is where a mismatch would otherwise go unnoticed.
    """
    assert stats_df["syllable key"].unique().tolist() == [STATISTICS["syllable_key"]]


@pytest.mark.xfail(
    strict=True,
    reason="the cell offers 'labels' as a syllable_key, but scalars_to_dataframe "
    "names that column 'labels (original)'. A user who takes the comment at its "
    "word gets KeyError: 'Column not found: labels'.",
)
def test_the_unsorted_labelling_the_cell_offers_works(moseq_df):
    """10. One of the three options in the cell's comment does not exist.

    The other two name real columns. This one names the model's own
    syllable numbering, which cell 4 produces under a different name, so
    the only way to ask for it is to know that the comment is wrong.
    """
    compute_behavioral_statistics(moseq_df, **{**STATISTICS, "syllable_key": "labels"})
