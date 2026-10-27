"""Cell 7: write the syllable statistics out as csv.

    save_path = progress_paths['model_session_path']
    filename = 'stats_df.csv'
    stats_df.to_csv(join(save_path, filename), index=False)

The same export as cell 5, of the table a user is far more likely to open:
a thousand rows of per-session, per-syllable statistics rather than
hundreds of thousands of frames. This is what goes into a figure or a
statistics package.

Cell 5's tests cover what the two exports share. What is different here
is the content. ``stats_df`` has gaps in it by design, where a session
never performed a syllable and so has no duration to report, and csv
represents a gap as an empty field. If that came back as zero, every
average duration computed outside MoSeq would be pulled toward it, and
the file would look complete.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype


@pytest.fixture(scope="module")
def exported(analysis_paths, stats_df):
    """The cell, run for real, cleaned up afterwards."""
    path = Path(analysis_paths["model_session_path"]) / "stats_df.csv"
    stats_df.to_csv(path, index=False)
    yield path
    path.unlink(missing_ok=True)


def test_the_file_lands_beside_the_other_export_under_its_own_name(exported, analysis_paths):
    """1. Two exports, one folder, no collision.

    Cells 5 and 7 both write into ``model_session_path``. They are told
    apart by filename alone, so the names have to differ.
    """
    assert exported.exists()
    assert exported.parent == Path(analysis_paths["model_session_path"])
    assert exported.name == "stats_df.csv"
    assert exported.name != "moseq_df.csv", "cell 7 would overwrite cell 5's export"


def test_the_whole_table_is_written_with_its_own_columns(exported, stats_df):
    """2. Every row and every column, and no pandas row numbers."""
    with open(exported, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == list(stats_df.columns)
    assert "" not in rows[0] and "Unnamed: 0" not in rows[0]
    assert len(rows) == len(stats_df) + 1


def test_the_statistics_come_back_as_they_went_out(exported, stats_df):
    """3. A round trip through the reader a user will reach for.

    Small enough to read back whole, unlike cell 5's export, so every
    column is compared rather than a sample.
    """
    back = pd.read_csv(exported)
    assert list(back.columns) == list(stats_df.columns)
    assert len(back) == len(stats_df)

    for column in stats_df.columns:
        # Text columns are compared as text. pandas 2 gives them the object
        # dtype and pandas 3 a string dtype of its own, so the question asked
        # here is whether the column is numeric rather than what text looks
        # like.
        if not is_numeric_dtype(stats_df[column]):
            assert back[column].tolist() == stats_df[column].tolist(), column
        else:
            np.testing.assert_allclose(
                back[column].to_numpy(dtype="float64"),
                stats_df[column].to_numpy(dtype="float64"),
                rtol=1e-6,
                equal_nan=True,
                err_msg=column,
            )


def test_a_syllable_that_never_happened_stays_blank_rather_than_zero(exported, stats_df):
    """4. The gaps survive the export.

    A syllable a session never performed has a usage of zero and no
    duration at all, which cell 6 keeps apart. csv writes the missing
    duration as an empty field; a zero written there would read as a bout
    of no length and would be averaged in by anything downstream.
    """
    missing = stats_df["duration"].isna()
    assert missing.any(), "no session is missing a duration; nothing to check"

    back = pd.read_csv(exported)
    assert back["duration"].isna().tolist() == missing.tolist()
    assert not (back.loc[missing, "duration"] == 0).any()
