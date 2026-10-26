"""Cell 5: write the frame-by-frame table out as csv.

    save_path = progress_paths['model_session_path']
    filename = 'moseq_df.csv'
    moseq_df.to_csv(join(save_path, filename), index=False)

One pandas call, and the notebook's first handover to anything outside
MoSeq. A user who wants to plot in R, open the data in a spreadsheet or
run their own statistics leaves through this cell.

That makes the size the point rather than an obstacle: twenty sessions of
a twelve-hour experiment is seven hundred thousand rows, a few hundred
megabytes of csv, and whether that writes and reads back intact is the
only thing this cell can get wrong. So the export is performed for real,
once, and the file is deleted afterwards.

``index=False`` is what keeps pandas from writing its row numbers as a
first unnamed column. With it, a reader gets the columns cell 4 built and
nothing else.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

#: A few columns to read back in full: one of each kind the table holds.
#: Reading all thirty-three would double the suite's memory for no more
#: evidence than these give.
SAMPLED = ("uuid", "group", "labels (usage sort)", "velocity_3d_mm", "timestamps")


@pytest.fixture(scope="module")
def exported(analysis_paths, moseq_df):
    """The cell, run for real, cleaned up afterwards."""
    path = Path(analysis_paths["model_session_path"]) / "moseq_df.csv"
    moseq_df.to_csv(path, index=False)
    yield path
    path.unlink(missing_ok=True)


def test_the_file_lands_where_the_cell_prints_that_it_did(exported, analysis_paths):
    """1. Inside the model's own folder, named moseq_df.csv.

    The cell prints this path and a user goes looking for it. Cell 2 put
    ``model_session_path`` inside the model's directory so that two models
    analysed in one project do not overwrite each other's exports.
    """
    assert exported.exists()
    assert exported.parent == Path(analysis_paths["model_session_path"])
    assert exported.name == "moseq_df.csv"


def test_the_whole_table_is_written(exported, moseq_df):
    """2. Every row, not a truncated view.

    Counted a line at a time rather than by reading the file back, which
    at this size costs as much memory again as the table already holds.
    """
    with open(exported, newline="") as f:
        lines = sum(1 for _ in f)
    assert lines == len(moseq_df) + 1, "the export is not the whole table plus a header"


def test_the_header_is_the_columns_cell_4_built(exported, moseq_df):
    """3. ``index=False`` in practice.

    Without it pandas writes its row numbers into a leading unnamed
    column, which every reader then has to know to discard, and which
    silently shifts a positional read by one.
    """
    with open(exported, newline="") as f:
        header = next(csv.reader(f))
    assert header == list(moseq_df.columns)
    assert "" not in header and "Unnamed: 0" not in header


def test_the_values_come_back_as_they_went_out(exported, moseq_df):
    """4. The export is readable by the tool a user will open it with.

    Round-tripping through pandas is the closest check available to
    "someone else can use this": the identifiers have to survive as
    written, the labels as integers, and the measurements to the precision
    extraction recorded them at.
    """
    back = pd.read_csv(exported, usecols=list(SAMPLED))
    assert len(back) == len(moseq_df)

    for column in ("uuid", "group"):
        assert back[column].tolist() == moseq_df[column].tolist()

    assert back["labels (usage sort)"].tolist() == moseq_df["labels (usage sort)"].tolist()

    for column in ("velocity_3d_mm", "timestamps"):
        np.testing.assert_allclose(
            back[column].to_numpy(dtype="float64"),
            moseq_df[column].to_numpy(dtype="float64"),
            rtol=1e-6,
            equal_nan=True,
        )
