"""Cell 4: line the scalars up against the syllables.

    _, sorted_index = parse_index(progress_paths['index_file'])
    moseq_df = scalars_to_dataframe(sorted_index, model_path=progress_paths['model_path'])

The join the rest of the notebook rests on. Extraction measured a mouse
frame by frame: where it was, how fast, how big, which way it faced. The
model, separately, assigned a syllable to each of those frames. This cell
brings the two together into one row per frame, and every question asked
afterwards is a groupby on it: what a group's syllable usage looks like,
how fast the mouse moves during one syllable, which syllables separate
two conditions.

The two halves fail differently. ``parse_index`` resolving the wrong
paths gives an immediate read error. A misaligned join gives a table of
the right shape, full of scalars attributed to the wrong syllable, and
nothing downstream can tell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: The three label columns cell 6 offers as ``syllable_key``. The model
#: numbers its syllables arbitrarily; the two sorted columns renumber them
#: by how much they are used, which is what makes "syllable 0" mean the
#: most common one.
LABEL_COLUMNS = ("labels (original)", "labels (usage sort)", "labels (frames sort)")

#: What extraction measured, as the dataframe names it. Cell 6 averages
#: every one of these within each syllable, and cell 8's fingerprint plots
#: a subset.
SCALARS = (
    "angle",
    "area_mm",
    "area_px",
    "centroid_x_mm",
    "centroid_x_px",
    "centroid_y_mm",
    "centroid_y_px",
    "height_ave_mm",
    "length_mm",
    "length_px",
    "velocity_2d_mm",
    "velocity_2d_px",
    "velocity_3d_mm",
    "velocity_3d_px",
    "velocity_theta",
    "width_mm",
    "width_px",
)

#: The columns that say which recording a row came from.
IDENTITY = ("uuid", "group", "SessionName", "SubjectName", "StartTime", "h5_path")


def test_the_index_resolves_every_session_it_names(sorted_index):
    """1. The project can be read from wherever it was put.

    The index holds each session's paths relative to itself, and
    ``parse_index`` joins them against the index file's own directory. A
    project that was copied or downloaded resolves; one whose index kept
    the absolute paths of the machine it was built on does not, and the
    failure is a missing file several cells later.
    """
    assert sorted_index["files"], "the index names no sessions"
    for uuid, entry in sorted_index["files"].items():
        for path in entry["path"]:
            assert Path(path).is_absolute(), f"{uuid} kept a relative path: {path}"
            assert Path(path).exists(), f"the index names {path}, which is not there"
    assert Path(sorted_index["pca_path"]).exists(), "the scores file the index names is missing"


def test_there_is_one_row_for_every_frame_the_model_labelled(analysis_paths, moseq_df):
    """2. Nothing was dropped and nothing was duplicated.

    The model holds one label array per session. Each of those frames gets
    a row, so the table's length is the total of their lengths. A session
    silently missing here would skew every group statistic downstream
    without changing the shape of anything.
    """
    from moseq2_viz.model.util import parse_model_results

    model = parse_model_results(analysis_paths["model_path"])
    expected = {
        uuid: len(labels)
        for uuid, labels in zip(model["metadata"]["uuids"], model["labels"])
    }

    got = moseq_df.groupby("uuid").size().to_dict()
    assert got == expected
    assert len(moseq_df) == sum(expected.values())


def test_every_session_in_the_index_is_in_the_table(sorted_index, moseq_df):
    """3. The index and the model agree about which recordings these are.

    ``parse_index`` keys the sessions by uuid, which is the identifier the
    model also stores, so this is the join the whole notebook runs on.
    """
    assert set(moseq_df["uuid"]) == set(sorted_index["files"])


def test_each_row_carries_the_group_the_index_assigns_its_session(sorted_index, moseq_df):
    """4. Where every group comparison in the notebook comes from.

    The group is a property of the recording, set by hand in the
    extraction and modeling notebook and stored in the index. It reaches
    the statistics only by being copied onto each of that recording's
    rows here.
    """
    expected = {uuid: entry["group"] for uuid, entry in sorted_index["files"].items()}
    got = moseq_df.groupby("uuid")["group"].unique()

    for uuid, groups in got.items():
        assert list(groups) == [expected[uuid]], f"{uuid} carries {groups}"
    assert len(set(expected.values())) > 1, "every session is in one group; nothing to compare"


def test_the_table_says_which_recording_each_row_came_from(moseq_df):
    """5. The metadata cells 6, 9 and 11 group and label by."""
    missing = [c for c in IDENTITY if c not in moseq_df.columns]
    assert not missing, f"the table does not identify its rows: {missing}"


def test_every_scalar_extraction_measured_is_here_and_numeric(moseq_df):
    """6. Cell 6 averages these within each syllable.

    It picks the columns to average by dtype rather than by name, so a
    scalar that arrived as an object column would be silently left out of
    every summary.
    """
    missing = [c for c in SCALARS if c not in moseq_df.columns]
    assert not missing, f"missing scalars: {missing}"
    not_numeric = [c for c in SCALARS if moseq_df[c].dtype not in (np.float32, np.float64)]
    assert not not_numeric, f"these would be dropped from every summary: {not_numeric}"


def test_the_three_label_columns_cell_6_offers_are_all_here(moseq_df):
    """7. ``syllable_key`` names one of these."""
    missing = [c for c in LABEL_COLUMNS if c not in moseq_df.columns]
    assert not missing, f"missing label columns: {missing}"


def test_the_sorted_labels_are_a_renumbering_of_the_originals(moseq_df):
    """8. Sorting renames syllables, it does not regroup frames.

    "Syllable 3" means a different thing in each column, but the frames
    behind it are the same frames. If the renumbering were not one-to-one,
    two of the model's syllables would be merged and every usage and
    duration computed afterwards would be wrong.
    """
    labelled = moseq_df[moseq_df["labels (original)"] >= 0]
    for column in ("labels (usage sort)", "labels (frames sort)"):
        pairs = labelled.groupby("labels (original)")[column].nunique()
        assert (pairs == 1).all(), f"{column} splits a syllable across several ids"
        assert labelled[column].nunique() == labelled["labels (original)"].nunique()


def test_unlabelled_frames_are_negative_rather_than_missing(moseq_df):
    """9. How cell 6 discards them.

    The model cannot label the first frames of a recording: an
    autoregressive model needs a few frames of history before it has
    anything to condition on. Those frames still carry scalars, so they
    are in the table, marked with a negative label.

    ``compute_behavioral_statistics`` drops them with ``labels (original)
    >= 0``. A comparison against NaN is False, so that query would drop
    every row instead, and the next cell would come back empty.
    """
    for column in LABEL_COLUMNS:
        assert not moseq_df[column].isna().any(), f"{column} holds NaN"
        assert np.issubdtype(moseq_df[column].dtype, np.integer), f"{column} is not integer"

    unlabelled = moseq_df["labels (original)"] < 0
    assert unlabelled.any(), "no frame is unlabelled; the padding convention has changed"
    assert (moseq_df.loc[unlabelled, "labels (usage sort)"] < 0).all()
