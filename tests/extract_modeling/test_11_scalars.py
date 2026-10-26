"""Cell 11: look at the measurements before modeling anything.

    from moseq2_app.main import interactive_scalar_summary

    viewer = interactive_scalar_summary(progress_paths['index_file'])

The last check before PCA. It reads every recording's scalars through the
index, reduces each to a mean and a standard deviation, and draws them as
violins split by the groups cell 10 assigned, so a recording that sits apart
from its condition is visible before it is modelled.

The viewer is the one thing in the notebook that is kept rather than
displayed: cell 12 reads ``viewer.sorted_index`` out of it. That makes this
cell's return value part of the surface, not an implementation detail.

Seven measurements are offered and two are shown by default. Six come
straight from the extraction; ``dist_to_center_px`` is derived here from the
centroid and the arena, so it exists in this dataframe and in no h5.
"""

from __future__ import annotations

import pytest

from moseq2_app.main import interactive_scalar_summary

from tests.extract_modeling.test_05_arena_mask import RECORDINGS
from tests.extract_modeling.test_09_aggregate import entries

#: What a user can plot, in the order offered.
OFFERED = [
    "area_mm",
    "height_ave_mm",
    "length_mm",
    "velocity_2d_mm",
    "velocity_3d_mm",
    "width_mm",
    "dist_to_center_px",
]

#: What is plotted before anyone touches the selector.
SHOWN_BY_DEFAULT = ["area_mm", "velocity_2d_mm"]

#: Frames per recording, so the dataframe's length is predictable.
FRAMES = 600


def test_the_cell_displays_its_controls_and_hands_back_the_viewer(grouped, displayed):
    """1. Both halves of what the cell produces.

    Cell 12 reads the viewer, so returning it matters as much as drawing it.
    """
    viewer = interactive_scalar_summary(str(grouped))

    assert viewer is not None
    assert viewer.ui_tools in displayed.objects


def test_a_missing_index_is_reported_rather_than_raised(tmp_path, capsys):
    """2. The cell is reached with a path a user typed, so it can be wrong."""
    assert interactive_scalar_summary(str(tmp_path / "nope.yaml")) is None
    assert "Index file does not exist" in capsys.readouterr().out


def test_the_measurements_on_offer(scalar_viewer):
    """3. What a user can choose between, and what they get without choosing."""
    assert list(scalar_viewer.checked_list.options) == OFFERED
    assert list(scalar_viewer.checked_list.value) == SHOWN_BY_DEFAULT


def test_every_frame_of_every_recording_is_read(scalar_viewer):
    """4. The dataframe behind the plots.

    One row per frame per recording, carrying the group so the violins can be
    split, and the uuid so a point on a plot can be traced back.
    """
    df = scalar_viewer.scalar_df

    assert len(df) == len(RECORDINGS) * FRAMES
    assert sorted(df["SessionName"].unique()) == RECORDINGS
    assert sorted(df["group"].unique()) == ["amph", "saline"]
    assert df["uuid"].nunique() == len(RECORDINGS)
    for column in OFFERED:
        assert column in df.columns


def test_each_recording_becomes_one_point_per_measurement(scalar_viewer):
    """5. What the violins are made of.

    A recording contributes its mean, not its frames, so a long recording
    does not outweigh a short one.
    """
    for frame in (scalar_viewer.mean_df, scalar_viewer.std_df):
        assert len(frame) == len(RECORDINGS)
        assert sorted(frame["SessionName"]) == RECORDINGS
        assert sorted(frame["group"]) == ["amph", "amph", "saline", "saline"]


def test_the_groups_come_from_the_index(scalar_viewer, grouped):
    """6. What cell 10 did is what this cell splits on."""
    from_index = {
        entry["metadata"]["SessionName"]: entry["group"]
        for entry in entries(grouped).values()
    }
    from_viewer = dict(zip(scalar_viewer.mean_df["SessionName"], scalar_viewer.mean_df["group"]))
    assert from_viewer == from_index


def test_the_plot_shows_one_violin_per_group_per_measurement(scalar_viewer):
    """7. The cell's product.

    Two columns, one row per chosen measurement, and within each a violin per
    condition, which is the comparison the plot exists to make.
    """
    scalar_viewer.make_graphs()
    traces = scalar_viewer.fig.data

    assert {trace.type for trace in traces} == {"violin"}
    assert sorted({trace.name for trace in traces}) == ["amph", "saline"]
    assert len(traces) == len(SHOWN_BY_DEFAULT) * 2 * 2


def test_choosing_more_measurements_adds_rows(scalar_viewer):
    """8. The selector drives the figure."""
    scalar_viewer.checked_list.value = ["area_mm", "length_mm", "width_mm"]
    try:
        scalar_viewer.make_graphs()
        assert len(scalar_viewer.fig.data) == 3 * 2 * 2
    finally:
        scalar_viewer.checked_list.value = SHOWN_BY_DEFAULT
        scalar_viewer.make_graphs()


def test_the_viewer_carries_the_index_for_the_next_cell(scalar_viewer, grouped):
    """9. The handover to cell 12.

    ``sorted_index['files']`` is keyed by uuid rather than listed, which is
    what makes a lookup by uuid possible at all.
    """
    files = scalar_viewer.sorted_index["files"]

    assert sorted(files) == sorted(entry["uuid"] for entry in entries(grouped).values())
    for entry in files.values():
        assert {"group", "metadata", "path"} <= set(entry)
