"""What the pipeline asks a file about itself before reading any of it.

Every reader starts here: how big the frames are, how many there are, how
fast they came, and which of an mkv's tracks is the depth one. A wrong
answer from any of these is not a wrong frame, it is a wrong recording.
"""

from __future__ import annotations

import pytest
from moseq2_extract.io.video import (
    get_stream_names,
    get_video_info,
    load_timestamps_from_movie,
)


def test_stream_names_map_titles_to_indices(azure_mkv):
    """An Azure Kinect recording names its tracks and the depth one is not first."""
    assert get_stream_names(str(azure_mkv / "depth.mkv")) == {"COLOR": 0, "DEPTH": 1}


def test_stream_names_fall_back_when_no_titles(untitled_mkv):
    """A file from somewhere else has one track and no name for it.

    The fallback calls it DEPTH, because the caller is about to read depth
    out of it and there is nothing else to read.
    """
    assert get_stream_names(str(untitled_mkv)) == {"DEPTH": 0}


def test_video_info_reads_dimensions_and_rate(raw_sessions):
    info = get_video_info(str(raw_sessions / "saline_example_0" / "depth.avi"))
    assert info["dims"] == (512, 424)
    assert info["fps"] == 30.0
    assert info["nframes"] == 600


def test_video_info_reports_no_frame_count_for_mkv(azure_mkv):
    """Matroska carries a duration, not a frame count.

    ``None`` rather than ``0``: a caller that asks for every frame tests this
    and falls back to the timestamps, where ``0`` would read nothing and
    report success.
    """
    assert get_video_info(str(azure_mkv / "depth.mkv"))["nframes"] is None


def test_timestamps_are_one_per_frame(azure_mkv):
    ts = load_timestamps_from_movie(str(azure_mkv / "depth.mkv"))
    assert len(ts) == 60
    assert ts[0] == 0.0
    assert ts[1] == pytest.approx(1 / 30, abs=1e-3)
