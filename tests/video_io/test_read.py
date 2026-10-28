"""Getting frames back out of a recording.

Three containers reach these functions: the ``.avi`` a Kinect v2 rig writes,
the ``.dat`` of raw frames some rigs write instead, and the ``.mkv`` an Azure
Kinect writes. The first two hold depth at native byte order in a single
track; the third holds several tracks and stores depth big-endian.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from moseq2_extract.io.video import read_frames, read_mkv


def depth_avi(raw_sessions):
    return str(raw_sessions / "saline_example_0" / "depth.avi")


def test_read_frames_returns_the_requested_frames(raw_sessions):
    frames = read_frames(depth_avi(raw_sessions), range(10, 20))
    assert frames.shape == (10, 424, 512)
    assert frames.dtype == np.uint16


def test_read_frames_is_stable_across_calls(raw_sessions):
    """Two reads of one range agree.

    The reader seeks by converting a frame index to a time, which is the
    kind of arithmetic that lands a frame either side of where it meant to.
    """
    first = read_frames(depth_avi(raw_sessions), range(0, 5))
    assert np.array_equal(first, read_frames(depth_avi(raw_sessions), range(0, 5)))


def test_read_frames_starts_where_it_was_asked_to(raw_sessions):
    """Frame 10 of a read starting at 0 is frame 0 of a read starting at 10."""
    whole = read_frames(depth_avi(raw_sessions), range(0, 20))
    later = read_frames(depth_avi(raw_sessions), range(10, 20))
    assert np.array_equal(whole[10:], later)


def test_read_frames_seeks_by_the_recordings_own_rate(sixty_hz_avi):
    """A recording at 60 Hz, read without being told its rate.

    Every pixel of frame ``i`` is ``i``, so what came back says which frame
    it was. Dividing by 30 when the file runs at 60 lands on frame 20.
    """
    frames = read_frames(str(sixty_hz_avi), range(10, 15))
    assert frames[0, 0, 0] == 10


def test_read_frames_addresses_a_stream_that_starts_late(offset_mkv):
    """Frame 0 is the recording's first frame, not its clock's origin.

    Worked out from the rate the stream declares, frame 3 of a recording
    beginning two frames in is frame 1 of the recording.
    """
    frames = read_frames(str(offset_mkv), range(3, 6), mapping=0)
    assert [int(frame[0, 0]) for frame in frames] == [3, 4, 5]


def test_read_mkv_returns_the_depth_stream(azure_mkv):
    frames = read_mkv(str(azure_mkv / "depth.mkv"), range(0, 10))
    assert frames.shape == (10, 48, 64)


def test_read_mkv_reads_the_big_endian_depth_samples(azure_mkv):
    """The fixture's samples are ``frame * 1000 + row * 100 + column``.

    Read at the wrong byte order this is 34060 rather than 3205, which is
    not a frame a mouse was ever in.
    """
    frames = read_mkv(str(azure_mkv / "depth.mkv"), range(0, 10))
    assert frames[3, 2, 5] == 3205


def test_read_mkv_does_not_return_the_colour_stream(azure_mkv):
    """COLOR is stream 0 and flat; DEPTH is stream 1 and is not.

    Whatever byte order the samples come back in, a flat track has one value
    in it and the depth track has many.
    """
    frames = read_mkv(str(azure_mkv / "depth.mkv"), range(0, 2))
    assert len(np.unique(frames[0])) == 48 * 64


def test_reading_past_the_end_raises(raw_sessions):
    """The recording has 600 frames and the caller asked for frames 595 to 609.

    Fewer frames arrive than were asked for, and what the caller gets is an
    exception rather than a short array it would index off the end of.
    """
    with pytest.raises(ValueError):
        read_frames(depth_avi(raw_sessions), range(595, 610))


def test_reading_a_missing_file_raises(tmp_path):
    """Nothing to probe, so nothing to read.

    The probe runs first and has no dimensions to report, which is where
    this stops rather than somewhere further down holding an empty array.
    """
    with pytest.raises(ValueError):
        read_frames(str(tmp_path / "nope.avi"), range(0, 2))


def test_reading_a_truncated_file_raises(tmp_path, raw_sessions):
    """A recording that probes fine and then runs out part way through.

    Fewer frames arrive than were asked for, which stops here rather than
    handing back a short array a caller would index off the end of.
    """
    truncated = tmp_path / "truncated.avi"
    truncated.write_bytes(Path(depth_avi(raw_sessions)).read_bytes()[:20_000])
    with pytest.raises(ValueError):
        read_frames(str(truncated), range(0, 2))
