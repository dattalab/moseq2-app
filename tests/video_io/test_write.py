"""The two kinds of file the pipeline writes.

One is for the pipeline to read again: extraction's depth output, which has
to come back bit for bit or every scalar computed from it moves. The other
is for a person to watch in a browser, which is a property of the encoding
rather than of the frames, and the two previews are written by different
packages.

Neither preview puts the index in front of the frame data, so a browser
fetches the whole file before it shows anything. That is how the pipeline
has always written them and these tests do not ask for more.
"""

from __future__ import annotations

import numpy as np
from moseq2_extract.io.video import (
    get_video_info,
    read_frames,
    write_frames,
    write_frames_preview,
)

from video import probe

WIDTH, HEIGHT = 64, 48


def depth_block(frames=5):
    """Frames whose every sample is distinct, so a lossy codec cannot hide."""
    return (np.arange(frames * HEIGHT * WIDTH).reshape(frames, HEIGHT, WIDTH) % 2**16).astype(
        np.uint16
    )


def test_ffv1_round_trip_is_lossless(tmp_path):
    """What extraction writes, read back.

    Depth is distances in millimetres. A codec that rounds them is not a
    codec that stored the recording.
    """
    out = tmp_path / "depth.avi"
    data = depth_block()
    write_frames(str(out), data)
    assert np.array_equal(read_frames(str(out), range(0, len(data))), data)


def test_preview_is_browser_playable(tmp_path):
    """The extraction preview a user opens to see whether extraction worked."""
    out = tmp_path / "preview.mp4"
    write_frames_preview(str(out), depth_block())
    assert probe(out) == ["h264", "yuv420p"]


def test_crowd_movie_is_browser_playable(tmp_path):
    """The crowd movies that show what a syllable looks like.

    moseq2-viz writes these with its own copy of the preview writer, so a
    change to one says nothing about the other.
    """
    from moseq2_viz.io.video import write_frames_preview as write_crowd_movie

    out = tmp_path / "crowd.mp4"
    write_crowd_movie(str(out), depth_block())
    assert probe(out) == ["h264", "yuv420p"]
