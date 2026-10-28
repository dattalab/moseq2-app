"""Files the video tests read, and the two they write to read back.

Every other directory in this suite walks a notebook cell. These tests walk
functions instead: ``moseq2_extract.io.video`` and the one writer in
``moseq2_viz.io.video`` are what every cell that touches a recording goes
through, and a cell that extracts a mouse cannot say which of a reader, a
probe or a writer broke.

The recordings here are deliberately small and synthetic. What a real
recording would add is mouse, and none of these functions know what a mouse
is.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixtures import fetch_azure_mkv

WIDTH, HEIGHT = 64, 48


@pytest.fixture(scope="session")
def azure_mkv():
    """The synthetic Azure Kinect recording, holding ``depth.mkv``."""
    return fetch_azure_mkv()


@pytest.fixture
def untitled_mkv(tmp_path):
    """An mkv with one video track and no ``title`` tag on it.

    Every Azure Kinect recording names its tracks, so the fallback that runs
    when none are named is reached only by a file from somewhere else. It is
    reached, though: ``get_stream_names`` is called on whatever a user points
    the pipeline at.
    """
    import av

    path = tmp_path / "untitled.mkv"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("ffv1", rate=30)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "gray16le"
        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((HEIGHT, WIDTH), np.uint16), format="gray16le"
            )
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
    return path


@pytest.fixture
def sixty_hz_avi(tmp_path):
    """A 60 Hz recording in which every pixel of frame ``i`` equals ``i``.

    The readers take the frame rate as an argument rather than reading it
    from the file, and it defaults to 30. A recording at any other rate is
    the only thing that tells the two apart, and the pixel values say which
    frame came back without reference to anything else.
    """
    import av

    path = tmp_path / "sixty.avi"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("ffv1", rate=60)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "gray16le"
        for i in range(120):
            frame = av.VideoFrame.from_ndarray(
                np.full((HEIGHT, WIDTH), i, np.uint16), format="gray16le"
            )
            out.mux(stream.encode(frame))
        out.mux(stream.encode())
    return path
