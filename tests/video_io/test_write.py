"""The two kinds of file the pipeline writes.

One is for the pipeline to read again: extraction's depth output, which has
to come back bit for bit or every scalar computed from it moves. The other
is for a person to watch in a browser, which is a property of the encoding
rather than of the frames: the extraction preview, the flip widget's
preview and the crowd movies, all written by the one writer in
``moseq2_common.video``.
"""

from __future__ import annotations

import av
import pytest
import numpy as np
from moseq2_common.video import write_frames_preview
from moseq2_extract.io.video import read_frames, write_frames

from video import probe

WIDTH, HEIGHT = 64, 48


def depth_block(frames=5, height=HEIGHT, width=WIDTH):
    """Frames whose every sample is distinct, so a lossy codec cannot hide."""
    return (np.arange(frames * height * width).reshape(frames, height, width) % 2**16).astype(
        np.uint16
    )


def decoded(path):
    """Every frame of a movie, as ``(height, width, 3)`` uint8 arrays."""
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def top_level_boxes(path):
    """The names of an mp4's top-level boxes, in file order."""
    data = path.read_bytes()
    names, pos = [], 0
    while pos + 8 <= len(data):
        size = int.from_bytes(data[pos : pos + 4], "big")
        names.append(data[pos + 4 : pos + 8].decode("latin-1"))
        if size == 1:
            size = int.from_bytes(data[pos + 8 : pos + 16], "big")
        if size == 0:
            break
        pos += size
    return names


def test_ffv1_round_trip_is_lossless(tmp_path):
    """What extraction writes, read back.

    Depth is distances in millimetres. A codec that rounds them is not a
    codec that stored the recording.
    """
    out = tmp_path / "depth.avi"
    data = depth_block()
    write_frames(str(out), data)
    assert np.array_equal(read_frames(str(out), range(0, len(data))), data)


def test_odd_frame_dimensions_are_padded_to_even(tmp_path):
    """H.264 cannot encode an odd width or height, so a row and a column are added."""
    out = tmp_path / "odd.mp4"
    write_frames_preview(str(out), depth_block(height=47, width=63))
    assert probe(out, ("width", "height")) == [64, 48]


def test_a_size_pair_opens_the_file_before_frames_arrive(tmp_path):
    """Extraction opens the preview from its frame size, then writes chunk by chunk."""
    out = tmp_path / "chunks.mp4"
    pipe = write_frames_preview(str(out), (WIDTH, HEIGHT), close_pipe=False)
    for _ in range(2):
        pipe = write_frames_preview(str(out), depth_block(), pipe=pipe, close_pipe=False)
    pipe.communicate()
    assert probe(out, ("width", "height")) == [WIDTH, HEIGHT]
    assert len(decoded(out)) == 10


def test_frame_numbers_and_text_overlays_are_drawn(tmp_path):
    """Extraction numbers each frame; crowd movies write one caption on every frame.

    The overlay is white on a colormap whose zero is dark blue, so the frames
    that carry one hold near-white pixels that the plain frames do not.
    """
    frames = np.zeros((3, HEIGHT, WIDTH), np.uint16)
    plain, numbered, captioned = (tmp_path / f"{n}.mp4" for n in ("plain", "numbered", "captioned"))
    write_frames_preview(str(plain), frames)
    write_frames_preview(str(numbered), frames, frame_numbers=[7, 8, 9])
    write_frames_preview(str(captioned), frames, text="12")

    def white_pixels(path):
        return (decoded(path).min(axis=-1) > 200).sum(axis=(1, 2))

    assert (white_pixels(plain) == 0).all()
    assert (white_pixels(numbered) > 0).all()
    assert (white_pixels(captioned) > 0).all()


def test_the_file_is_h264_yuv420p_with_faststart(tmp_path):
    """What a browser needs to play the movie, and to start before it has all of it.

    Browser H.264 is 8-bit 4:2:0 only, and the mp4 index (``moov``) has to
    come before the frames (``mdat``) for playback to start while the file
    is still arriving.
    """
    out = tmp_path / "preview.mp4"
    write_frames_preview(str(out), depth_block())
    assert probe(out) == ["h264", "yuv420p"]
    boxes = top_level_boxes(out)
    assert boxes.index("moov") < boxes.index("mdat")


def test_closing_returns_the_filename(tmp_path):
    """Crowd movies are listed by the path the writer hands back."""
    out = tmp_path / "crowd.mp4"
    assert write_frames_preview(str(out), depth_block(), text="1") == str(out)


def test_the_old_frame_range_keyword_is_refused_by_name(tmp_path):
    """1.5.0's extraction preview took ``frame_range``; the shared writer calls it ``frame_numbers``.

    The writer ignores keywords it does not know, so a script that kept the
    old name would get a preview without its frame numbers and no message.
    """
    with pytest.raises(TypeError, match="frame_numbers"):
        write_frames_preview(str(tmp_path / "fr.mp4"), depth_block(3), frame_range=[1, 2, 3])
    assert not (tmp_path / "fr.mp4").exists()


def test_viz_no_longer_offers_the_preview_writer():
    """The crowd-movie writer moved to ``moseq2_common.video``, and the old path says so."""
    with pytest.raises(ImportError):
        from moseq2_viz.io.video import write_frames_preview  # noqa: F401
