"""The background image extraction subtracts each frame from.

``get_bground_im_file`` takes the per-pixel median of frames sampled through
the recording. A depth of 0 is the camera's "no reading" (every supported
camera has a minimum range of 0.25 m or more), so a pixel's median is taken
over the frames that observed it, and a pixel no sampled frame observed stays
at 0.
"""

from __future__ import annotations

import numpy as np

from moseq2_extract.extract.proc import get_bground_im_file

FLOOR = 700


def background(tmp_path, frames):
    path = tmp_path / "depth.dat"
    frames.astype("<u2").tofile(path)
    finfo = {"nframes": len(frames), "dims": (frames.shape[2], frames.shape[1])}
    return get_bground_im_file(str(path), frame_stride=1, med_scale=1,
                               output_dir=str(tmp_path), finfo=finfo)


def test_a_pixel_missing_in_most_frames_takes_the_depth_it_was_seen_at(tmp_path):
    frames = np.full((3, 8, 8), FLOOR)
    frames[:2, 2:6, 2:6] = 0
    np.testing.assert_array_equal(background(tmp_path, frames), FLOOR)


def test_a_pixel_no_frame_observed_stays_at_zero(tmp_path):
    frames = np.full((3, 8, 8), FLOOR)
    frames[:, 0, 0] = 0
    result = background(tmp_path, frames)
    assert result[0, 0] == 0
    assert (result[1:, 1:] == FLOOR).all()
