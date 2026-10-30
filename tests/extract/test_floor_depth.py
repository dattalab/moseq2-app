"""The floor depth the arena step guesses when the depth range is not set by hand.

``get_roi_wrapper`` finds the centre of the background's far-away region and
reads the floor depth there, then searches for the floor within 50 mm of it.
The depth is the median of the pixels with a reading in a 5 x 5 patch around
the centre, so one missing or odd pixel does not decide it, and a centre with
no reading anywhere near it is reported rather than turned into a range
around 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from moseq2_extract.helpers import wrappers
from moseq2_extract.util import _floor_depth

FLOOR = 700.0


def test_a_missing_centre_pixel_reads_the_floor_around_it():
    background = np.full((40, 40), FLOOR)
    background[20, 20] = 0
    assert _floor_depth(background, 20, 20) == FLOOR


def test_one_odd_pixel_does_not_decide_the_depth():
    background = np.full((40, 40), FLOOR)
    background[20, 20] = 300
    assert _floor_depth(background, 20, 20) == FLOOR


def test_a_centre_with_no_reading_nearby_is_reported():
    background = np.full((40, 40), FLOOR)
    background[18:23, 18:23] = 0
    with pytest.raises(ValueError, match='Set "bg_roi_depth_range" by hand'):
        _floor_depth(background, 20, 20)


def test_the_arena_step_searches_around_the_floor_read_from_the_patch(tmp_path, monkeypatch):
    """``get_roi_wrapper`` sets the search range from the patch, not the centre pixel alone.

    The background is floor everywhere but its centre pixel, which has no
    reading. The step stops once the range is set: loading the first frame
    is replaced by an error.
    """
    background = np.full((41, 41), FLOOR, dtype=np.float32)
    background[20, 20] = 0

    class RangeSet(Exception):
        pass

    def stop(*args, **kwargs):
        raise RangeSet

    monkeypatch.setattr(wrappers, "get_bground_im_file", lambda *args, **kwargs: background)
    monkeypatch.setattr(wrappers, "load_movie_data", stop)
    config = {"finfo": {"dims": (41, 41)}, "camera_type": "kinect", "manual_set_depth_range": False}

    with pytest.raises(RangeSet):
        wrappers.get_roi_wrapper(str(tmp_path / "depth.dat"), config, output_dir=str(tmp_path / "proc"))

    assert config["bg_roi_depth_range"] == [FLOOR - 50, FLOOR + 50]
