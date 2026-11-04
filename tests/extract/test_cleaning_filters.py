"""The cable and tail filters in extraction's frame cleaning.

``clean_frames`` erodes each frame ``iters_min`` times with ``strel_min``
(the cable filter, for thinning a fibre or tether out of the frame) and
opens it ``iters_tail`` times with ``strel_tail`` (the tail filter).
Extraction sets the two counts from ``cable_filter_iters`` and
``tail_filter_iters``; both are off or at 1 in a generated config.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import moseq2_extract.extract.extract as extract_module
from moseq2_extract.extract.proc import clean_frames

SQUARE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

#: The settings extract_chunk takes from its caller, which has no defaults for them.
CHUNK_SETTINGS = dict(
    use_tracking_model=False, spatial_filter_size=(3,), temporal_filter_size=None, tail_filter_iters=1,
    cable_filter_iters=0, min_height=10, max_height=100, use_cc=False, tracking_model_segment=True,
    flip_classifier=None, flip_classifier_smoothing=51, frame_dtype="uint8", progress_bar=True,
    crop_size=(80, 80), centroid_hampel_span=5, centroid_hampel_sig=3, angle_hampel_span=5,
    angle_hampel_sig=3, model_smoothing_clips=(-300, -150), tracking_model_init="raw",
    compute_raw_scalars=False,
)


def body_with_tail():
    """A 20 x 20 body with a tail 4 pixels wide and 12 long."""
    frames = np.zeros((1, 41, 41), np.uint8)
    frames[0, 5:25, 5:25] = 100
    frames[0, 13:17, 25:37] = 100
    return frames


def test_more_cable_filter_iterations_erode_more():
    frames = body_with_tail()
    once = clean_frames(frames, prefilter_space=None, iters_min=1, strel_min=SQUARE)
    three = clean_frames(frames, prefilter_space=None, iters_min=3, strel_min=SQUARE)
    assert (three > 0).sum() < (once > 0).sum()


def test_more_tail_filter_iterations_open_more():
    """One opening with a 3 x 3 element keeps a 4-pixel tail; two remove it."""
    frames = body_with_tail()
    tail = (0, slice(13, 17), slice(27, 37))
    once = clean_frames(frames, prefilter_space=None, iters_tail=1, strel_tail=SQUARE)
    twice = clean_frames(frames, prefilter_space=None, iters_tail=2, strel_tail=SQUARE)
    assert once[tail].any()
    assert not twice[tail].any()


def test_the_tracking_models_first_mask_opens_as_often_as_asked():
    """``em_init`` opens the first mask ``strel_iters`` times; nothing passes more than its default 1."""
    from moseq2_extract.extract.track import em_init

    depth = body_with_tail()[0].astype(float)
    once = em_init(depth, 50, 150, init_strel=SQUARE, strel_iters=1)
    twice = em_init(depth, 50, 150, init_strel=SQUARE, strel_iters=2)
    assert once[13:17, 27:37].any()
    assert not twice[13:17, 27:37].any()


def test_extraction_hands_both_filters_their_iterations(monkeypatch):
    """The config's names reach ``clean_frames``, the way extraction passes the whole config."""
    seen = {}

    class Stop(Exception):
        pass

    def record(*args, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(extract_module, "clean_frames", record)
    with pytest.raises(Stop):
        extract_module.extract_chunk(
            np.zeros((2, 20, 20), np.uint8),
            **{**CHUNK_SETTINGS, "cable_filter_iters": 2, "tail_filter_iters": 3, "progress_bar": False},
        )
    assert (seen["iters_min"], seen["iters_tail"]) == (2, 3)


#: Extraction's default filter shapes and sizes, as a generated config.yaml has them.
STREL_SETTINGS = dict(
    bg_roi_shape="ellipse", bg_roi_dilate=(10, 10), bg_roi_erode=(1, 1),
    tail_filter_shape="ellipse", tail_filter_size=(9, 9),
    cable_filter_shape="rectangle", cable_filter_size=(5, 5),
)


def test_a_none_tail_filter_shape_turns_the_tail_filter_off():
    """``tail_filter_shape: none`` leaves the frames as the tail filter found them.

    A default 9 x 9 ellipse removes the 4-pixel tail; with the shape ``none``
    there is no element, and ``clean_frames`` skips the opening even though
    ``tail_filter_iters`` asks for one.
    """
    from moseq2_extract.util import get_strels

    frames = body_with_tail()
    strels = get_strels({**STREL_SETTINGS, "tail_filter_shape": "none"})
    assert strels["strel_tail"] is None
    out = clean_frames(frames, prefilter_space=None, iters_tail=1, strel_tail=strels["strel_tail"])
    np.testing.assert_array_equal(out, frames)


def test_a_zero_size_cable_filter_turns_the_cable_filter_off():
    """``cable_filter_size: [0, 0]`` gives no element, and the erosion is skipped."""
    from moseq2_extract.util import get_strels

    frames = body_with_tail()
    strels = get_strels({**STREL_SETTINGS, "cable_filter_size": (0, 0)})
    assert strels["strel_min"] is None
    out = clean_frames(frames, prefilter_space=None, iters_min=2, strel_min=strels["strel_min"])
    np.testing.assert_array_equal(out, frames)


def test_a_none_roi_shape_does_not_dilate_the_graduated_background(monkeypatch, tmp_path):
    """With ``bg_roi_shape: none`` there is no dilation element, so graduating the
    walls skips the background's dilation rather than handing OpenCV no kernel,
    which it would replace with a 3 x 3 square."""
    import moseq2_extract.util as util

    class Stop(Exception):
        pass

    dilations = []
    monkeypatch.setattr(util.cv2, "dilate", lambda *args, **kwargs: dilations.append(args))
    monkeypatch.setattr(util, "get_bucket_center", lambda *args, **kwargs: (_ for _ in ()).throw(Stop()))
    strels = util.get_strels({**STREL_SETTINGS, "bg_roi_shape": "none"})
    assert strels["strel_dilate"] is None
    with pytest.raises(Stop):
        util.graduate_dilated_wall_area(
            np.full((20, 20), 670.0, np.float32), {"true_depth": 670.0, "dilate_iterations": 2},
            strels["strel_dilate"], str(tmp_path),
        )
    assert dilations == []
