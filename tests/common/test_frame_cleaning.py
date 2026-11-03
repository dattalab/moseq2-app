"""The spatial frame cleaning PCA and crowd movies share, in moseq2-common.

PCA and crowd movies each had their own ``clean_frames``, with the same tail
filter, median filter and Gaussian filter, and PCA had a ``select_strel`` that
builds the tail filter's structuring element. Both live in
``moseq2_common.frames`` now. PCA adds its temporal filters after the shared
spatial ones; crowd movies round an even median kernel down before calling it,
as they did before.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from fixtures import fetch_analysis_bundle
from moseq2_common.frames import clean_frames, select_strel


def fixture_block(n=20):
    """The first ``n`` extracted frames of the analysis bundle's first session."""
    h5s = sorted(Path(fetch_analysis_bundle()).glob("aggregate_results/*.h5"))
    with h5py.File(h5s[0], "r") as f:
        return f["frames"][:n]


def test_select_strel_builds_ellipses_and_rectangles():
    np.testing.assert_array_equal(
        select_strel("ellipse", (7, 5)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    )
    np.testing.assert_array_equal(
        select_strel("rectangle", [7, 5]), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    )
    # anything else is an ellipse
    np.testing.assert_array_equal(
        select_strel("disk", (7, 5)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    )


@pytest.mark.parametrize(
    "shape, size",
    [(None, (7, 7)), ("none", (7, 7)), ("", (7, 7)), ("ellipse", (0, 0)), ("ellipse", [0, 0])],
)
def test_select_strel_gives_no_element_for_none_and_zero_size(shape, size):
    assert select_strel(shape, size) is None


def test_clean_frames_applies_median_then_gaussian_as_opencv_does():
    frames = fixture_block()
    expected = np.copy(frames)
    for i in range(len(expected)):
        expected[i] = cv2.medianBlur(expected[i], 3)
        expected[i] = cv2.medianBlur(expected[i], 5)
        expected[i] = cv2.GaussianBlur(expected[i], (21, 21), 1.5, 1)

    out = clean_frames(frames, medfilter_space=[3, 5], gaussfilter_space=[1.5, 1])
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == frames.dtype


def test_the_tail_filter_zeros_pixels_below_threshold():
    frame = np.zeros((1, 40, 40), dtype="uint8")
    frame[0, 5:20, 5:20] = 30  # a body, kept
    frame[0, 25, 5:35] = 30  # a one-pixel tail, opened away
    frame[0, 25:38, 25:38] = 4  # a wide region no higher than the threshold
    out = clean_frames(frame, tail_filter=select_strel("ellipse", (5, 5)), tail_threshold=5)

    assert (out[0, 6:19, 6:19] == 30).all()
    assert (out[0, 25, 5:24] == 0).all()
    assert (out[0, 25:38, 25:38] == 0).all()


def test_clean_frames_leaves_its_input_unchanged():
    frames = fixture_block()
    before = frames.copy()
    clean_frames(
        frames,
        medfilter_space=[3],
        gaussfilter_space=[1.5, 1],
        tail_filter=select_strel("ellipse", (9, 9)),
    )
    np.testing.assert_array_equal(frames, before)


def test_crowd_movies_still_round_an_even_median_kernel_down(tmp_path, monkeypatch):
    """A crowd movie asked for a median kernel of 4 filters with 3, and says so."""
    from moseq2_viz.viz import make_crowd_matrix

    path = tmp_path / "session.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f["frames"] = rng.integers(0, 60, size=(40, 100, 100), dtype="uint8")
        f["scalars/centroid_x_px"] = np.full(40, 50.0)
        f["scalars/centroid_y_px"] = np.full(40, 50.0)
        f["scalars/angle"] = np.zeros(40)
        f.create_group("metadata/extraction")

    kernels = []
    median_blur = cv2.medianBlur

    def record(src, ksize):
        kernels.append(ksize)
        return median_blur(src, ksize)

    monkeypatch.setattr(cv2, "medianBlur", record)
    with pytest.warns(UserWarning, match="Reducing 4 to 3"):
        make_crowd_matrix(
            [((10, 15), "uuid", str(path))],
            nexamples=1,
            pad=2,
            max_dur=10,
            raw_size=(100, 100),
            crop_size=(20, 20),
            medfilter_space=[4],
            gaussfilter_space=[0, 0],
        )
    assert kernels and set(kernels) == {3}


def test_viz_no_longer_offers_clean_frames():
    """Crowd movies' cleaning moved to ``moseq2_common.frames``, and the old path says so."""
    with pytest.raises(ImportError):
        from moseq2_viz.viz import clean_frames  # noqa: F401
