"""The spatial frame cleaning PCA and crowd movies share, in moseq2-common.

PCA and crowd movies each had their own ``clean_frames``, with the same tail
filter, median filter and Gaussian filter, and PCA had a ``select_strel`` that
builds the tail filter's structuring element. Both live in
``moseq2_common.frames`` now. PCA adds its temporal filters after the shared
spatial ones.

A median filter takes only odd kernel sizes. Extraction, PCA, crowd movies
and flip training round an even size up to the next odd one, through
``moseq2_common.frames.odd_kernel``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from fixtures import fetch_analysis_bundle, fetch_raw_sessions
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


def recorded_median_kernels(monkeypatch):
    """Every kernel size OpenCV's and scipy's median filters are handed, in a list that fills as they run."""
    import scipy.signal

    kernels = []
    median_blur, medfilt = cv2.medianBlur, scipy.signal.medfilt

    def blur(src, ksize):
        kernels.append(ksize)
        return median_blur(src, ksize)

    def filt(volume, kernel_size=None):
        kernels.append(int(np.atleast_1d(kernel_size)[0]))
        return medfilt(volume, kernel_size)

    monkeypatch.setattr(cv2, "medianBlur", blur)
    monkeypatch.setattr(scipy.signal, "medfilt", filt)
    return kernels


def crowd_movie_session(path):
    """A 40-frame session with the scalars a crowd movie reads, written to ``path``."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f["frames"] = rng.integers(0, 60, size=(40, 100, 100), dtype="uint8")
        f["scalars/centroid_x_px"] = np.full(40, 50.0)
        f["scalars/centroid_y_px"] = np.full(40, 50.0)
        f["scalars/angle"] = np.zeros(40)
        f.create_group("metadata/extraction")


def clean_for_pca(tmp_path):
    from moseq2_pca.util import clean_frames as clean_frames_pca

    frames = np.random.default_rng(0).integers(0, 60, size=(12, 20, 20)).astype("float32")
    clean_frames_pca(frames, medfilter_space=[4], medfilter_time=[4])


def clean_for_a_crowd_movie(tmp_path):
    from moseq2_viz.viz import make_crowd_matrix

    crowd_movie_session(tmp_path / "session.h5")
    make_crowd_matrix(
        [((10, 15), "uuid", str(tmp_path / "session.h5"))],
        nexamples=1,
        pad=2,
        max_dur=10,
        raw_size=(100, 100),
        crop_size=(20, 20),
        medfilter_space=[4],
        gaussfilter_space=[0, 0],
    )


def clean_for_flip_training(tmp_path):
    import pickle

    from moseq2_app.flip.train import create_training_dataset
    from moseq2_common.frames import CleaningOptions

    crowd_movie_session(tmp_path / "session.h5")
    with open(tmp_path / "flip-training-frame-ranges.p", "wb") as f:
        pickle.dump(({"s": str(tmp_path / "session.h5")}, {"s": [(False, slice(0, 10))]}), f)
    create_training_dataset(
        str(tmp_path / "flip-training-frame-ranges.p"),
        CleaningOptions(spatial_filter_size=(4,), temporal_filter_size=(4,)),
    )


def clean_for_extract_file(tmp_path):
    """``extract FILE --config-file`` on a copy of one fixture recording, whose
    settings reach extraction as tuples."""
    import shutil

    from click.testing import CliRunner

    import moseq2_extract.cli as extract_cli
    from moseq2_common.yaml import write_yaml

    session = tmp_path / "saline_example_0"
    shutil.copytree(Path(fetch_raw_sessions()) / "saline_example_0", session)
    config = tmp_path / "config.yaml"
    write_yaml({"spatial_filter_size": [4], "temporal_filter_size": [4]}, str(config))
    result = CliRunner().invoke(
        extract_cli.cli,
        ["extract", str(session / "depth.avi"), "--config-file", str(config), "--num-frames", "20"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "clean",
    [clean_for_extract_file, clean_for_pca, clean_for_a_crowd_movie, clean_for_flip_training],
    ids=["extract_file", "pca", "crowd_movies", "flip_training"],
)
def test_an_even_median_kernel_rounds_up_everywhere(clean, tmp_path, monkeypatch):
    """Asked for a median kernel of 4, each step filters with 5, and says so.

    Extraction's spatial and temporal median filters, PCA's, crowd movies'
    spatial one and flip training's spatial and temporal ones, which are
    extraction's settings. Every size reaching OpenCV or scipy is 5.
    """
    kernels = recorded_median_kernels(monkeypatch)
    with pytest.warns(UserWarning, match="rounds up to 5"):
        clean(tmp_path)
    assert kernels and set(kernels) == {5}


def test_extraction_rounds_every_kernel_and_leaves_its_settings_alone():
    """``check_filter_sizes`` rounds each kernel of both lists, takes tuples, and
    returns new lists rather than editing the settings it was given."""
    from moseq2_extract.util import check_filter_sizes

    given = {"spatial_filter_size": (5, 4), "temporal_filter_size": (0, 6)}
    with pytest.warns(UserWarning, match="rounds up"):
        checked = check_filter_sizes(given)
    assert checked["spatial_filter_size"] == [5, 5]
    assert checked["temporal_filter_size"] == [0, 7]
    assert given == {"spatial_filter_size": (5, 4), "temporal_filter_size": (0, 6)}


def test_odd_kernel_keeps_odd_sizes_and_sizes_that_turn_the_filter_off():
    from moseq2_common.frames import odd_kernel

    assert [odd_kernel(size) for size in (0, 1, 3, 5)] == [0, 1, 3, 5]
    with pytest.warns(UserWarning, match="rounds up to 7"):
        assert odd_kernel(6) == 7


def test_viz_no_longer_offers_clean_frames():
    """Crowd movies' cleaning moved to ``moseq2_common.frames``, and the old path says so."""
    with pytest.raises(ImportError):
        from moseq2_viz.viz import clean_frames  # noqa: F401
