"""Finding the frames a camera dropped, and putting an empty row where each was.

``insert_nans`` takes the frame period from the median gap between
timestamps, so the drops themselves do not move it, and counts
``round(gap / period)`` frame steps in each gap: a gap of two periods is one
dropped frame, three periods two. A Kinect v2 delivers frames about 33.0 ms
apart with a millisecond or so of jitter, which rounds to one step and adds
nothing.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from dask.distributed import Client

from moseq2_extract.util import load_timestamps
from moseq2_pca.pca.util import apply_pca_dask
from moseq2_pca.util import get_changepoints, insert_nans, write_yaml

N = 300
EXACT = np.arange(N) / 30


def drop(timestamps, *frames):
    return np.delete(timestamps, list(frames))


def empty_rows(timestamps):
    data = np.arange(len(timestamps), dtype="float64")
    filled, idx, filled_ts = insert_nans(timestamps, data)
    assert np.array_equal(np.isnan(filled), np.isnan(idx))
    return np.flatnonzero(np.isnan(filled)), filled_ts


def test_one_frame_dropped_at_exact_spacing():
    rows, filled_ts = empty_rows(drop(EXACT, 20))
    np.testing.assert_array_equal(rows, [20])
    np.testing.assert_allclose(filled_ts, EXACT)


def test_two_frames_dropped_in_a_row():
    rows, filled_ts = empty_rows(drop(EXACT, 20, 21))
    np.testing.assert_array_equal(rows, [20, 21])
    np.testing.assert_allclose(filled_ts, EXACT)


def test_drops_in_a_short_series_do_not_move_the_period():
    """Two drops in ten frames pull the mean gap to 43 ms; the median stays at 33.3."""
    rows, _ = empty_rows(drop(np.arange(10) / 30, 3, 7))
    np.testing.assert_array_equal(rows, [3, 7])


def kinect_like(n, seed=0):
    """Frames 33.02 ms apart with up to 1.2 ms of jitter, like the fixture recordings."""
    gaps = 0.03302 + np.random.default_rng(seed).uniform(-0.0012, 0.0012, n - 1)
    return np.concatenate([[0], np.cumsum(gaps)])


def test_jittered_kinect_spacing_without_drops_adds_nothing():
    rows, _ = empty_rows(kinect_like(N))
    assert len(rows) == 0


def test_one_frame_dropped_from_kinect_spacing():
    """A Kinect's period is 33.0 ms rather than 33.3, so a single drop is a 66 ms gap."""
    rows, _ = empty_rows(drop(np.arange(N) * 0.03302, 100))
    np.testing.assert_array_equal(rows, [100])


@pytest.mark.parametrize(
    "recording",
    ["amphetamine_example_0", "amphetamine_example_1", "saline_example_0", "saline_example_1"],
)
def test_the_fixture_recordings_have_no_dropped_frames(raw_sessions, recording):
    timestamps = load_timestamps(str(Path(raw_sessions) / recording / "depth_ts.txt")) / 1000
    rows, _ = empty_rows(timestamps)
    assert len(rows) == 0


# ---------------------------------------------------------------- through the pipeline


@pytest.fixture
def client():
    with Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None) as c:
        yield c


def test_scores_get_an_empty_row_for_a_single_drop_at_exact_spacing(tmp_path, client):
    timestamps = drop(EXACT, 20)
    h5 = tmp_path / "s.h5"
    with h5py.File(h5, "w") as f:
        f["frames"] = np.random.default_rng(0).random((len(timestamps), 4, 4))
        f["timestamps"] = timestamps * 1000
    write_yaml({"uuid": "s"}, str(tmp_path / "s.yaml"))
    save_file = str(tmp_path / "pca_scores")
    apply_pca_dask(
        np.ones((2, 16)), [str(h5)], [str(tmp_path / "s.yaml")], use_fft=False,
        clean_params={"gaussfilter_time": 0, "medfilter_time": [0]},
        save_file=save_file, chunk_size=64, mask_params={}, missing_data=False,
        client=client, h5_path="/frames",
    )
    with h5py.File(save_file + ".h5", "r") as f:
        idx = f["scores_idx/s"][()]
    assert len(idx) == N
    np.testing.assert_array_equal(np.flatnonzero(np.isnan(idx)), [20])


def test_changepoints_get_an_empty_row_for_a_single_drop_at_exact_spacing():
    timestamps = drop(EXACT, 20)
    rps = np.random.default_rng(0).random((4, len(timestamps)))
    _, normed = get_changepoints(rps, timestamps=timestamps)
    assert normed.shape[0] == N
