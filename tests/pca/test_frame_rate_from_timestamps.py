"""The frame rate PCA works out from a session's timestamps.

Writing PC scores and computing changepoints both put an empty row wherever
the camera dropped a frame, on a grid at the frame rate the timestamps imply
(``insert_nans``). Timestamps that are all equal give no rate at all, and
timestamps a thousand times too large give a rate that rounds to zero
frames per second; either one has to stop with the values that caused it
(moseq2-pca#53, #75). A normal 30 Hz recording is unaffected.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from dask.distributed import Client

from moseq2_pca.pca.util import apply_pca_dask
from moseq2_pca.util import get_changepoints

N = 300
# a Kinect's frames arrive about 33.4 ms apart
SECONDS_30HZ = np.arange(N) * 0.0334
# one frame dropped after frame 19
SECONDS_30HZ_DROPPED = np.concatenate([SECONDS_30HZ[:20], SECONDS_30HZ[21:]])
CONSTANT = np.full(N, 5.0)
MILLISECONDS = np.arange(N) * 33.3


@pytest.fixture
def client():
    with Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None) as c:
        yield c


def session(tmp_path, uuid, timestamps_ms):
    """One extraction whose /timestamps holds the given values (read as milliseconds)."""
    h5 = tmp_path / f"{uuid}.h5"
    with h5py.File(h5, "w") as f:
        f["frames"] = np.random.default_rng(0).random((len(timestamps_ms), 4, 4))
        f["timestamps"] = timestamps_ms
        f["metadata/uuid"] = uuid
    return str(h5), str(h5)


def apply(tmp_path, client, h5, yml):
    save_file = str(tmp_path / "pca_scores")
    apply_pca_dask(
        np.ones((2, 16)), [h5], [yml], use_fft=False,
        clean_params={"gaussfilter_time": 0, "medfilter_time": [0]},
        save_file=save_file, chunk_size=8, mask_params={}, missing_data=False,
        client=client, h5_path="/frames",
    )
    return save_file + ".h5"


@pytest.mark.parametrize(
    "timestamps, cause",
    [(CONSTANT, "all equal"), (MILLISECONDS, "milliseconds")],
    ids=["constant", "milliseconds"],
)
def test_changepoints_stop_on_timestamps_without_a_frame_rate(timestamps, cause):
    scores = np.random.default_rng(0).random((4, len(timestamps)))
    with pytest.raises(ValueError) as e:
        get_changepoints(scores, timestamps=timestamps)
    message = str(e.value)
    assert cause in message
    assert "frames per second" in message
    first_diff = f"{np.diff(timestamps)[0]:g}"
    assert first_diff in message


@pytest.mark.parametrize(
    "timestamps_ms, cause",
    [(CONSTANT * 1000, "all equal"), (MILLISECONDS * 1000, "milliseconds")],
    ids=["constant", "milliseconds"],
)
def test_scores_stop_on_timestamps_without_a_frame_rate(tmp_path, client, timestamps_ms, cause):
    h5, yml = session(tmp_path, "bad-session", timestamps_ms)
    with pytest.raises(ValueError) as e:
        apply(tmp_path, client, h5, yml)
    message = str(e.value)
    assert "bad-session" in message and h5 in message
    assert cause in message


def test_a_30hz_recording_is_unchanged(tmp_path, client):
    """A dropped frame becomes one empty row, as before."""
    h5, yml = session(tmp_path, "ok", SECONDS_30HZ_DROPPED * 1000)
    with h5py.File(apply(tmp_path, client, h5, yml)) as f:
        scores = f["scores/ok"][()]
        idx = f["scores_idx/ok"][()]
    assert scores.shape[0] == N
    assert np.isnan(idx[20]) and np.isnan(scores[20]).all()
    assert np.isfinite(np.delete(idx, 20)).all()

    rps = np.random.default_rng(0).random((4, N - 1))
    _, normed = get_changepoints(rps, timestamps=SECONDS_30HZ_DROPPED)
    assert normed.shape[0] == N
