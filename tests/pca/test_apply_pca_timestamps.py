"""Writing PC scores for extractions that carry no timestamps.

``apply_pca_dask``, the path ``apply-pca`` takes with ``cluster_type: local``,
reads each session's timestamps to know where dropped frames go. An extraction
without a ``/timestamps`` dataset falls back to one frame every 1/fps
seconds, counted from that session's own frames, so each session's scores
keep its own length.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from dask.distributed import Client

from moseq2_pca.pca.util import apply_pca_dask
from moseq2_pca.util import write_yaml

LENGTHS = {"short": 5, "long": 9}


@pytest.fixture
def client():
    with Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None) as c:
        yield c


@pytest.fixture
def sessions(tmp_path):
    """Two extractions of different lengths, neither with timestamps."""
    h5s, yamls = [], []
    for uuid, length in LENGTHS.items():
        with h5py.File(tmp_path / f"{uuid}.h5", "w") as f:
            f["frames"] = np.random.default_rng(0).random((length, 4, 4))
        write_yaml({"uuid": uuid}, str(tmp_path / f"{uuid}.yaml"))
        h5s.append(str(tmp_path / f"{uuid}.h5"))
        yamls.append(str(tmp_path / f"{uuid}.yaml"))
    return h5s, yamls


def test_each_session_keeps_its_own_length(tmp_path, sessions, client):
    """Scores and their frame index are as long as the session they came from."""
    h5s, yamls = sessions
    save_file = str(tmp_path / "pca_scores")
    apply_pca_dask(
        np.ones((2, 16)), h5s, yamls, use_fft=False,
        clean_params={"gaussfilter_time": 0, "medfilter_time": [0]},
        save_file=save_file, chunk_size=4, mask_params={}, missing_data=False,
        client=client, h5_path="/frames",
    )
    with h5py.File(save_file + ".h5") as f:
        for uuid, length in LENGTHS.items():
            assert f[f"scores/{uuid}"].shape[0] == length
            assert f[f"scores_idx/{uuid}"].shape[0] == length
