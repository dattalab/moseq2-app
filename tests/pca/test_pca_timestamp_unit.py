"""PCA reads each extraction's frame times in the unit the file records.

Extraction writes ``/timestamps`` in milliseconds and says so in the dataset's
``unit`` attribute. PCA converts to seconds by that attribute; a file written
before the attribute existed is read as milliseconds, as it always was. An
Azure recording extracted with the attribute reaches PCA at its own 30 frames
per second rather than 30,000, which lets the frame-rate check be tight for
any file that records its unit while still accepting older Azure extractions,
whose seconds read as milliseconds give about 30,000.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner
from dask.distributed import Client

import moseq2_extract.cli as extract_cli
from moseq2_pca.pca.util import apply_pca_dask, get_timestamps
from moseq2_pca.util import frame_rate_from_timestamps

from fixtures import fetch_azure_mkv

N = 300
SECONDS = np.arange(N) / 30


@pytest.fixture
def client():
    with Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None) as c:
        yield c


def session(tmp_path, uuid, timestamps, unit=None):
    h5 = tmp_path / f"{uuid}.h5"
    with h5py.File(h5, "w") as f:
        f["frames"] = np.random.default_rng(0).random((len(timestamps), 4, 4))
        f["timestamps"] = timestamps
        if unit is not None:
            f["timestamps"].attrs["unit"] = unit
        f["metadata/uuid"] = uuid
    return str(h5), str(h5)


def apply(tmp_path, client, h5, yml):
    save_file = str(tmp_path / "pca_scores")
    apply_pca_dask(
        np.ones((2, 16)), [h5], [yml], use_fft=False,
        clean_params={"gaussfilter_time": 0, "medfilter_time": [0]},
        save_file=save_file, chunk_size=64, mask_params={}, missing_data=False,
        client=client, h5_path="/frames",
    )
    return save_file + ".h5"


@pytest.mark.parametrize(
    "stored, unit",
    [(SECONDS * 1000, "milliseconds"), (SECONDS, "seconds"), (SECONDS * 1000, None)],
    ids=["milliseconds", "seconds", "no-attribute"],
)
def test_get_timestamps_reads_the_recorded_unit(tmp_path, stored, unit):
    h5, _ = session(tmp_path, "s", stored, unit)
    with h5py.File(h5, "r") as f:
        np.testing.assert_allclose(get_timestamps(f, f["frames"]), SECONDS)


def test_an_unknown_unit_stops_with_the_file_and_the_unit(tmp_path):
    h5, _ = session(tmp_path, "s", SECONDS, "fortnights")
    with h5py.File(h5, "r") as f, pytest.raises(ValueError) as e:
        get_timestamps(f, f["frames"])
    message = str(e.value)
    assert h5 in message and "'fortnights'" in message and "milliseconds" in message


@pytest.fixture(scope="module")
def azure_extraction(tmp_path_factory):
    session_dir = tmp_path_factory.mktemp("azure")
    shutil.copy(Path(fetch_azure_mkv()) / "depth.mkv", session_dir / "depth.mkv")
    (session_dir / "metadata.json").write_text("{}")
    result = CliRunner().invoke(
        extract_cli.cli,
        ["extract", str(session_dir / "depth.mkv"), "--camera-type", "azure",
         "--output-dir", str(session_dir / "proc")],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    return session_dir / "proc" / "results.h5"


def test_an_azure_extraction_reaches_pca_at_30_fps(azure_extraction):
    with h5py.File(azure_extraction, "r") as f:
        timestamps = get_timestamps(f, f["frames"])
    assert frame_rate_from_timestamps(timestamps) == 30


def test_an_older_azure_extraction_without_the_unit_is_still_accepted(tmp_path, client):
    """Seconds in a file without the attribute read as milliseconds: about 30,000 fps."""
    h5, yml = session(tmp_path, "old-azure", SECONDS)
    with h5py.File(apply(tmp_path, client, h5, yml), "r") as f:
        assert f["scores/old-azure"].shape[0] == N


def test_a_rate_out_of_camera_range_stops_when_the_unit_is_recorded(tmp_path, client):
    """Seconds labelled as milliseconds give 30,000 fps, which no depth camera records."""
    h5, yml = session(tmp_path, "mislabelled", SECONDS, "milliseconds")
    with pytest.raises(ValueError) as e:
        apply(tmp_path, client, h5, yml)
    message = str(e.value)
    assert "mislabelled" in message and h5 in message
    assert "1,000" in message and "frames per second" in message
    # the timestamps may have come from any of the three sources extraction reads
    assert "depth_ts.txt" in message and "timestamps.csv" in message and ".mkv" in message
