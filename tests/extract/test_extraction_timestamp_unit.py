"""The unit extraction writes a recording's frame times in.

Every extraction stores its frame times as ``/timestamps`` in milliseconds and
says so in the dataset's ``unit`` attribute. A Kinect v2 ``depth_ts.txt`` is
already in milliseconds and is written as read; ``timestamps.csv`` holds
seconds and is converted on reading; an Azure ``.mkv`` carries its own frame
times in seconds, which are converted before they are written. Everything
after extraction (PCA's gap filling, the scalar tables, the validation cell)
reads ``/timestamps``, so one unit there is what keeps an Azure recording
from reaching them a thousand times off.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
from moseq2_extract.helpers.data import handle_extract_metadata
from moseq2_extract.io.video import load_timestamps_from_movie
from moseq2_extract.util import load_timestamps

from fixtures import fetch_azure_mkv


def extract(recording, *args):
    output_dir = recording.parent / "proc"
    result = CliRunner().invoke(
        extract_cli.cli,
        ["extract", str(recording), "--output-dir", str(output_dir), *args],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    return output_dir / "results_00.h5"


@pytest.fixture
def kinect_recording(tmp_path, raw_sessions):
    """A copy of one fixture Kinect v2 recording, with its depth_ts.txt."""
    source = Path(raw_sessions) / "saline_example_0"
    session = tmp_path / "saline_example_0"
    session.mkdir()
    for name in ("depth.avi", "depth_ts.txt", "metadata.json"):
        shutil.copy(source / name, session / name)
    return session


@pytest.fixture
def azure_recording(tmp_path):
    """A copy of the synthetic Azure recording, with no depth_ts.txt beside it."""
    session = tmp_path / "azure"
    session.mkdir()
    shutil.copy(Path(fetch_azure_mkv()) / "depth.mkv", session / "depth.mkv")
    (session / "metadata.json").write_text("{}")
    return session


def test_a_kinect_extraction_writes_depth_ts_as_it_was_read(kinect_recording):
    """The values are depth_ts.txt's own, in milliseconds, as before."""
    h5 = extract(kinect_recording / "depth.avi", "--num-frames", "20")
    expected = load_timestamps(str(kinect_recording / "depth_ts.txt"))
    with h5py.File(h5, "r") as f:
        written = f["timestamps"][()]
        unit = f["timestamps"].attrs.get("unit")
    assert written.dtype == expected.dtype
    np.testing.assert_array_equal(written, expected[: len(written)])
    assert unit == "milliseconds"


def test_an_azure_extraction_writes_milliseconds(azure_recording):
    """The container's seconds are written as milliseconds, 33 or 34 ms apart at 30 fps."""
    mkv = azure_recording / "depth.mkv"
    seconds = np.asarray(load_timestamps_from_movie(str(mkv)))
    h5 = extract(mkv, "--camera-type", "azure")
    with h5py.File(h5, "r") as f:
        written = f["timestamps"][()]
        unit = f["timestamps"].attrs.get("unit")
    assert unit == "milliseconds"
    np.testing.assert_allclose(written, seconds[: len(written)] * 1000)
    assert np.median(np.diff(written)) == pytest.approx(1000 / 30, abs=0.5)


def test_an_mkv_s_own_timestamps_are_read_in_milliseconds(azure_recording):
    _, timestamps, _ = handle_extract_metadata(
        str(azure_recording / "depth.mkv"), str(azure_recording)
    )
    assert np.median(np.diff(timestamps)) == pytest.approx(1000 / 30, abs=0.5)
