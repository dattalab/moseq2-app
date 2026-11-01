"""What extraction says about a recording's frame count and its own options.

A Kinect v2 ``.dat`` knows its frame count from its size and an ``.avi``
usually records one, but an Azure ``.mkv`` stores a duration rather than a
count. Extraction then takes the number of frame timestamps as the count: the
container's own frame times when no ``depth_ts.txt`` sits beside the ``.mkv``,
and the file's lines when one does. The message names the recording, the count
and where the count came from.

``-t`` is the short flag of ``--temporal-filter-size`` on ``extract`` and
``batch-extract``, as it has been since 2018. ``--threads``, which those two
commands take and do not use, carries no short flag there; it keeps ``-t`` on
``convert-raw-to-avi`` and ``copy-slice``, where nothing else claims it.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli

from fixtures import fetch_azure_mkv


# ---------------------------------------------------------------- frame count


@pytest.fixture
def azure_recording(tmp_path):
    """A copy of the synthetic Azure recording (60 frames), no depth_ts.txt beside it."""
    session = tmp_path / "azure"
    session.mkdir()
    shutil.copy(Path(fetch_azure_mkv()) / "depth.mkv", session / "depth.mkv")
    (session / "metadata.json").write_text("{}")
    return session


def extract(recording):
    """Extract a recording; return what it printed and every warning it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = CliRunner().invoke(
            extract_cli.cli,
            ["extract", str(recording), "--output-dir", str(recording.parent / "proc"),
             "--camera-type", "azure"],
        )
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result.output, [str(w.message) for w in caught]


def test_an_mkv_counts_its_frames_from_its_own_timestamps(azure_recording):
    """The count is the video's own, so nothing is said about depth_ts.txt
    and nothing is left to check by hand."""
    mkv = azure_recording / "depth.mkv"
    output, caught = extract(mkv)
    everything = output + "\n".join(caught)

    assert "depth_ts.txt" not in everything
    assert not any("frame count" in m.lower() for m in caught), caught
    assert (
        f"{mkv} records no frame count of its own; extracting the 60 frames "
        f"counted from the frame timestamps inside {mkv}." in output
    )


def test_an_mkv_beside_a_depth_ts_names_that_file(azure_recording):
    """With a depth_ts.txt beside it, that file's lines are the count, and
    they cannot be checked against the video."""
    mkv = azure_recording / "depth.mkv"
    depth_ts = azure_recording / "depth_ts.txt"
    depth_ts.write_text("".join(f"{1000 * i / 30:.4f} 0\n" for i in range(60)))

    _, caught = extract(mkv)
    message = [m for m in caught if "frame count" in m]
    assert len(message) == 1, caught
    assert message[0] == (
        f"{mkv} records no frame count of its own, so extraction takes the 60 "
        f"timestamps in {depth_ts} as its number of frames. That count cannot be "
        f"checked against the video. If {depth_ts} is not this recording's own "
        f"timestamp file, or the extraction stops early or comes out shorter than "
        f"the recording, check that the file belongs to {mkv}."
    )


# ---------------------------------------------------------------- -t


def options(command):
    return {p.name: p for p in command.params}


def duplicate_flag_warnings(command):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = CliRunner().invoke(extract_cli.cli, [command, "--help"])
    assert result.exit_code == 0, result.output
    return [str(w.message) for w in caught if "used more than once" in str(w.message)]


def test_extract_help_does_not_warn_about_a_duplicate_flag():
    assert duplicate_flag_warnings("extract") == []


def test_batch_extract_help_does_not_warn_about_t():
    """batch-extract's ``-c`` is shared by ``--crop-size`` and ``--ncpus``, a
    separate matter; only ``-t`` is checked here."""
    assert [m for m in duplicate_flag_warnings("batch-extract") if "-t " in m] == []


@pytest.mark.parametrize("command", [extract_cli.extract, extract_cli.batch_extract])
def test_t_is_the_temporal_filter_on_extraction_commands(command):
    params = options(command)
    assert "-t" in params["temporal_filter_size"].opts
    assert params["threads"].opts == ["--threads"]
    args = ["x.dat", "-t", "5"] if command is extract_cli.extract else [".", "-t", "5"]
    ctx = command.make_context(command.name, args, resilient_parsing=True)
    assert ctx.params["temporal_filter_size"] == (5,)
    assert ctx.params["threads"] == 8


@pytest.mark.parametrize("name", ["convert_raw_to_avi", "copy_slice"])
def test_t_is_still_threads_on_the_video_commands(name):
    command = getattr(extract_cli, name)
    assert options(command)["threads"].opts == ["-t", "--threads"]
