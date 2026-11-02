"""Reading a recording's frame times, and refusing to extract without them.

``depth_ts.txt`` (or ``timestamps.csv``, or an ``.mkv`` container's own
timestamps) gives the acquisition time of every depth frame. PCA uses them to
find the frames the camera dropped and to put an empty row where each one was,
so a recording extracted without them looks to every later step as though no
frame was ever dropped.

The columns are separated by whitespace, so a comma inside a field can only be
a decimal separator, as a system set to a decimal-comma locale writes it
(moseq2-app#289).
"""

from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
from moseq2_extract.helpers.data import handle_extract_metadata
from moseq2_extract.util import load_timestamps

from fixtures import fetch_azure_mkv


def write(path, text):
    path.write_text(text)
    return str(path)


def test_decimal_commas_one_column(tmp_path):
    ts = load_timestamps(write(tmp_path / "depth_ts.txt", "25044816,2766\n25044849,6\n"))
    np.testing.assert_array_equal(ts, [25044816.2766, 25044849.6])


def test_decimal_commas_two_columns(tmp_path):
    path = write(tmp_path / "depth_ts.txt", "25044816,2766 0\n25044849,6 1\n6306387,884 0\n")
    np.testing.assert_array_equal(load_timestamps(path, col=0), [25044816.2766, 25044849.6, 6306387.884])
    np.testing.assert_array_equal(load_timestamps(path, col=1), [0, 1, 0])


def test_decimal_points_are_unchanged(tmp_path):
    path = write(tmp_path / "depth_ts.txt", "674717230.8241 0\n674717263.8803 0\n")
    np.testing.assert_array_equal(load_timestamps(path), [674717230.8241, 674717263.8803])


def test_timestamps_csv_reads_decimal_commas(tmp_path):
    """The ``alternate`` path parses the same way, then converts seconds to milliseconds."""
    ts = load_timestamps(write(tmp_path / "timestamps.csv", "1,5\n1,533\n"), alternate=True)
    np.testing.assert_allclose(ts, [1500.0, 1533.0])


def test_decimal_commas_inside_a_tarball(tmp_path):
    """A tarball's member is read as bytes rather than text."""
    member = io.BytesIO(b"25044816,2766 0\n25044849,6 0\n")
    np.testing.assert_array_equal(load_timestamps(member), [25044816.2766, 25044849.6])


def test_an_unreadable_field_names_the_file_and_line(tmp_path):
    path = write(tmp_path / "depth_ts.txt", "1.5 0\n2,5,1 0\n")
    with pytest.raises(ValueError) as e:
        load_timestamps(path)
    message = str(e.value)
    assert path in message and "line 2" in message and "'2,5,1'" in message


# ---------------------------------------------------------------- no timestamps


@pytest.fixture
def recording_without_timestamps(tmp_path, raw_sessions):
    """A copy of one fixture recording with its depth_ts.txt left behind."""
    source = Path(raw_sessions) / "saline_example_0"
    session = tmp_path / "saline_example_0"
    session.mkdir()
    for name in ("depth.avi", "metadata.json"):
        shutil.copy(source / name, session / name)
    return session


def test_extraction_without_timestamps_stops_before_writing(recording_without_timestamps):
    session = recording_without_timestamps
    output_dir = session / "proc"
    result = CliRunner().invoke(
        extract_cli.cli,
        ["extract", str(session / "depth.avi"), "--output-dir", str(output_dir), "--num-frames", "20"],
    )
    assert isinstance(result.exception, FileNotFoundError), result.output
    message = str(result.exception)
    assert str(session / "depth.avi") in message
    assert str(session / "depth_ts.txt") in message
    assert str(session / "timestamps.csv") in message
    assert "frames the camera dropped cannot be found" in message
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_a_tarball_without_timestamps_names_the_members_looked_for(tmp_path):
    tarball = tmp_path / "session.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for name, data in (("metadata.json", b"{}"), ("depth.dat", b"\0" * 16)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with pytest.raises(FileNotFoundError) as e:
        handle_extract_metadata(str(tarball), str(tmp_path))
    message = str(e.value)
    assert str(tarball) in message and "depth_ts.txt" in message and "timestamps.csv" in message


def test_an_empty_depth_ts_counts_as_missing(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "depth.avi").touch()
    write(session / "metadata.json", "{}")
    write(session / "depth_ts.txt", "")
    with pytest.raises(FileNotFoundError, match="empty"):
        handle_extract_metadata(str(session / "depth.avi"), str(session))


def test_an_mkv_carries_its_own_timestamps(tmp_path):
    """No depth_ts.txt beside an .mkv is fine: the container has the frame times."""
    session = tmp_path / "session"
    session.mkdir()
    shutil.copy(Path(fetch_azure_mkv()) / "depth.mkv", session / "depth.mkv")
    write(session / "metadata.json", "{}")
    _, timestamps, _ = handle_extract_metadata(str(session / "depth.mkv"), str(session))
    assert len(timestamps) == 60


# ---------------------------------------------------------------- frame count mismatch


@pytest.fixture
def recording_with_short_timestamps(tmp_path, raw_sessions):
    """A copy of one fixture recording, and its timestamps one line short."""
    source = Path(raw_sessions) / "saline_example_0"
    session = tmp_path / "saline_example_0"
    session.mkdir()
    for name in ("depth.avi", "metadata.json"):
        shutil.copy(source / name, session / name)
    lines = (source / "depth_ts.txt").read_text().splitlines(keepends=True)
    return session, lines[:-1]


def mismatch(session):
    result = CliRunner().invoke(
        extract_cli.cli,
        ["extract", str(session / "depth.avi"), "--output-dir", str(session / "proc"), "--num-frames", "20"],
    )
    assert isinstance(result.exception, ValueError), result.output
    return str(result.exception)


def expected_mismatch(session, n_frames, n_timestamps, source):
    recording = session / "depth.avi"
    return (
        f"Frame count mismatch for {recording}: the depth video has {n_frames} frames but "
        f"{n_timestamps} timestamps were read from {source}. Each frame needs one timestamp, "
        f"so the video or the timestamps are incomplete, or they do not belong together. "
        f"Check that {source} is this recording's own timestamp file and that neither was "
        f"cut short when copied, then extract the recording again."
    )


def test_a_frame_count_mismatch_names_depth_ts(recording_with_short_timestamps):
    session, lines = recording_with_short_timestamps
    (session / "depth_ts.txt").write_text("".join(lines))
    assert mismatch(session) == expected_mismatch(
        session, len(lines) + 1, len(lines), session / "depth_ts.txt"
    )


def test_a_frame_count_mismatch_names_timestamps_csv(recording_with_short_timestamps):
    """timestamps.csv is read when there is no depth_ts.txt; the message names it."""
    session, lines = recording_with_short_timestamps
    seconds = [float(line.split()[0]) / 1000 for line in lines]
    (session / "timestamps.csv").write_text("".join(f"{t:.6f} 0\n" for t in seconds))
    message = mismatch(session)
    assert "depth_ts.txt" not in message
    assert message == expected_mismatch(
        session, len(lines) + 1, len(lines), session / "timestamps.csv"
    )


def test_a_frame_count_mismatch_names_a_tarball_member(tmp_path):
    """A tarball's three-frame depth.dat with two lines in its timestamps.csv."""
    tarball = tmp_path / "session_1.tar.gz"
    members = (
        ("metadata.json", b"{}"),
        ("depth.dat", b"\0" * (512 * 424 * 2 * 3)),
        ("timestamps.csv", b"0.000000 0\n0.033333 0\n"),
    )
    with tarfile.open(tarball, "w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    result = CliRunner().invoke(
        extract_cli.cli, ["extract", str(tarball), "--output-dir", str(tmp_path / "proc")]
    )
    assert isinstance(result.exception, ValueError), result.output
    source = f"timestamps.csv inside {tarball}"
    assert str(result.exception) == (
        f"Frame count mismatch for {tarball}: the depth video has 3 frames but "
        f"2 timestamps were read from {source}. Each frame needs one timestamp, "
        f"so the video or the timestamps are incomplete, or they do not belong together. "
        f"Check that {source} is this recording's own timestamp file and that neither was "
        f"cut short when copied, then extract the recording again."
    )
