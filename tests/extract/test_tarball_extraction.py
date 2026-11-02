"""Extracting a compressed session (``.tar.gz``) to the end.

A kinect2-nidaq acquisition can be kept as one ``session_XXXX.tar.gz`` holding
``depth.dat``, ``depth_ts.txt`` and ``metadata.json``. Extraction reads the
frames straight from the archive and writes its results where it would for an
uncompressed recording: ``proc/`` beside the archive.

The tarball is built from the first 200 frames of the ``saline_example_0``
fixture recording, with the matching 200 lines of its ``depth_ts.txt``.
"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import h5py
import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
from moseq2_extract.io.video import get_movie_info, load_movie_data

N_FRAMES = 200


@pytest.fixture
def tarball(tmp_path, raw_sessions):
    """``session_20190320/session_20190320.tar.gz``, a 200-frame kinect2-nidaq session."""
    source = Path(raw_sessions) / "saline_example_0"
    avi = str(source / "depth.avi")
    finfo = get_movie_info(avi)
    frames = load_movie_data(avi, range(N_FRAMES), frame_size=finfo["dims"], finfo=finfo)

    stage = tmp_path / "stage"
    stage.mkdir()
    frames.astype("<u2").tofile(stage / "depth.dat")
    lines = (source / "depth_ts.txt").read_text().splitlines(keepends=True)[:N_FRAMES]
    (stage / "depth_ts.txt").write_text("".join(lines))
    shutil.copy(source / "metadata.json", stage / "metadata.json")

    session = tmp_path / "project" / "session_20190320"
    session.mkdir(parents=True)
    path = session / "session_20190320.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name in ("depth.dat", "depth_ts.txt", "metadata.json"):
            tar.add(stage / name, arcname=name)
    return path


def test_a_tarball_extracts_beside_the_archive(tarball):
    result = CliRunner().invoke(extract_cli.cli, ["extract", str(tarball)])
    assert result.exception is None, result.output

    proc = tarball.parent / "proc"
    for name in ("results_00.h5", "results_00.mp4", "results_00.yaml"):
        assert (proc / name).is_file(), f"{name} missing from {proc}: {sorted(p.name for p in proc.iterdir())}"
    with h5py.File(proc / "results_00.h5", "r") as f:
        assert f["frames"].shape[0] == N_FRAMES
        assert len(f["timestamps"]) == N_FRAMES
    assert "complete: true" in (proc / "results_00.yaml").read_text()
