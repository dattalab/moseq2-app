"""Listing a project's recordings and extractions for the notebooks' project and QC cells.

``extract_modeling/test_02_progress.py`` walks the project cell; the tests
here hold the listing to what it reports.
"""

from __future__ import annotations

from moseq2_app.gui.progress import get_session_paths, get_sessions


def test_a_project_under_a_flipped_directory_lists_its_sessions(tmp_path):
    """Only the movies flip correction writes, ``*_flipped.mp4``, are told apart as flipped.

    The progress and QC cells list a project's recordings and extractions with
    ``get_session_paths``. A project kept under a directory with "flipped" in
    its name still has two recordings, two extractions and two flipped movies.
    """
    project = tmp_path / "flipped_mice" / "project"
    for session in ("s0", "s1"):
        proc = project / session / "proc"
        proc.mkdir(parents=True)
        (project / session / "depth.dat").write_bytes(b"")
        (project / session / "metadata.json").write_text("{}")
        (proc / "results_00.mp4").write_bytes(b"")
        (proc / "results_00_flipped.mp4").write_bytes(b"")

    raw = get_session_paths(str(project))
    extracted = get_session_paths(str(project), extracted=True)
    flipped = get_session_paths(str(project), extracted=True, flipped=True)

    assert raw == {s: str(project / s / "depth.dat") for s in ("s0", "s1")}
    assert extracted == {s: str(project / s / "proc" / "results_00.mp4") for s in ("s0", "s1")}
    assert flipped == {s: str(project / s / "proc" / "results_00_flipped.mp4") for s in ("s0", "s1")}


def test_a_project_under_a_depth_directory_lists_only_its_depth_recordings(tmp_path):
    """A session folder holds one recording, and one with "depth" in its name is taken first.

    A session can hold other movies beside its depth recording, such as
    ``ir.avi`` and ``rgb.avi``. A project kept under a directory with "depth"
    in its name still lists one recording per session.
    """
    project = tmp_path / "depth_study" / "project"
    for session in ("s0", "s1"):
        (project / session).mkdir(parents=True)
        (project / session / "depth.avi").write_bytes(b"")
        (project / session / "ir.avi").write_bytes(b"")
        (project / session / "rgb.avi").write_bytes(b"")

    assert get_session_paths(str(project)) == {s: str(project / s / "depth.avi") for s in ("s0", "s1")}


def test_the_arena_tool_takes_the_depth_recording_not_the_nidaq_stream(tmp_path):
    """With its NI-DAQ stream on, the Kinect v2 app writes ``nidaq.dat`` beside
    ``depth.dat``. The arena tool keeps one file per session folder, so a
    second ``.dat`` would replace the recording."""
    for session in ("s0", "s1"):
        (tmp_path / session).mkdir()
        (tmp_path / session / "depth.dat").write_bytes(b"")
        (tmp_path / session / "nidaq.dat").write_bytes(b"")
        (tmp_path / session / "metadata.json").write_text("{}")

    assert get_sessions(str(tmp_path), skip_extracted=False) == [
        str(tmp_path / s / "depth.dat") for s in ("s0", "s1")
    ]
