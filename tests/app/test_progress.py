"""Building ``progress.yaml`` from what is already in a project.

The notebooks' first cell calls ``restore_progress_vars(..., init=True)``.
When there is no progress file yet, or it is asked to overwrite one, that
scans the project with ``find_progress`` and records the paths it finds.
``extract_modeling/test_02_progress.py`` walks the cell; the tests here hold
the scan to what it reports.
"""

from __future__ import annotations

import os

from moseq2_app.gui.progress import find_progress, get_session_paths


def test_the_latest_model_is_the_one_chosen(tmp_path):
    """With several models in the project, the scan says it takes the latest."""
    old, new = tmp_path / "old.p", tmp_path / "new.p"
    old.write_bytes(b"model")
    new.write_bytes(b"model")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    progress = dict(
        base_dir=str(tmp_path),
        config_file=str(tmp_path / "missing.yaml"),
        pca_dirname=str(tmp_path),
    )
    assert find_progress(progress)["model_path"] == str(new)


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
    """A recording is a .dat or .avi file with "depth" in its name.

    A session can hold other movies beside its depth recording, such as
    ``ir.avi``. A project kept under a directory with "depth" in its name
    still lists one recording per session.
    """
    project = tmp_path / "depth_study" / "project"
    for session in ("s0", "s1"):
        (project / session).mkdir(parents=True)
        (project / session / "depth.avi").write_bytes(b"")
        (project / session / "ir.avi").write_bytes(b"")
        (project / session / "rgb.avi").write_bytes(b"")

    assert get_session_paths(str(project)) == {s: str(project / s / "depth.avi") for s in ("s0", "s1")}
