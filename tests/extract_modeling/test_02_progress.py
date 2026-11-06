"""Cell 2: name the project and report what is already done.

    from moseq2_app.project import start

    base_dir = './' # Add the path to your data folder here.

    project = start(base_dir)

``start`` writes nothing. It returns the project (the folder that holds, or
will hold, ``config.yaml``) for the later cells, and prints how many of the
project's recordings are extracted and what its ``moseq2-project.yaml`` lists:
the sessions, the PCA files and the models. Every later cell takes the project
folder and reads that file when it needs a path, so nothing is carried between
cells.

A recording is extracted when its ``results.h5`` exists in its folder in
``aggregate_results/``, and an extraction stopped partway leaves its unfinished
``.results.h5.partial`` there, so a project in any state of extraction is a few
empty files, and the reported states below are built that way rather than by
running the pipeline.
"""

from __future__ import annotations

import shutil

import pytest

from moseq2_app.project import start
from moseq2_common.yaml import write_yaml


def build_project(root, n_sessions=4, n_extracted=0, n_incomplete=0):
    """A project folder whose recordings are extracted up to a given point.

    Sessions are found by their recording files and extractions by the
    ``results.h5`` in each session's folder, so empty files are enough to be
    found.
    """
    root.mkdir(parents=True, exist_ok=True)
    marked = 0
    for i in range(n_sessions):
        session = root / f"session_{i}"
        session.mkdir(parents=True, exist_ok=True)
        (session / "depth.avi").touch()
        if marked < n_extracted + n_incomplete:
            folder = root / "aggregate_results" / f"session_{i}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / ("results.h5" if marked < n_extracted else ".results.h5.partial")).touch()
            marked += 1
    return root


@pytest.fixture(scope="module")
def fresh_project(tmp_path_factory, raw_sessions):
    """This cell run on its own copy of the four recordings, which is all a project holds at this point."""
    root = tmp_path_factory.mktemp("fresh_project")
    shutil.copytree(raw_sessions, root, dirs_exist_ok=True)
    return root


def listing(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_returns_the_project_and_writes_nothing(fresh_project):
    before = listing(fresh_project)

    project = start(str(fresh_project))

    assert project.folder == fresh_project
    assert project.config_file == fresh_project / "config.yaml"
    assert listing(fresh_project) == before


def test_a_recording_without_metadata_gets_nothing_written(tmp_path):
    """The cell only reports: a recording folder without metadata.json is left as it is."""
    root = build_project(tmp_path / "proj", n_sessions=2)
    before = listing(root)

    start(str(root))

    assert listing(root) == before
    assert not (root / "session_0" / "metadata.json").exists()


def test_reports_nothing_done_in_a_fresh_project(fresh_project, capsys):
    """The real project at this point in the notebook: four recordings, no work done."""
    start(str(fresh_project))
    out = capsys.readouterr().out

    assert "Extraction Progress: 0 out of 4 session(s) extracted" in out
    for session in ("saline_example_0", "saline_example_1",
                    "amphetamine_example_0", "amphetamine_example_1"):
        assert f"Not yet extracted: {session}" in out
    assert "config.yaml: not written yet" in out
    assert "Sessions: 0 aggregated" in out
    assert "PCA: not run yet" in out
    assert "Models: none fitted yet" in out


@pytest.mark.parametrize(
    "state, expected",
    [
        pytest.param(dict(n_extracted=0), "Extraction Progress: 0 out of 4 session(s) extracted",
                     id="nothing-extracted"),
        pytest.param(dict(n_extracted=2), "Extraction Progress: 2 out of 4 session(s) extracted",
                     id="half-extracted"),
        pytest.param(dict(n_extracted=4), "Extraction Progress: 4 out of 4 session(s) extracted",
                     id="all-extracted"),
        pytest.param(dict(n_extracted=1, n_incomplete=1),
                     "Extraction Progress: 1 out of 4 session(s) extracted",
                     id="one-incomplete-does-not-count"),
    ],
)
def test_extraction_counts(tmp_path, capsys, state, expected):
    path = build_project(tmp_path / "proj", **state)
    start(str(path))
    assert expected in capsys.readouterr().out


def test_an_incomplete_extraction_is_named(tmp_path, capsys):
    """An extraction that left its unfinished file was stopped partway, most
    often by a job or kernel killed for memory or time. The line says so,
    where to look and how to redo it."""
    path = build_project(tmp_path / "proj", n_extracted=0, n_incomplete=1)
    start(str(path))
    session = tmp_path / "proj" / "session_0"
    proc = tmp_path / "proj" / "aggregate_results" / "session_0"
    assert (
        f"Extraction of session_0 started but did not finish: {proc / '.results.h5.partial'} "
        "is the unfinished extraction. An extraction stopped partway leaves this behind, most "
        "often because the job or notebook kernel running it was stopped, for example "
        "killed for running out of memory or for reaching its time limit. Look in "
        f"{proc} for what it wrote and, if it ran on SLURM, in the job's log, "
        "slurm-<job id>.out in the directory the extraction jobs were submitted from. "
        f"Run the extraction again for {session / 'depth.avi'} to redo it.\n"
    ) in capsys.readouterr().out



def test_a_directory_named_for_mp4_does_not_hide_extractions(tmp_path, capsys):
    """However the folders holding the project are named, an extraction is its session folder's
    ``results.h5``."""
    path = build_project(tmp_path / "mp4_archive" / "proj", n_extracted=4)
    start(str(path))
    assert "Extraction Progress: 4 out of 4 session(s) extracted" in capsys.readouterr().out




def test_reports_what_the_bookkeeping_file_lists(tmp_path, capsys):
    root = build_project(tmp_path / "proj", n_extracted=4)
    (root / "config.yaml").write_text("camera_type: auto\n")
    write_yaml({
        "sessions": {f"session_{i}/depth.avi": {"recording": f"session_{i}/depth.avi",
                                                 "folder": f"aggregate_results/session_{i}", "uuid": str(i)}
                     for i in range(4)},
        "pca": {"components": "_pca/pca.h5", "scores": "_pca/pca_scores.h5"},
        "models": {"fitted": ["models/model.p", "models/model_kappa_100.p"], "current": "models/model.p",
                   "analysis": {}},
        "applied_models": [],
    }, str(root / "moseq2-project.yaml"))

    start(str(root))
    out = capsys.readouterr().out

    assert "config.yaml: not written yet" not in out
    assert "Sessions: 4 aggregated" in out
    assert "PCA: components _pca/pca.h5, scores _pca/pca_scores.h5" in out
    assert "Models: 2 fitted; current models/model.p" in out
