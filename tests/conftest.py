"""Fixtures shared by the whole suite.

The extraction and modeling notebook is a pipeline: each cell consumes the
directory tree the previous cell wrote. Running that pipeline once per test
is unaffordable, and running all of it inside a single fixture turns any
mid-pipeline break into an error reported against whichever test happened to
ask for it first.

So each cell gets its own session-scoped fixture that depends on the previous
cell's. The work happens once, and a failure inside one of them errors that
cell's test and everything downstream, which is the right signal: the pipeline
stopped there. Those fixtures are added as their cells are written. This file
holds what every cell needs.
"""

from __future__ import annotations

import os
import shutil

# Before anything imports matplotlib. Nothing here has a display.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from tests.fixtures import fetch_analysis_bundle, fetch_raw_sessions
from tests.widgets import capture_display


@pytest.fixture
def displayed(monkeypatch):
    """Capture whatever moseq2-app hands to ``display()`` during this test."""
    return capture_display(monkeypatch)


@pytest.fixture(scope="session")
def raw_sessions():
    """The downloaded, read-only tier of four cut-down depth recordings."""
    return fetch_raw_sessions()


@pytest.fixture(scope="session")
def analysis_bundle():
    """The downloaded, read-only snapshot of a finished twenty-session project."""
    return fetch_analysis_bundle()


@pytest.fixture(scope="session")
def arena(project, progress_file, configured, flipped):
    """Cell 5: the per-recording parameters extraction reads.

    The cell registers a session config path, names the subdirectory
    extractions are written to, and opens the arena mask widget. Opening it is
    what writes the file: the widget detects each recording's camera
    parameters and saves a copy of the configuration per recording.

    Clicking "save session parameters and move to next" once per recording is
    what the walkthrough did. It keeps the detected parameters as they are and
    records them, which is the state the walkthrough's own session config is
    in. Advancing also computes each recording's arena mask and a short test
    extraction, so the chain reaches cell 6 having already exercised both.
    """
    import panel as pn

    from moseq2_app.gui.progress import update_progress
    from moseq2_app.roi.widget import ArenaMaskWidget
    from moseq2_app.util import update_config

    from tests.widgets import buttons_of, click

    pn.extension()

    session_config = project / "session_config.yaml"
    update_progress(str(progress_file), "session_config", str(session_config))
    with update_config(str(configured)) as config:
        config["camera_type"] = "auto"
        config["output_dir"] = "proc"

    widget = ArenaMaskWidget(str(project), str(configured), str(session_config))
    buttons = buttons_of(widget.view)
    for _ in widget.sessions:
        click(buttons["Save session parameters and move to next"])
    return session_config


@pytest.fixture(scope="session")
def flipped(project, configured, raw_sessions):
    """Cell 4's effect, without the download.

    The cell fetches a flip classifier and writes its path into the config.
    The classifier it fetches by default ships in the fixture bundle already,
    so the chain points the config at that copy and reaches the same state
    offline. Whether the download itself works is cell 4's own network test.
    """
    from moseq2_app.util import update_config

    classifier = raw_sessions / "flip_classifier_k2_c57_10to13weeks.npz"
    with update_config(str(configured)) as config:
        config["flip_classifier"] = str(classifier)
    return classifier


@pytest.fixture(scope="session")
def configured(project, progress_file):
    """Cell 3: the config file every later step reads its parameters from."""
    from moseq2_app.gui.progress import update_progress
    from moseq2_extract.gui import generate_config_command

    config = project / "config.yaml"
    generate_config_command(str(config), camera_type="k2")
    update_progress(str(progress_file), "config_file", str(config))
    return config


@pytest.fixture(scope="session")
def progress_file(project):
    """Cell 2: the state file every later cell writes its output paths into.

    Created in a project that holds nothing but recordings, which is where a
    user starts.
    """
    from moseq2_app.gui.progress import restore_progress_vars

    path = project / "progress.yaml"
    restore_progress_vars(str(path), init=True, overwrite=False)
    return path


@pytest.fixture(scope="session")
def project(tmp_path_factory, raw_sessions):
    """A writable project directory holding the four recordings.

    This is where a user starts: a folder with one subdirectory per recording
    and nothing else. Every later fixture in the chain builds on it, so it is
    copied out of the read-only cache once per session.
    """
    root = tmp_path_factory.mktemp("project")
    shutil.copytree(raw_sessions, root, dirs_exist_ok=True)
    return root
