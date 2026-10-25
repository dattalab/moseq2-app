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
