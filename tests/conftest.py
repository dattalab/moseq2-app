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
from pathlib import Path

# Before anything imports matplotlib. Nothing here has a display.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from tests.fixtures import fetch_analysis_bundle, fetch_raw_sessions
from tests.widgets import capture_display


def pytest_configure(config):
    """Keep the suite's scratch directories off ``/tmp``.

    ``extract_found_sessions`` discards every recording whose path contains
    ``/tmp/`` before extracting anything, so a project under pytest's default
    temporary directory extracts nothing while reporting that it finished.
    Anchoring the base directory to the rootdir keeps the pipeline fixtures
    out of that filter's way. An explicit ``--basetemp`` still wins.
    """
    if config.option.basetemp is None:
        config.option.basetemp = str(Path(config.rootpath) / ".pytest-tmp")


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
def aggregated(project, progress_file, extracted):
    """Cell 9: gather every extraction into one directory and index it.

    Until now each recording's output has sat beside the recording. Modeling
    reads one directory, so the cell copies every extraction into
    ``aggregate_results/`` under a name built from the acquisition metadata,
    and writes ``moseq2-index.yaml`` listing what it found. That index is
    what every cell from here on is pointed at.
    """
    from moseq2_app.gui.progress import update_progress
    from moseq2_extract.gui import aggregate_extract_results_command

    update_progress(
        str(progress_file), "train_data_dir", str(project / "aggregate_results") + "/"
    )
    index = aggregate_extract_results_command(
        str(project), "{start_time}_{session_name}_{subject_name}", "aggregate_results/"
    )
    update_progress(str(progress_file), "index_file", index)
    return Path(index)


@pytest.fixture(scope="session")
def extracted(project, configured, arena):
    """Cell 6: extract every recording in the project.

    The heaviest cell, and the one the rest of the notebook is built on. It
    searches the project for depth files, hands them to a local extraction
    loop, and leaves a ``proc`` directory beside each recording.

    ``skip_extracted`` carries over from cell 5, where the notebook sets it,
    so a second run of the cell extracts nothing.
    """
    from moseq2_app.util import update_config
    from moseq2_extract.gui import extract_found_sessions

    with update_config(str(configured)) as config:
        config["cluster_type"] = "local"

    extract_found_sessions(
        str(project),
        str(configured),
        [".avi", ".dat", ".tar.gz"],
        extract_all=True,
        skip_extracted=True,
    )
    return project


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
def grouped(aggregated):
    """Cell 10: label each recording with the condition it belongs to.

    Until this runs every recording is in group ``default`` and a group
    comparison compares the experiment against itself. The cell shows an
    editable table; a user selects rows, types a name, sets it, and saves the
    table back to the index. The chain does the same for the two conditions
    the fixture recordings come from.
    """
    from moseq2_app.main import interactive_group_setting

    widget = interactive_group_setting(str(aggregated))
    for prefix, group in (("saline", "saline"), ("amphetamine", "amph")):
        table = widget.qgrid_widget.get_changed_df()
        widget.qgrid_widget.change_selection(
            [i for i, name in zip(table.index, table["SessionName"]) if name.startswith(prefix)]
        )
        widget.group_input.value = group
        widget.save_button.click()
    widget.update_index_button.click()
    return aggregated


@pytest.fixture(scope="session")
def scalar_viewer(grouped):
    """Cell 11: the scalar summary, and the viewer cell 12 reads out of it.

    Reading every recording's scalars takes a moment, and the viewer is
    immutable once built, so the whole suite shares one.
    """
    from moseq2_app.main import interactive_scalar_summary

    return interactive_scalar_summary(str(grouped))


@pytest.fixture(scope="session")
def pca(project, progress_file, configured, grouped):
    """Cell 13: learn the pose basis every later step works in.

    Eighty by eighty pixels of mouse is too much to model directly, so the
    aggregated frames are reduced to a handful of components describing how a
    mouse's shape varies. Everything from here on is expressed in that basis
    rather than in pixels.

    Runs under dask, which is why the cell configures workers.
    """
    from moseq2_app.gui.progress import update_progress
    from moseq2_app.util import update_config
    from moseq2_pca.gui import train_pca_command

    pca_dirname = str(project / "_pca") + "/"
    paths = update_progress(str(progress_file), "pca_dirname", pca_dirname)
    with update_config(str(configured)) as config:
        config["camera_type"] = "k2"
        config["overwrite_pca"] = False
        config["gaussfilter_space"] = (1.5, 1)
        config["medfilter_space"] = [0]
        config["medfilter_time"] = [0]
        config["nworkers"] = 1

    train_pca_command(paths, pca_dirname, "pca")
    return Path(pca_dirname)


@pytest.fixture(scope="session")
def scores(project, progress_file, pca):
    """Cell 15: express every frame in the basis cell 13 learned.

    The decomposition describes how a mouse's shape varies; this projects
    each recording's frames onto it, turning 6400 pixels per frame into a
    couple of dozen coordinates. The model is fit on these.
    """
    from moseq2_app.gui.progress import update_progress
    from moseq2_pca.gui import apply_pca_command

    scores_file = str(pca / "pca_scores.h5")
    paths = update_progress(str(progress_file), "scores_path", scores_file)
    apply_pca_command(paths, "pca_scores")
    return Path(scores_file)


@pytest.fixture(scope="session")
def changepoints(project, progress_file, configured, scores):
    """Cell 16: find where the mouse's pose changes, without a model.

    A model-free estimate of how long a behavioural unit lasts, computed from
    the PC scores alone. Cell 19 scores a fitted model against it, which is
    how a kappa is chosen, so this is the yardstick rather than an output.
    """
    from moseq2_app.gui.progress import update_progress
    from moseq2_app.util import update_config
    from moseq2_pca.gui import compute_changepoints_command

    with update_config(str(configured)) as config:
        config["dims"] = 300
    paths = update_progress(str(progress_file), "changepoints_path", "changepoints")
    compute_changepoints_command(paths["train_data_dir"], paths, "changepoints")
    return Path(paths["pca_dirname"]) / "changepoints.h5"


#: What cell 17 sets before the model is fit.
MODEL_SETTINGS = {
    "checkpoint_freq": -1,
    "use_checkpoint": False,
    "npcs": 10,
    "max_states": 100,
    "robust": True,
    "save_model": True,
    "separate_trans": False,
    "num_iter": 100,
    "select_groups": False,
    "kappa": None,
    "hold_out": False,
    "nfolds": 2,
    "cluster_type": "local",
    "ncpus": 0,
}


@pytest.fixture(scope="session")
def model_settings(project, progress_file, configured, changepoints):
    """Cell 17: choose how the model is fit, and where it goes.

    Nothing is computed here. The cell records the model's paths and writes
    the modelling parameters into the configuration cell 18 reads.
    """
    from moseq2_app.gui.progress import update_progress
    from moseq2_app.util import update_config

    models = project / "models"
    update_progress(str(progress_file), "model_path", str(models / "model.p"))
    paths = update_progress(str(progress_file), "base_model_path", str(models) + "/")
    with update_config(str(configured)) as config:
        config.update(MODEL_SETTINGS)
    return paths


@pytest.fixture(scope="session")
def model(model_settings):
    """Cell 18: fit the AR-HMM.

    The heaviest cell in the notebook. One fit, shared by the whole suite.
    """
    from moseq2_model.gui import learn_model_command

    learn_model_command(model_settings)
    return Path(model_settings["model_path"])


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
