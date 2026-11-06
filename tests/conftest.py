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

from fixtures import fetch_analysis_bundle, fetch_raw_sessions
from widgets import capture_display


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
def analysis_project(tmp_path_factory, analysis_bundle):
    """A writable copy of the finished project the analysis notebook opens.

    This is where the analysis notebook's user starts, and it is a different
    starting point from ``project``: twenty sessions aggregated, decomposed
    and modeled, rather than four recordings and nothing else. The extraction
    and modeling notebook produced it, so the suite reads it rather than
    building it. Eighty seconds of mouse yields syllables that are noise, and
    the dendrogram, the transition matrix and the group comparisons are only
    worth testing against syllables that are not.

    The bundle is a project 1.5.0 made; ``project_1_6_from_bundle`` lays the
    copy out as a 1.6 project, whose ``moseq2-project.yaml`` holds every path
    relative to the project folder, so the copy works wherever it landed.
    The analysis cells write into it.
    """
    from fixtures import project_1_6_from_bundle

    return project_1_6_from_bundle(analysis_bundle, tmp_path_factory.mktemp("analysis"))


@pytest.fixture(scope="session")
def analysis_project_1_6(tmp_path_factory, analysis_bundle):
    """A second copy of the finished project, for the ``moseq2-viz`` commands.

    Shared and read-only: a test that writes into it takes its own copy.
    """
    from fixtures import project_1_6_from_bundle

    return project_1_6_from_bundle(analysis_bundle, tmp_path_factory.mktemp("analysis_1_6"))


@pytest.fixture(scope="session")
def analysis_started(analysis_project):
    """Cell 1: name the project and show what it holds.

    Every later cell takes the project folder and reads what it needs from
    the project's ``moseq2-project.yaml``, so this is the only cell that
    knows where the project is.
    """
    from moseq2_app.project import start

    return start(str(analysis_project))


@pytest.fixture(scope="session")
def analysis(analysis_started):
    """Cell 2: the model the notebook analyses, and the folder its analysis goes in.

    The project's current model, as the extraction and modeling notebook left
    it. Everything the analysis notebook produces afterwards belongs to that
    model: the figures, the crowd movies, the syllable labels and the
    dataframes, all inside its analysis folder.
    """
    from moseq2_app.util import choose_model

    return choose_model(str(analysis_started.folder))


@pytest.fixture(scope="session")
def sorted_index(analysis):
    """The project's sessions, by uuid, as the analysis reads them.

    Built from ``moseq2-project.yaml`` and ``config.yaml``'s groups. Everything
    downstream addresses sessions by uuid, and this is where that mapping
    comes from.
    """
    from moseq2_common.bookkeeping import session_index
    from moseq2_viz.util import parse_index

    _, index = parse_index(session_index(analysis.project))
    return index


@pytest.fixture(scope="session")
def made_df(analysis):
    """Cell 4, first half: ``make-df``'s step writes the dataframe and its statistics.

    Both go into the model's analysis folder as parquet, which reads back
    with the types they were written with.
    """
    from moseq2_common.bookkeeping import session_index
    from moseq2_viz.steps import make_df

    return make_df(str(analysis.model), session_index(analysis.project), analysis.folder / "moseq_df.parquet")


@pytest.fixture(scope="session")
def moseq_df(made_df):
    """Cell 4, second half: one row per frame, scalars beside syllable labels.

    The join the whole notebook rests on. Each recording's extraction
    scalars are read out of its h5 and lined up with the syllable the model
    assigned to that frame, so that every later question (which syllables
    does this group use, what is the mouse doing during one) is a groupby
    on this table.

    Big enough to be worth building once: seven hundred thousand rows over
    the fixture's twenty sessions.
    """
    import pandas as pd

    return pd.read_parquet(made_df.dataframe_file)


#: What ``make-df``'s step computes its statistics with: usage as a probability
#: rather than a raw count, one row per session per syllable, syllables
#: numbered by how much they are used.
STATISTICS = {
    "count": "usage",
    "groupby": ["group", "uuid"],
    "usage_normalization": True,
    "syllable_key": "labels (usage sort)",
}


@pytest.fixture(scope="session")
def stats_df(made_df):
    """Cell 6: one row per session per syllable, as ``make-df``'s step wrote it.

    Collapses the frame-by-frame table into the unit every later cell
    works in. How much each session used each syllable, how long its bouts
    lasted, and what the mouse's scalars looked like while it was doing
    it. The dendrogram, the fingerprint, the hypothesis tests and the
    transition graphs all read this.
    """
    import pandas as pd

    return pd.read_parquet(made_df.stats_file)


#: What cell 9 asks for. ``max_syllables=None`` makes the cell work out how
#: many syllables are worth looking at instead of being told. How the crowd
#: movies are made is config.yaml's ``crowd_movies`` section.
LABELLING = {
    "max_syllables": None,
    "n_explained": 99,
}


@pytest.fixture(scope="session")
def labeled(analysis):
    """Cell 9: build a crowd movie for every syllable, and open the labeller.

    Up to here a syllable is a number. This is the cell that makes it
    something a person can look at: for each one, a movie of twenty
    instances of it played together, so that what the mouse is doing
    during that syllable is visible rather than inferred from scalars.

    Generating those movies is the longest step in the notebook. It runs
    once, and the cells after it read what it wrote.

    Returns the number of syllables the cell settled on, which cells 10,
    11, 12 and 14 all take as their ``max_syllables``.
    """
    from moseq2_app.main import label_syllables

    return label_syllables(str(analysis.project.folder), **LABELLING)


#: How cell 12 normalises its transition matrices: every entry is the share
#: of all transitions that went from one syllable to another.
TRANSITION_NORMALIZE = "bigram"


@pytest.fixture(scope="session")
def transitions(analysis, labeled):
    """Cell 12's computing half: a matrix and a usage count per group.

    Usage says how often a syllable happens; a transition matrix says
    what it follows and what follows it. Two groups can use the same
    syllables equally often and string them together quite differently.

    Cell 12 draws these and cell 13 exports them, so the computation is
    shared.
    """
    from moseq2_viz.model.trans_graph import get_group_trans_mats, get_trans_graph_groups
    from moseq2_viz.model.util import parse_model_results, relabel_by_usage

    model_data = parse_model_results(str(analysis.model))
    model_data["labels"] = relabel_by_usage(model_data["labels"], count="usage")[0]

    label_group, uuids = get_trans_graph_groups(model_data)
    group = sorted(set(label_group))
    trans_mats, usages = get_group_trans_mats(
        model_data["labels"], label_group, group, labeled, normalize=TRANSITION_NORMALIZE
    )
    return {
        "group": group,
        "label_group": label_group,
        "uuids": uuids,
        "trans_mats": trans_mats,
        "usages": usages,
    }


@pytest.fixture(scope="session")
def arena(project, started, configured, flipped):
    """Cell 5: the per-recording parameters extraction reads.

    The cell opens the arena mask widget on the project folder. Opening it
    writes nothing: the widget shows each recording's settings, with its
    camera parameters detected. Clicking "save session parameters and move to
    next" once per recording is what the walkthrough did; each click saves
    that recording's arena settings as its entry in ``session_config.yaml``,
    which is the state the walkthrough's own session config is in. Advancing
    also computes each recording's arena mask and a short test extraction, so
    the chain reaches cell 6 having already exercised both.
    """
    import panel as pn

    from moseq2_app.roi.widget import ArenaMaskWidget

    from widgets import buttons_of, click

    pn.extension()

    widget = ArenaMaskWidget(str(project))
    buttons = buttons_of(widget.view)
    for _ in widget.sessions:
        click(buttons["Save session parameters and move to next"])
    return project / "session_config.yaml"


@pytest.fixture(scope="session")
def aggregation(tmp_path_factory, project, started, extracted):
    """Cell 9: list every extraction as one of the project's sessions.

    Extraction has written each recording's files into its own folder in
    ``aggregate_results/``; the cell copies nothing and lists each extracted
    recording as one of the project's sessions in ``moseq2-project.yaml``.
    That file is what every cell from here on reads its sessions from.

    Returns the bookkeeping file and a copy of it as this cell wrote it, since
    later cells add their outputs to it and the chain may already have gone
    past them when a test describing this cell runs.
    """
    from moseq2_extract.gui import aggregate_extract_results_command

    aggregate_extract_results_command(str(project))
    bookkeeping = project / "moseq2-project.yaml"

    as_written = tmp_path_factory.mktemp("aggregated") / "moseq2-project.yaml"
    shutil.copy2(bookkeeping, as_written)
    return bookkeeping, as_written


@pytest.fixture(scope="session")
def aggregated(aggregation):
    """Cell 9's ``moseq2-project.yaml`` in the project, as later cells change it."""
    return aggregation[0]


@pytest.fixture(scope="session")
def bookkeeping_as_aggregated(aggregation):
    """A copy of ``moseq2-project.yaml`` as cell 9 wrote it: the sessions and nothing else."""
    return aggregation[1]


@pytest.fixture(scope="session")
def extracted(project, configured, arena):
    """Cell 6: extract every recording in the project.

    The heaviest cell, and the one the rest of the notebook is built on. It
    searches the project for depth files, hands them to a local extraction
    loop, and leaves a ``proc`` directory beside each recording.

    ``skip_extracted`` carries over from cell 5, where the notebook sets it,
    so a second run of the cell extracts nothing.
    """
    from moseq2_extract.gui import extract_found_sessions

    extract_found_sessions(
        str(project),
        str(configured),
        extract_all=True,
        skip_extracted=True,
        run_cmd=False,
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
    with update_config(str(configured), section="extract") as config:
        config["flip_classifier"] = str(classifier)
    return classifier


@pytest.fixture(scope="session")
def grouped(project, aggregated):
    """Cell 10: label each recording with the condition it belongs to.

    Until this runs every recording is in group ``default`` and a group
    comparison compares the experiment against itself. The cell shows an
    editable table; a user selects rows, types a name, sets it, and saves the
    groups into config.yaml. The chain does the same for the two conditions
    the fixture recordings come from.

    Returns the project folder.
    """
    from moseq2_app.main import interactive_group_setting

    widget = interactive_group_setting(str(project))
    for prefix, group in (("saline", "saline"), ("amphetamine", "amph")):
        table = widget.qgrid_widget.get_changed_df()
        widget.qgrid_widget.change_selection(
            [i for i, name in zip(table.index, table["SessionName"]) if name.startswith(prefix)]
        )
        widget.group_input.value = group
        widget.save_button.click()
    widget.save_groups_button.click()
    return project


@pytest.fixture(scope="session")
def scalar_viewer(grouped):
    """Cell 11: the scalar summary, and the viewer cell 12 reads out of it.

    Reading every recording's scalars takes a moment, and the viewer is
    immutable once built, so the whole suite shares one.
    """
    from moseq2_app.main import interactive_scalar_summary

    return interactive_scalar_summary(str(grouped))


@pytest.fixture(scope="session")
def pca(project, started, configured, grouped):
    """Cell 13: learn the pose basis every later step works in.

    Eighty by eighty pixels of mouse is too much to model directly, so the
    aggregated frames are reduced to a handful of components describing how a
    mouse's shape varies. Everything from here on is expressed in that basis
    rather than in pixels.

    Runs under dask on the ``pca`` section's settings, which the cell leaves
    at the defaults config.yaml holds.
    """
    from moseq2_pca.gui import train_pca_command

    train_pca_command(str(project), "pca")
    return project / "_pca"


@pytest.fixture(scope="session")
def scores(project, pca):
    """Cell 15: express every frame in the basis cell 13 learned.

    The decomposition describes how a mouse's shape varies; this projects
    each recording's frames onto it, turning 6400 pixels per frame into a
    couple of dozen coordinates. The model is fit on these.
    """
    from moseq2_pca.gui import apply_pca_command

    apply_pca_command(str(project), "pca_scores")
    return pca / "pca_scores.h5"


@pytest.fixture(scope="session")
def changepoints(project, configured, scores):
    """Cell 16: find where the mouse's pose changes, without a model.

    A model-free estimate of how long a behavioural unit lasts, computed from
    the PC scores alone. Cell 19 scores a fitted model against it, which is
    how a kappa is chosen, so this is the yardstick rather than an output.
    """
    from moseq2_pca.gui import compute_changepoints_command

    compute_changepoints_command(str(project), "changepoints")
    return scores.parent / "changepoints.h5"


#: The model section's values cell 17 shows (it sets ``robust`` and ``nfolds``; the rest are the defaults
#: config.yaml already holds).
MODEL_SETTINGS = {
    "checkpoint_freq": -1,
    "use_checkpoint": False,
    "npcs": 10,
    "max_states": 100,
    "robust": True,
    "save_model": True,
    "separate_trans": False,
    "num_iter": 100,
    "kappa": None,
    "hold_out": False,
    "nfolds": 2,
    "cluster_type": "local",
    "ncpus": 0,
}


@pytest.fixture(scope="session")
def model_settings(project, configured, changepoints):
    """Cell 17: choose how the model is fit.

    Nothing is computed here. The cell writes the modelling parameters into
    the configuration cell 18 reads. Returns the project folder.
    """
    from moseq2_app.util import update_config

    with update_config(str(configured), section="model") as config:
        config.update(MODEL_SETTINGS)
    return project


@pytest.fixture(scope="session")
def model(model_settings):
    """Cell 18: fit the AR-HMM, into the project's ``models/model.p``.

    The heaviest cell in the notebook. One fit, shared by the whole suite.
    """
    from moseq2_model.gui import learn_model_command

    learn_model_command(str(model_settings))
    return model_settings / "models" / "model.p"


#: The kappa the second model is fit at, so cell 19 has a real choice to
#: make. Well away from the first, whose kappa is the frame count.
SECOND_KAPPA = 100


@pytest.fixture(scope="session")
def two_models(project, configured, model):
    """A second model beside the first, fit at a different kappa.

    Cell 19 picks the model whose syllable durations best match the
    changepoints. With one model in the folder it has nothing to choose
    between, so the chain fits a second at a kappa far from the first, which
    is what a user comparing kappas would have.
    """
    from moseq2_app.util import update_config
    from moseq2_model.gui import learn_model_command

    second = model.parent / f"model_kappa_{SECOND_KAPPA}.p"
    with update_config(str(configured), section="model") as config:
        config["kappa"] = SECOND_KAPPA
    try:
        learn_model_command(str(project), second.name)
    finally:
        with update_config(str(configured), section="model") as config:
            config["kappa"] = MODEL_SETTINGS["kappa"]
    return {"model.p": model, second.name: second}


@pytest.fixture(scope="session")
def configured(project, started):
    """Cell 3: the config file every later step reads its parameters from."""
    from moseq2_extract.gui import generate_config_command

    config = project / "config.yaml"
    generate_config_command(str(config), camera_type="auto")
    return config


@pytest.fixture(scope="session")
def started(project):
    """Cell 2: name the project folder and show what it holds.

    Run in a project that holds nothing but recordings, which is where a user
    starts. Every later cell takes the project folder and reads what earlier
    cells recorded from its ``moseq2-project.yaml``.
    """
    from moseq2_app.project import start

    return start(str(project))


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
