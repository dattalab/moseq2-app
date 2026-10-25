"""Cell 2: create the progress file and report what is already done.

    from os.path import join
    from moseq2_app.gui.progress import check_progress, restore_progress_vars

    base_dir = './'
    progress_filepath = join(base_dir, 'progress.yaml')

    progress_paths = restore_progress_vars(progress_filepath, init=True, overwrite=False)
    check_progress(progress_filepath)

``restore_progress_vars`` does two things: it creates ``progress.yaml`` when
there is not one, and it scans the base directory to fill in whatever it can
already see. The file it creates is the notebook's state between cells. Nearly
every later cell ends by writing its output path back into it, and the
analysis notebook picks the pipeline up by reading it.

``check_progress`` only reports. It prints how many sessions are extracted,
how many of the four PCA items exist, and how many models it found.

Every question it asks about doneness is an ``exists()`` call, apart from one
``complete`` key in a per-session yaml. So a project in any state of
completion is a few empty files, and the reported states below are built that
way rather than by running the pipeline. The tests drive ``check_progress``,
the call the cell makes, and never the reporting underneath it.
"""

from __future__ import annotations

import uuid

import pytest
from ruamel.yaml import YAML

from moseq2_app.gui.progress import check_progress, restore_progress_vars, update_progress

#: The keys a freshly generated progress file carries.
EXPECTED_KEYS = {
    "base_dir",
    "config_file",
    "session_config",
    "index_file",
    "train_data_dir",
    "pca_dirname",
    "scores_filename",
    "scores_path",
    "changepoints_path",
    "model_path",
    "crowd_dir",
    "syll_info",
    "plot_path",
    "snapshot",
}

#: The four things `check_progress` counts as PCA progress.
PCA_ITEMS = ("pca_dirname", "scores_path", "changepoints_path", "index_file")


def read_progress(path):
    with open(path) as handle:
        return YAML().load(handle)


def build_project(root, n_sessions=4, n_extracted=0, n_incomplete=0, pca_items=(), n_models=0):
    """Assemble a project directory in a given state of completion.

    Sessions are discovered by globbing ``*/*.avi`` and extractions by
    globbing ``*/proc/*.mp4``, so empty files are enough to be found. The one
    thing read rather than stat'd is the per-session ``complete`` flag.
    """
    root.mkdir(parents=True, exist_ok=True)
    marked = 0
    for i in range(n_sessions):
        session = root / f"session_{i}"
        (session / "proc").mkdir(parents=True, exist_ok=True)
        (session / "depth.avi").touch()
        if marked < n_extracted + n_incomplete:
            complete = marked < n_extracted
            (session / "proc" / "results_00.mp4").touch()
            (session / "proc" / "results_00.yaml").write_text(f"complete: {complete}\n")
            marked += 1

    pca_dir = root / "_pca"
    pca_dir.mkdir(exist_ok=True)
    progress = {
        "base_dir": str(root),
        "pca_dirname": str(pca_dir),
        "scores_path": str(pca_dir / "pca_scores.h5"),
        "changepoints_path": "changepoints",
        "index_file": str(root / "moseq2-index.yaml"),
        "plot_path": str(root / "plots"),
        "snapshot": str(uuid.uuid4()),
    }
    if "pca_dirname" in pca_items:
        (pca_dir / "pca.h5").touch()
    if "scores_path" in pca_items:
        (pca_dir / "pca_scores.h5").touch()
    if "changepoints_path" in pca_items:
        (pca_dir / "changepoints.h5").touch()
    if "index_file" in pca_items:
        (root / "moseq2-index.yaml").touch()

    if n_models:
        models = root / "models"
        models.mkdir(exist_ok=True)
        progress["base_model_path"] = str(models)
        for i in range(n_models):
            (models / f"model_{i}.p").touch()

    path = root / "progress.yaml"
    with open(path, "w") as handle:
        YAML().dump(progress, handle)
    return path


# --- restore_progress_vars ------------------------------------------------


def test_creates_a_progress_file_with_the_expected_keys(progress_file):
    """A project that had only recordings now has the notebook's state file."""
    assert progress_file.exists()
    stored = read_progress(progress_file)
    assert set(stored) == EXPECTED_KEYS


def test_returned_variables_match_the_file_on_disk(project, tmp_path):
    """What the cell binds to `progress_paths` is what was written."""
    path = tmp_path / "progress.yaml"
    returned = restore_progress_vars(str(path), init=True, overwrite=False)
    assert dict(returned) == dict(read_progress(path))


def test_initial_values_are_empty_apart_from_the_three_it_derives(tmp_path):
    """Only `base_dir`, `plot_path` and `snapshot` start with a value."""
    path = tmp_path / "progress.yaml"
    stored = restore_progress_vars(str(path), init=True, overwrite=False)

    assert stored["base_dir"] == str(tmp_path)
    assert stored["plot_path"].rstrip("/").endswith("plots")
    uuid.UUID(stored["snapshot"])

    derived = {"base_dir", "plot_path", "snapshot"}
    assert all(stored[key] == "" for key in EXPECTED_KEYS - derived)


def test_reopening_without_overwrite_keeps_what_later_cells_wrote(tmp_path):
    """`init=True` on an existing file loads it; it does not start over.

    This is what makes the file usable as state: a user re-runs cell 2 after
    reopening the notebook and the paths later cells recorded are still there.
    """
    path = tmp_path / "progress.yaml"
    first = restore_progress_vars(str(path), init=True, overwrite=False)
    update_progress(str(path), "config_file", "/somewhere/config.yaml")

    again = restore_progress_vars(str(path), init=True, overwrite=False)
    assert again["config_file"] == "/somewhere/config.yaml"
    assert again["base_dir"] == first["base_dir"]
    assert again["plot_path"] == first["plot_path"]


def test_snapshot_rolls_whenever_a_value_is_written(tmp_path):
    """`snapshot` marks that the file changed; it is not a stable identifier.

    Every successful `update_progress` replaces it with a fresh uuid4, so two
    reads of the same file agree only while nothing has been written between
    them.
    """
    path = tmp_path / "progress.yaml"
    before = restore_progress_vars(str(path), init=True, overwrite=False)["snapshot"]
    after = update_progress(str(path), "config_file", "/somewhere/config.yaml")["snapshot"]
    assert after != before
    uuid.UUID(after)


def test_overwrite_starts_a_fresh_file(tmp_path):
    """`overwrite=True` discards what was recorded and takes a new snapshot."""
    path = tmp_path / "progress.yaml"
    first = restore_progress_vars(str(path), init=True, overwrite=False)
    update_progress(str(path), "config_file", "/somewhere/config.yaml")

    fresh = restore_progress_vars(str(path), init=True, overwrite=True)
    assert fresh["config_file"] == ""
    assert fresh["snapshot"] != first["snapshot"]


# --- check_progress -------------------------------------------------------


def test_reports_nothing_extracted_in_a_fresh_project(progress_file, capsys):
    """The real project at this point in the notebook: four recordings, no work done."""
    check_progress(str(progress_file))
    out = capsys.readouterr().out

    assert "Extraction Progress: 0 out of 4 session(s) extracted" in out
    for session in ("saline_example_0", "saline_example_1",
                    "amphetamine_example_0", "amphetamine_example_1"):
        assert f"Not yet extracted: {session}" in out
    assert "PCA Progress: 0 out of 4 items finished" in out
    assert "model(s)" not in out


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
    check_progress(str(path))
    assert expected in capsys.readouterr().out


def test_an_incomplete_extraction_is_named(tmp_path, capsys):
    path = build_project(tmp_path / "proj", n_extracted=0, n_incomplete=1)
    check_progress(str(path))
    assert "is listed as incomplete." in capsys.readouterr().out


@pytest.mark.parametrize("n_done", [0, 1, 3])
def test_pca_item_counts(tmp_path, capsys, n_done):
    path = build_project(tmp_path / "proj", pca_items=PCA_ITEMS[:n_done])
    check_progress(str(path))
    assert f"PCA Progress: {n_done} out of 4 items finished" in capsys.readouterr().out


def test_all_pca_items_done_reports_an_empty_remainder(tmp_path, capsys):
    """Current behaviour, captured rather than corrected.

    The branch that lists outstanding PCA items is reached only when none are
    outstanding, so the list it appends is always empty and a finished project
    reports a trailing `:  left`, with the gap where the names would go.
    """
    path = build_project(tmp_path / "proj", pca_items=PCA_ITEMS)
    check_progress(str(path))
    assert "PCA Progress: 4 out of 4 items finished:  left" in capsys.readouterr().out


@pytest.mark.parametrize("n_models", [1, 3])
def test_model_counts(tmp_path, capsys, n_models):
    path = build_project(tmp_path / "proj", n_models=n_models)
    check_progress(str(path))
    assert f"Found {n_models} model(s)" in capsys.readouterr().out


def test_a_missing_progress_file_reports_nothing(tmp_path, capsys):
    """`check_progress` on a path that does not exist prints nothing and raises nothing."""
    check_progress(str(tmp_path / "absent.yaml"))
    assert capsys.readouterr().out == ""
