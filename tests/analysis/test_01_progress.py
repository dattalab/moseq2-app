"""Cell 1: find the finished project.

    progress_filepath = './progress.yaml'
    from_CLI = False
    pretrained_model_used = False
    progress_paths = restore_progress_vars(progress_file=progress_filepath,
                                           init=from_CLI, overwrite=False)
    if from_CLI and not pretrained_model_used:
        best_model_fit = get_best_fit_model(progress_paths, output_file,
                                            plot_all=True, objective=objective)
        progress_paths = update_progress(progress_filepath, 'model_path',
                                         best_model_fit[f'best model - {objective}'])
    progress_path_sanity_check(progress_paths, progress_filepath)

The analysis notebook starts where the extraction and modeling notebook
stopped, and this cell is the handover: a single dictionary of paths, read
out of the file that notebook wrote. Nothing downstream looks at the
directory. A path that is wrong here is wrong everywhere.

``from_CLI`` is the branch for a user who ran the pipeline from the command
line and so has no progress file keyed to a chosen model. It asks
moseq2-viz which of the models in the project best matches the model-free
changepoints and writes that choice into the progress file. It is the only
caller of ``get_best_fit_model`` in either notebook.

``pretrained_model_used`` is not covered: it is for a model produced by
applying an existing model to new recordings, which this project does not
contain.

The fixture holds one model, so the comparison runs over a single candidate
and the ranking itself is not exercised. What the branch's tests establish
is that it loads the model, reaches the changepoints, computes the
objective, draws the figure and records the result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from moseq2_app.gui.progress import (
    progress_path_sanity_check,
    restore_progress_vars,
    update_progress,
)

#: The keys later cells index out of what this cell returns. Each is read by
#: name, so a missing one is an immediate KeyError rather than a bad result.
REQUIRED = {
    "base_dir": "cell 1 builds the comparison figure's path from it",
    "config_file": "the extraction parameters every scalar is expressed in",
    "index_file": "cell 4 parses it to find the sessions",
    "train_data_dir": "where the aggregated extractions are",
    "pca_dirname": "cell 1's CLI branch resolves the changepoints against it",
    "scores_path": "cell 4 reads scores_idx out of it",
    "changepoints_path": "what the CLI branch scores a model against",
    "model_path": "cell 2 names the model folder after it",
    "base_model_path": "cell 2 globs it for models",
}

#: Keys whose value is a path into the project. ``changepoints_path`` is not
#: one: the pipeline leaves it as the bare stem ``changepoints``, which
#: moseq2-viz joins onto ``pca_dirname`` itself.
PATHS = tuple(k for k in REQUIRED if k != "changepoints_path")

OBJECTIVE = "median_loglikelihood"


def test_the_cell_returns_the_paths_the_rest_of_the_notebook_reads(analysis_progress):
    """1. The handover carries everything the later cells ask it for."""
    missing = {k: why for k, why in REQUIRED.items() if k not in analysis_progress}
    assert not missing, f"the progress file is missing keys later cells read: {missing}"


def test_every_path_it_hands_over_resolves(analysis_progress):
    """2. The paths name things that are actually there.

    The progress file is written with absolute paths into the directory the
    pipeline ran in, so a project that has been moved or copied carries a
    dictionary of paths to nothing. Every cell downstream reads these.
    """
    from pathlib import Path

    broken = {k: analysis_progress[k] for k in PATHS if not Path(analysis_progress[k]).exists()}
    assert not broken, f"the progress file names paths that do not exist: {broken}"


def test_a_progress_file_that_is_not_there_comes_back_as_none(tmp_path, capsys):
    """3. What a user who mistypes the path gets.

    ``progress_filepath`` is the one thing in the cell a user edits by hand.
    With ``from_CLI`` False, ``init`` is False too, so nothing is created:
    the function prints a message, returns None, and the cell carries on to
    hand None to the sanity check. The error a user sees is a TypeError
    several lines later, about subscripting None, which names neither the
    path nor the file.
    """
    restored = restore_progress_vars(
        progress_file=str(tmp_path / "nope.yaml"), init=False, overwrite=False
    )
    assert restored is None
    assert "Progress file not found" in capsys.readouterr().out

    with pytest.raises(TypeError):
        restored["base_dir"]


def test_coming_from_the_cli_chooses_a_model_and_draws_the_comparison(cli_best_fit):
    """4. The ``from_CLI`` branch runs and produces what the cell names.

    moseq2-viz loads every model in the project, turns each one's labels
    into changepoints, compares those against the model-free changepoints
    computed from the PC scores, and reports the closest under the chosen
    objective. ``plot_all=True`` draws every model's duration distribution
    against the model-free one, which is the figure a user picks a kappa
    from.
    """
    best, figure, _ = cli_best_fit

    key = f"best model - {OBJECTIVE}"
    assert key in best, f"no {key!r} in {sorted(best)}"
    assert best[key].endswith(".p"), f"the chosen model is {best[key]!r}"

    assert figure.exists(), f"the cell names {figure.name} but nothing was written"
    assert figure.stat().st_size > 0, f"{figure.name} is empty"


def test_coming_from_the_cli_writes_the_choice_into_the_progress_file(cli_best_fit):
    """5. The choice survives the cell.

    Cell 2 reads ``model_path`` back and names the model's folder after it,
    so a branch that chose a model without recording it would leave every
    later cell pointed at whatever was there before.
    """
    _, _, progress = cli_best_fit

    restored = restore_progress_vars(progress_file=str(progress), init=False, overwrite=False)
    assert restored["model_path"].endswith(".p")


@pytest.fixture(scope="module")
def cli_best_fit(analysis_project, tmp_path_factory):
    """Cell 1 with ``from_CLI`` set, run against its own progress file.

    The branch reads the project's models, scores and changepoints, and
    writes two things: the comparison figure, under ``base_dir``, and the
    chosen model, into the progress file. Giving it a copy of the progress
    file in its own directory keeps both writes out of the project the rest
    of the suite shares, without changing anything the branch reads.
    """
    from pathlib import Path

    from moseq2_viz.gui import get_best_fit_model

    root = tmp_path_factory.mktemp("from_cli")
    progress_file = root / "progress.yaml"
    source = analysis_project / "progress.yaml"
    progress_file.write_text(source.read_text().replace(str(analysis_project), str(root), 1))

    paths = restore_progress_vars(progress_file=str(progress_file), init=True, overwrite=False)
    output_file = str(Path(paths["base_dir"]) / "model_vs_pc_changepoints")
    best = get_best_fit_model(paths, output_file, plot_all=True, objective=OBJECTIVE)
    update_progress(str(progress_file), "model_path", best[f"best model - {OBJECTIVE}"])

    return best, Path(output_file + ".png"), progress_file


def _break(progress, root, what):
    """A copy of the project's paths with one input the analysis cells read gone.

    The index is copied before it is edited, so the shared project stays whole.
    """
    from moseq2_app._yaml import write_yaml
    from moseq2_viz.util import read_yaml

    broken = dict(progress)
    if what == "model":
        broken["model_path"] = str(root / "models" / "gone.p")
        return broken, "gone.p"

    index = read_yaml(progress["index_file"])
    if what == "pca":
        index["pca_path"] = "_pca/gone.h5"
        missing = "gone.h5"
    else:
        index["files"][0]["path"][0] = "proc/gone/results_00.h5"
        missing = "results_00.h5"
    copy = Path(progress["index_file"]).parent / f"moseq2-index-{what}.yaml"
    write_yaml(index, str(copy))
    broken["index_file"] = str(copy)
    return broken, missing


@pytest.mark.parametrize("what", ["model", "pca", "session"])
def test_the_sanity_check_notices_an_input_that_is_gone(analysis_progress, analysis_project, capsys, what):
    """6. What the cell's comment promises: ``# check all necessary paths exist``.

    A moved or cleaned project fails several cells later, on a file whose
    name says little about why. The check opens nothing but asks whether each
    input the analysis cells read is there: the model and its folder, the
    index and config named in the progress file, and the PCA scores and
    extraction results the index names.
    """
    broken, missing = _break(analysis_progress, analysis_project, what)
    try:
        progress_path_sanity_check(broken, str(analysis_project / "progress.yaml"))
    finally:
        if what != "model":
            Path(broken["index_file"]).unlink()

    printed = capsys.readouterr().out
    assert missing in printed
    assert len(printed.strip().splitlines()) == 1, printed


def test_the_sanity_check_is_quiet_about_a_healthy_project(analysis_project, analysis_progress, capsys):
    """7. The other half.

    This project is the one the extraction and modeling notebook produces
    and every later cell reads without complaint, so the check says nothing.
    """
    progress_path_sanity_check(analysis_progress, str(analysis_project / "progress.yaml"))
    assert capsys.readouterr().out == ""
