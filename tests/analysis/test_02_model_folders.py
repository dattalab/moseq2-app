"""Cell 2: give the chosen model a folder of its own.

    best_model = basename(progress_paths['model_path'])
    model_dict = setup_model_folders(progress_paths)
    progress_paths = update_model_paths(best_model, model_dict, progress_filepath)

A project can hold several models, fit at different kappas and left
together in ``models/``. Everything the analysis notebook produces belongs
to exactly one of them: the crowd movies, the syllable labels, the figures,
the dataframes. Keeping them apart is this cell's whole job.

It does that by giving each model a directory named after it, copying the
model inside, and then writing one model's paths into the progress file.
From here on ``model_session_path`` is where output goes, and the four
paths derived from it are what cells 5, 7, 8, 9 and 12 write to.

The copy matters: the model stays in ``models/`` as well, which is what
lets this cell be run again, and what lets cell 1's ``from_CLI`` branch
keep finding every candidate in one place.
"""

from __future__ import annotations

from pathlib import Path

from moseq2_app.util import setup_model_folders

#: What ``update_model_paths`` writes into the progress file, and the cell
#: that reads each one back.
DERIVED = {
    "model_session_path": "cells 5, 7 and 13 export their csv files into it",
    "model_path": "cell 4 passes it to scalars_to_dataframe",
    "plot_path": "cells 8 and 12 save their figures there",
    "crowd_dir": "cell 9 writes one movie per syllable into it",
    "syll_info": "cell 9 writes the labels a user types into it",
}


def test_every_model_in_the_project_gets_a_folder(analysis_progress, model_dict):
    """1. One directory per model, named after the model.

    A user comparing kappas has several models in ``models/``. Each has to
    end up somewhere of its own or the second one's crowd movies overwrite
    the first one's.
    """
    models = sorted(p.name for p in Path(analysis_progress["base_model_path"]).glob("*.p"))
    assert models, "the project holds no models, so this cell has nothing to do"
    assert sorted(model_dict) == models

    for name, paths in model_dict.items():
        folder = Path(paths["model_session_path"])
        assert folder.is_dir(), f"{name} has no folder"
        assert folder.name == Path(name).stem


def test_the_model_is_copied_into_its_folder_and_left_where_it_was(
    analysis_progress, model_dict
):
    """2. Copied, not moved.

    ``models/`` is what cell 1's CLI branch globs for candidates, so a cell
    that moved each model into its own folder would leave that branch with
    nothing to choose between the second time it ran.
    """
    for name, paths in model_dict.items():
        original = Path(analysis_progress["base_model_path"]) / name
        copy = Path(paths["model_path"])
        assert original.exists(), f"{name} was moved out of models/"
        assert copy.exists(), f"{name} was not copied into its folder"
        assert copy.stat().st_size == original.stat().st_size


def test_the_progress_file_now_names_everything_the_later_cells_write(analysis_paths):
    """3. The three output paths, and the model they belong to."""
    missing = {k: why for k, why in DERIVED.items() if k not in analysis_paths}
    assert not missing, f"cell 2 did not record: {missing}"


def test_every_output_path_sits_inside_the_model_folder(analysis_paths):
    """4. Why the cell exists.

    If any of these resolved outside the model's own directory, two models
    analysed in the same project would write over each other.
    """
    session = Path(analysis_paths["model_session_path"])
    for key in DERIVED:
        if key == "model_session_path":
            continue
        assert session in Path(analysis_paths[key]).parents, (
            f"{key} is {analysis_paths[key]}, which is not under {session}"
        )


def test_the_choice_is_written_to_disk_not_just_returned(analysis_project, analysis_paths):
    """5. A user who restarts the kernel and re-runs cell 1 keeps it.

    The progress file is the notebook's memory. Cell 1 reads it back on
    every run, so a choice held only in the returned dictionary would be
    lost the moment the kernel restarted.
    """
    from moseq2_app.gui.progress import restore_progress_vars

    restored = restore_progress_vars(
        progress_file=str(analysis_project / "progress.yaml"), init=False, overwrite=False
    )
    for key in DERIVED:
        assert restored[key] == analysis_paths[key], f"{key} was not saved"


def test_running_the_cell_again_changes_nothing(analysis_paths, model_dict):
    """6. Re-running the notebook from the top is safe.

    A user who restarts the kernel runs cells 1 and 2 again against a
    project that now has the folders. The cell skips the directory it would
    have made and the copy it would have taken, so the second run has to
    land on the same answer rather than nesting a folder inside a folder.
    """
    again = setup_model_folders(analysis_paths)
    assert again == model_dict


def test_the_crowd_movie_and_figure_paths_are_directories(analysis_paths):
    """7. Cell 9 joins a filename onto one and cell 12 makedirs the other.

    Both are built with a trailing separator. ``crowd_dir`` is joined
    against a movie name, and ``plot_path`` is passed to ``makedirs``, so a
    value that named a file rather than a directory would break both.
    """
    for key in ("crowd_dir", "plot_path"):
        assert analysis_paths[key].endswith("/"), f"{key} is {analysis_paths[key]!r}"
        assert not Path(analysis_paths[key]).is_file()
