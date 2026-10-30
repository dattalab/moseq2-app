"""Cell 3: analyse a different model than the one cell 1 picked.

    print('Select a model from the following models:', [key for key in model_dict.keys()])
    desired_model = best_model  # override this with the file name to the desired model, eg. "model1.p"
    progress_paths = update_model_paths(desired_model, model_dict, progress_filepath)

Marked ``[OPTIONAL]`` in the notebook, and as it ships it re-selects the
model cell 2 already selected. It earns its place when a user has fit
several models at different kappas and wants to look at each in turn: the
cell is how they switch, by typing one of the filenames it prints.

Switching is pure bookkeeping. The cell moves the five paths the rest of
the notebook writes to, and touches no model data, so these tests run
against a project of placeholder model files rather than the fixture's
single real model. What is under test is which directory the output lands
in, and that is decided by filename alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from moseq2_app.gui.progress import restore_progress_vars
from moseq2_app.util import setup_model_folders, update_model_paths

#: Everything the cell re-points. ``model_session_path`` is the folder; the
#: rest are the files and directories inside it that later cells write.
MOVED = (
    "model_session_path",
    "model_path",
    "plot_path",
    "crowd_dir",
    "syll_info",
)


@pytest.fixture(scope="module")
def several_models(tmp_path_factory):
    """A project holding two models, which is when this cell is worth running.

    The fixture bundle ships one model, so a switch has nothing to switch
    to. These are placeholders: the cell reads no model data, only the name
    on the file.
    """
    from moseq2_extract._yaml import write_yaml

    root = tmp_path_factory.mktemp("several_models")
    models = root / "models"
    models.mkdir()
    for name in ("model.p", "model_kappa_100.p"):
        (models / name).write_bytes(b"not a model, and never opened by this cell")

    progress_file = root / "progress.yaml"
    with open(progress_file, "w") as f:
        write_yaml(
            {
                "base_dir": str(root),
                "base_model_path": str(models) + "/",
                "model_path": str(models / "model.p"),
            },
            f,
        )
    return progress_file


@pytest.fixture(scope="module")
def choices(several_models):
    """What cell 2 leaves for this cell to choose between."""
    return setup_model_folders(
        restore_progress_vars(progress_file=str(several_models), init=False, overwrite=False)
    )


def test_a_user_chooses_between_filenames(choices):
    """1. What the cell prints is what it expects typed back.

    The comment tells a user to enter a file name, and the assertion inside
    ``update_model_paths`` compares against these keys directly. Keys that
    were full paths would make the instruction wrong.
    """
    assert sorted(choices) == ["model.p", "model_kappa_100.p"]
    for name in choices:
        assert Path(name).name == name, f"{name!r} is a path, not a file name"


def test_switching_moves_every_output_path_to_the_other_model(several_models, choices):
    """2. The point of the cell.

    After the switch, nothing the notebook writes may still land in the
    first model's folder: the crowd movies, the syllable labels, the
    figures and the cached dataframe all belong to the model they describe.
    """
    first = update_model_paths("model.p", choices, str(several_models))
    first = {k: first[k] for k in MOVED}

    second = update_model_paths("model_kappa_100.p", choices, str(several_models))

    folder = Path(choices["model_kappa_100.p"]["model_session_path"])
    for key in MOVED:
        assert second[key] != first[key], f"{key} did not move"
        assert Path(second[key]) == folder or folder in Path(second[key]).parents, (
            f"{key} is {second[key]}, which is not under {folder}"
        )


def test_the_switch_survives_a_kernel_restart(several_models, choices):
    """3. It is written to the progress file, not just returned.

    Cell 1 restores from this file, so a user who switches models and then
    restarts would otherwise come back pointed at the old one while the
    notebook's variables said otherwise.
    """
    update_model_paths("model_kappa_100.p", choices, str(several_models))

    restored = restore_progress_vars(
        progress_file=str(several_models), init=False, overwrite=False
    )
    chosen = choices["model_kappa_100.p"]
    assert restored["model_session_path"] == chosen["model_session_path"]
    assert restored["model_path"] == chosen["model_path"]

    folder = Path(chosen["model_session_path"])
    for key in MOVED:
        assert Path(restored[key]) == folder or folder in Path(restored[key]).parents, (
            f"{key} came back as {restored[key]}"
        )


def test_switching_back_restores_the_first_model(several_models, choices):
    """4. A user can move between models as often as they like.

    Each selection has to be computed from the chosen model rather than
    accumulated on top of the last one, or the second switch would leave a
    mixture of the two.
    """
    update_model_paths("model_kappa_100.p", choices, str(several_models))
    back = update_model_paths("model.p", choices, str(several_models))

    folder = Path(choices["model.p"]["model_session_path"])
    for key in MOVED:
        assert Path(back[key]) == folder or folder in Path(back[key]).parents, (
            f"{key} is still {back[key]}"
        )


def test_asking_for_a_model_that_is_not_there_says_which_ones_are(several_models, choices):
    """5. The error a user meets when they mistype the name.

    This is the one line of the cell a user edits, so the failure has to
    name what they typed and list what they could have typed. It stops
    before anything is written, which keeps the progress file pointing at
    the model it was pointing at.
    """
    with pytest.raises(AssertionError) as caught:
        update_model_paths("model1.p", choices, str(several_models))

    message = str(caught.value)
    assert "model1.p" in message
    for name in choices:
        assert name in message, f"the message does not offer {name}"
