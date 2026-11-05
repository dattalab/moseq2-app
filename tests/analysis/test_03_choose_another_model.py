"""Cell 3: analyse a different model than the current one.

    desired_model = 'model.p'
    analysis = choose_model(str(project.folder), desired_model)

Marked ``[OPTIONAL]`` in the notebook. It earns its place when a user has
fit several models at different kappas and wants to look at each in turn:
the cell is how they switch, by typing a model's file name.

Switching is bookkeeping. The cell makes the model the project's current
one and gives its analysis a folder, and touches no model data, so these
tests run against a project of placeholder model files. What is under test
is which model the later cells read and which folder their output lands
in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from moseq2_app.util import choose_model, current_analysis
from moseq2_common.bookkeeping import BookkeepingError, load
from moseq2_common.yaml import write_yaml


@pytest.fixture
def several_models(tmp_path):
    """A project holding two fitted models, and a third fit by a kappa-scan job, which records nothing.

    The model files are placeholders: the cell reads no model data, only
    the name on the file.
    """
    root = tmp_path / "project"
    models = root / "models"
    models.mkdir(parents=True)
    for name in ("model.p", "model_kappa_100.p", "model_kappa_1000.p"):
        (models / name).write_bytes(b"not a model, and never opened by this cell")
    write_yaml({"sessions": {}, "pca": {},
                "models": {"fitted": ["models/model.p", "models/model_kappa_100.p"], "current": "models/model.p",
                           "analysis": {}},
                "applied_models": []}, str(root / "moseq2-project.yaml"))
    return root


def test_a_user_switches_by_file_name(several_models):
    """1. The later cells read the model typed, and write into its folder."""
    analysis = choose_model(str(several_models), "model_kappa_100.p")
    assert analysis.model == several_models / "models" / "model_kappa_100.p"
    assert analysis.folder == several_models / "models" / "model_kappa_100"
    assert current_analysis(str(several_models)) == analysis


def test_the_switch_survives_a_kernel_restart(several_models):
    """2. It is written to ``moseq2-project.yaml``, not just returned."""
    choose_model(str(several_models), "model_kappa_100.p")
    models = load(current_analysis(str(several_models)).project).models
    assert models["current"] == "models/model_kappa_100.p"
    assert models["analysis"]["models/model_kappa_100.p"] == "models/model_kappa_100"


def test_switching_back_restores_the_first_model(several_models):
    """3. A user can move between models as often as they like."""
    first = choose_model(str(several_models), "model.p")
    choose_model(str(several_models), "model_kappa_100.p")
    assert choose_model(str(several_models), "model.p") == first
    assert current_analysis(str(several_models)) == first


def test_a_model_no_step_recorded_can_be_chosen(several_models):
    """4. A kappa scan's fits run as jobs, which record nothing; the model's file is enough."""
    analysis = choose_model(str(several_models), "model_kappa_1000.p")
    assert analysis.model == several_models / "models" / "model_kappa_1000.p"
    assert "models/model_kappa_1000.p" in load(analysis.project).models["fitted"]


def test_asking_for_a_model_that_is_not_there_says_which_ones_are(several_models):
    """5. The error a user meets when they mistype the name.

    This is the one line of the cell a user edits, so the failure names what
    they typed and lists what they could have typed. It stops before
    anything is written.
    """
    bookkeeping = several_models / "moseq2-project.yaml"
    before = bookkeeping.read_bytes()
    with pytest.raises(BookkeepingError) as caught:
        choose_model(str(several_models), "model1.p")

    message = str(caught.value)
    assert "model1.p" in message
    for name in ("models/model.p", "models/model_kappa_100.p", "models/model_kappa_1000.p"):
        assert name in message, f"the message does not offer {name}"
    assert bookkeeping.read_bytes() == before


def test_a_project_with_no_model_yet_says_how_to_get_one(tmp_path):
    """6. The analysis needs a fitted model; the message says how to fit or name one."""
    write_yaml({"sessions": {}}, str(tmp_path / "moseq2-project.yaml"))
    with pytest.raises(BookkeepingError) as caught:
        choose_model(str(tmp_path))
    message = str(caught.value)
    assert str(Path(tmp_path) / "moseq2-project.yaml") in message
    assert "learn-model" in message
