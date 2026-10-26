"""Cell 17: choose how the model is fit, and where it goes.

    base_model_path = 'models/'
    model_name = 'model.p'
    session_path = join(progress_paths['base_dir'], base_model_path)
    model_path = join(session_path, model_name)

    progress_paths = update_progress(progress_filepath, 'model_path', model_path)
    progress_paths = update_progress(progress_filepath, 'base_model_path', session_path)

    with update_config(progress_paths['config_file']) as config_data:
        config_data['npcs'] = 10
        config_data['max_states'] = 100
        config_data['robust'] = True
        config_data['num_iter'] = 100
        config_data['kappa'] = None
        ...

Nothing is computed here. The cell records where the model will go and
writes the modelling parameters into the configuration cell 18 reads, which
makes it the longest cell in the notebook and the one a user actually edits.

Most of it is annotated defaults. Three choices matter: ``npcs`` decides how
much of the pose the model sees, ``max_states`` caps how many syllables it
may use, and ``kappa`` sets how long it expects them to last. Leaving kappa
as ``None`` means the frame count, which cell 18 resolves.

The cell also carries a branch the notebook leaves unentered. Setting
``kappa`` to ``'scan'`` turns cell 18 into a spooler that writes one fitting
command per kappa and, with the ``run_cmd`` the branch sets, runs all of them.
The tests exercise it as script generation, with the execution stubbed, since
running it would fit fifteen models.
"""

from __future__ import annotations

import pytest

from moseq2_app.gui.progress import restore_progress_vars
from moseq2_extract.util import read_yaml

from tests.conftest import MODEL_SETTINGS
from tests.extract_modeling.test_03_config import plain

#: How many models a scan spools, and over what scale.
SCAN = {"scan_scale": "log", "n_models": 15, "out_script": "train_out.sh"}


@pytest.fixture
def no_shell(monkeypatch):
    """Stub the shell, and fail loudly if anything reaches it."""
    calls = []
    monkeypatch.setattr(
        "moseq2_model.helpers.wrappers.os.system", lambda command: calls.append(command)
    )
    return calls


def scan_config(settings, configured, **overrides):
    """Point the configuration at a kappa scan, as the notebook's branch does."""
    from moseq2_app.util import update_config

    with update_config(str(configured)) as config:
        config["kappa"] = "scan"
        config.update(SCAN)
        config.update(overrides)
    return settings


def test_the_modelling_parameters_reach_the_configuration(model_settings, configured):
    """1. The cell's whole product: cell 18 reads these and nothing else."""
    config = plain(read_yaml(str(configured)))

    for key, value in MODEL_SETTINGS.items():
        assert config[key] == value, key


def test_the_model_paths_are_recorded(model_settings, project, progress_file):
    """2. Where the model goes, and where cell 19 looks for it.

    Both are recorded: the file for this model, and the folder, which is
    what cell 19 searches when it compares models against each other.
    """
    recorded = restore_progress_vars(str(progress_file), init=False, overwrite=False)

    assert recorded["model_path"] == str(project / "models" / "model.p")
    assert recorded["base_model_path"] == str(project / "models") + "/"
    assert model_settings["model_path"] == recorded["model_path"]


def test_a_scan_writes_one_fitting_command_per_kappa(
    model_settings, configured, project, no_shell
):
    """3. The branch the notebook ships commented out.

    A user who does not know what kappa to use sets it to 'scan' and gets a
    script of fits across a range, which they then compare in cell 19.
    """
    from moseq2_model.gui import learn_model_command

    scan_config(model_settings, configured, run_cmd=False)
    try:
        commands = learn_model_command(model_settings)
    finally:
        restore(configured)

    script = (project / "models" / "train_out.sh").read_text()
    assert script == commands
    assert script.count("learn-model") == SCAN["n_models"]
    assert no_shell == [], "a scan ran the fits instead of writing them"


def test_a_scan_spans_a_range_of_kappas(model_settings, configured, project, no_shell):
    """4. What makes the scan worth running.

    Fifteen fits at the same kappa would tell a user nothing. The values have
    to cover a range, and on a log scale, which is how kappa behaves.
    """
    import re

    from moseq2_model.gui import learn_model_command

    scan_config(model_settings, configured, run_cmd=False)
    try:
        commands = learn_model_command(model_settings)
    finally:
        restore(configured)

    kappas = [float(k) for k in re.findall(r"--kappa (\S+)", commands)]
    assert len(kappas) == SCAN["n_models"]
    assert kappas == sorted(kappas)
    assert kappas[-1] > kappas[0] * 10, f"the scan barely moves: {kappas[0]} to {kappas[-1]}"


def test_a_scan_set_to_run_hands_the_fits_to_the_shell(
    model_settings, configured, project, no_shell
):
    """5. Current behaviour, captured, because the notebook turns it on.

    The commented-out branch in cell 17 sets ``run_cmd = True``, so a user who
    uncomments it does not get a script to look over: the fifteen fits start
    immediately, in one shell call, with no confirmation.
    """
    from moseq2_model.gui import learn_model_command

    scan_config(model_settings, configured, run_cmd=True)
    try:
        learn_model_command(model_settings)
    finally:
        restore(configured)

    assert len(no_shell) == 1
    assert no_shell[0].count("learn-model") == SCAN["n_models"]


def test_an_unknown_scan_scale_is_rejected(model_settings, configured, no_shell):
    """6. The one thing the cell checks before spending an afternoon."""
    from moseq2_model.gui import learn_model_command

    scan_config(model_settings, configured, scan_scale="logarithmic", run_cmd=False)
    try:
        with pytest.raises(AssertionError, match='scan scale must be "log" or "linear"'):
            learn_model_command(model_settings)
    finally:
        restore(configured)

    assert no_shell == []


def restore(configured):
    """Put the configuration back the way cell 17 left it."""
    from moseq2_app.util import update_config

    with update_config(str(configured)) as config:
        config.update(MODEL_SETTINGS)
