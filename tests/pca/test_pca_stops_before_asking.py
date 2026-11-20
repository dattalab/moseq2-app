"""A PCA step that cannot go ahead stops before it asks whether to overwrite.

In a project, ``train-pca`` and ``apply-pca`` ask whether to write over an
output that already exists (with ``overwrite_pca_*`` off) before they print
anything. The stops that print nothing come before that question, in the
notebook and on the command line alike: a project whose ``moseq2-project.yaml``
lists no sessions yet or cannot be read, when the step reads the project's
sessions; and, for ``apply-pca``, components trained with another value of a
setting the ``pca`` section of ``config.yaml`` sets. The person is not asked a
question the step cannot act on, and answering "n" cannot hide the problem.
The command line's message adds how to read a folder of extractions instead.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from moseq2_common.bookkeeping import BookkeepingError, sessions_folder
from moseq2_common.project import Project
from moseq2_common.yaml import read_yaml, write_yaml
from moseq2_pca.helpers.data import TRAINED_SETTINGS, TrainedSettingsError, write_trained_settings

#: What each step writes into the project's ``_pca/``, and its notebook function.
OUTPUTS = {"train-pca": "pca.h5", "apply-pca": "pca_scores.h5", "compute-changepoints": "changepoints.h5"}
NOTEBOOK = {"train-pca": "train_pca_command", "apply-pca": "apply_pca_command",
            "compute-changepoints": "compute_changepoints_command"}

#: What the command line adds to the message of a project that lists no sessions.
INPUT_DIR_HINT = "To read a folder of extractions instead, pass it with --input-dir."

EARLIER = b"an earlier run's output"

#: (session key, folder in aggregate_results/, uuid, start time)
SESSIONS = (
    ("cohort/a/depth.avi", "cohort_a", "uuid-a", "2020-01-01T10:00:00.000000-05:00"),
    ("cohort/b/depth.avi", "cohort_b", "uuid-b", "2020-01-02T10:00:00.000000-05:00"),
)


def asked(*prompt):
    raise AssertionError("the step asked whether to overwrite an output")


@pytest.fixture
def root(tmp_path):
    """A project folder with its config.yaml and an earlier run's output of every PCA step in ``_pca/``."""
    from moseq2_extract.gui import generate_config_command

    root = tmp_path / "project"
    root.mkdir()
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    (root / "_pca").mkdir()
    for name in OUTPUTS.values():
        (root / "_pca" / name).write_bytes(EARLIER)
    return root


def outputs(root):
    return {name: (root / "_pca" / name).read_bytes() for name in OUTPUTS.values()}


# ------------------------------------------------------------------ a project that lists no sessions


def lists_no_sessions(root):
    write_yaml({"sessions": {}}, str(root / "moseq2-project.yaml"))


def cannot_be_read(root):
    (root / "moseq2-project.yaml").write_text("- not the sessions\n- and outputs\n")


def sessions_stop(root) -> str:
    """The message of the project's stop on its sessions, as ``moseq2_common.bookkeeping`` words it."""
    with pytest.raises(BookkeepingError) as raised:
        sessions_folder(Project.at(root))
    return str(raised.value)


@pytest.mark.parametrize("bookkeeping", [lists_no_sessions, cannot_be_read])
@pytest.mark.parametrize("step", list(OUTPUTS))
def test_the_notebook_stops_on_the_sessions_before_asking(root, monkeypatch, capsys, step, bookkeeping):
    import moseq2_pca.gui as gui

    bookkeeping(root)
    expected = sessions_stop(root)
    monkeypatch.setattr("builtins.input", asked)
    capsys.readouterr()

    with pytest.raises(BookkeepingError) as raised:
        getattr(gui, NOTEBOOK[step])(str(root))

    assert str(raised.value) == expected
    assert capsys.readouterr().out == ""
    assert outputs(root) == {name: EARLIER for name in OUTPUTS.values()}


@pytest.mark.parametrize("bookkeeping", [lists_no_sessions, cannot_be_read])
@pytest.mark.parametrize("step", list(OUTPUTS))
def test_the_command_line_stops_on_the_sessions_as_before(root, step, bookkeeping):
    from moseq2_pca.cli import cli

    bookkeeping(root)
    expected = sessions_stop(root)

    result = CliRunner().invoke(cli, [step, "--config-file", str(root / "config.yaml")])

    assert result.exit_code == 1, result.output
    assert result.output == f"Error: {expected} {INPUT_DIR_HINT}\n"
    assert outputs(root) == {name: EARLIER for name in OUTPUTS.values()}


# ------------------------------------------------------------------ components trained with other settings


@pytest.fixture
def trained(root):
    """``root`` with two sessions listed and components trained with config.yaml's settings, recorded in
    ``moseq2-project.yaml``; config.yaml's ``max_height`` then set to another value."""
    rng = np.random.default_rng(0)
    for key, folder, uuid, start in SESSIONS:
        path = root / "aggregate_results" / folder
        path.mkdir(parents=True)
        with h5py.File(path / "results.h5", "w") as f:
            f["frames"] = rng.integers(0, 60, (20, 80, 80)).astype("uint8")
            f["timestamps"] = np.arange(20) * 33.3
            f["metadata/uuid"] = uuid
            f["metadata/acquisition/SessionName"] = key.split("/")[1]
            f["metadata/acquisition/StartTime"] = start
    write_yaml({"sessions": {key: {"recording": key, "folder": f"aggregate_results/{folder}", "uuid": uuid}
                             for key, folder, uuid, _ in SESSIONS},
                "pca": {"components": "_pca/pca.h5"}},
               str(root / "moseq2-project.yaml"))
    config = read_yaml(str(root / "config.yaml"))
    (root / "_pca" / "pca.h5").unlink()
    with h5py.File(root / "_pca" / "pca.h5", "w") as f:
        f["components"] = np.zeros((10, 6400))
        write_trained_settings(f, {key: config["pca"][key] for key in TRAINED_SETTINGS})
    assert config["pca"]["max_height"] == 120
    config["pca"]["max_height"] = 100
    write_yaml(config, str(root / "config.yaml"))
    return root


def trained_settings_stop(root) -> str:
    return (f"The PCA components in {root / '_pca' / 'pca.h5'} were trained with max_height 120, but the pca "
            f"section of {root / 'config.yaml'} sets 100. Scores and changepoints must use the trained value: set "
            f"max_height back to 120 in {root / 'config.yaml'}, or train PCA again with these settings.")


def test_the_notebook_stops_on_changed_trained_settings_before_asking(trained, monkeypatch, capsys):
    from moseq2_pca.gui import apply_pca_command

    components = (trained / "_pca" / "pca.h5").read_bytes()
    monkeypatch.setattr("builtins.input", asked)
    capsys.readouterr()

    with pytest.raises(TrainedSettingsError) as raised:
        apply_pca_command(str(trained))

    assert str(raised.value) == trained_settings_stop(trained)
    assert capsys.readouterr().out == ""
    assert (trained / "_pca" / "pca_scores.h5").read_bytes() == EARLIER
    assert (trained / "_pca" / "pca.h5").read_bytes() == components


def test_the_command_line_stops_on_changed_trained_settings_before_asking(trained):
    from moseq2_pca.cli import cli

    result = CliRunner().invoke(cli, ["apply-pca", "--config-file", str(trained / "config.yaml")], input="y\n")

    assert result.exit_code == 1, result.output
    assert result.output == f"Error: {trained_settings_stop(trained)}\n"
    assert (trained / "_pca" / "pca_scores.h5").read_bytes() == EARLIER


def test_components_that_cannot_be_read_get_the_input_check_message(trained):
    """The trained-settings check reads the components only once they can be opened; a file that cannot be
    is named by the input check, with what to do about it."""
    from moseq2_pca.cli import cli

    (trained / "_pca" / "pca.h5").write_bytes(b"not an HDF5 file")
    (trained / "_pca" / "pca_scores.h5").unlink()

    result = CliRunner().invoke(cli, ["apply-pca", "--config-file", str(trained / "config.yaml")])

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert result.output.startswith(f"Error: apply-pca needs {trained / '_pca' / 'pca.h5'}, which could not be read")
    assert not (trained / "_pca" / "pca_scores.h5").exists()
