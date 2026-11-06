"""A step's inputs or settings that are not usable print as a message on the command line, not a traceback.

Every check at the top of a step raises one kind of error
(``moseq2_common.records.StepInputError``): a missing or unreadable input
(``InputError``), a bookkeeping file that lists no sessions or cannot be read
(``BookkeepingError``), PCA components trained with other settings than
``config.yaml``'s (``TrainedSettingsError``), a flip classifier that is missing,
unreadable or trained for another crop size, and an extraction whose
frames and timestamps disagree in number. A command (``ProjectCommand``) prints
its message and exits 1, so the person reads what to do instead of the last line
of a Python traceback.
"""

from __future__ import annotations

import types

import click
import h5py
import joblib
import numpy as np
import pytest
from click.testing import CliRunner

from moseq2_common.options import ProjectCommand
from moseq2_common.yaml import write_yaml


def empty_project(root):
    """A project whose bookkeeping file lists no sessions yet."""
    from moseq2_extract.gui import generate_config_command

    if (root / "config.yaml").exists():
        return root
    root.mkdir()
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    write_yaml({"sessions": {}}, str(root / "moseq2-project.yaml"))
    return root


def no_sessions(tmp_path):
    from moseq2_common.bookkeeping import sessions_folder
    from moseq2_common.project import Project

    sessions_folder(Project.at(empty_project(tmp_path / "project")))


def untrained_components(tmp_path):
    from moseq2_pca.helpers.data import read_trained_settings

    with h5py.File(tmp_path / "pca.h5", "w") as f:
        f["components"] = np.zeros((10, 6400))
    read_trained_settings(str(tmp_path / "pca.h5"))


def missing_classifier(tmp_path):
    from moseq2_extract.helpers.wrappers import validate_flip_classifier

    validate_flip_classifier({"flip_classifier": str(tmp_path / "flip.p"), "crop_size": (80, 80)})


def unreadable_classifier(tmp_path):
    from moseq2_extract.helpers.wrappers import validate_flip_classifier

    (tmp_path / "flip.p").write_bytes(b"not a classifier")
    validate_flip_classifier({"flip_classifier": str(tmp_path / "flip.p"), "crop_size": (80, 80)})


def classifier_for_other_crop_size(tmp_path):
    from moseq2_extract.helpers.wrappers import validate_flip_classifier

    joblib.dump(types.SimpleNamespace(training_settings={"crop_size": [20, 20]}), tmp_path / "flip.p")
    validate_flip_classifier({"flip_classifier": str(tmp_path / "flip.p"), "crop_size": (80, 80)})


def frames_without_timestamps(tmp_path):
    from moseq2_pca.util import check_frame_counts

    with h5py.File(tmp_path / "results.h5", "w") as f:
        f["frames"] = np.zeros((20, 8, 8), dtype="uint8")
        f["timestamps"] = np.arange(19, dtype="float64")
    check_frame_counts([str(tmp_path / "results.h5")])


@pytest.mark.parametrize("check", [no_sessions, untrained_components, missing_classifier, unreadable_classifier,
                                   classifier_for_other_crop_size,
                                   frames_without_timestamps])
def test_an_unusable_input_prints_as_a_message(tmp_path, monkeypatch, check):
    from moseq2_common.records import StepInputError

    with pytest.raises(StepInputError) as raised:
        check(tmp_path)
    message = str(raised.value)

    @click.command(cls=ProjectCommand)
    def step():
        check(tmp_path)

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(step, [])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert result.output == f"Error: {message}\n"


def test_train_pca_in_a_project_with_no_sessions_says_what_to_do(tmp_path, monkeypatch):
    """The reviewer's case: ``train-pca`` before any session is aggregated printed a 26-line traceback."""
    from moseq2_pca.cli import cli

    root = empty_project(tmp_path / "project")
    monkeypatch.chdir(root)
    result = CliRunner().invoke(cli, ["train-pca"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert result.output.startswith("Error: ")
    assert "lists no sessions yet. Aggregate the extractions first" in result.output
