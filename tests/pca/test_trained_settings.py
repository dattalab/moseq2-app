"""PCA components carry the settings they were trained with.

The frames ``train-pca`` learns its components from are filtered (the
Gaussian, median and tail filters), cut to a height range and, for
missing-data PCA, masked. ``apply-pca`` and ``compute-changepoints`` must
prepare each session's frames the same way, or the scores describe frames the
components never saw. So the components file holds those settings, as a YAML
attribute ``/settings`` of the ``.h5``, and the two steps take them from there
(``moseq2_pca.steps.TRAINED_SETTINGS``); there is no ``pca.yaml`` beside it.

When the project's ``config.yaml`` sets one of those settings to another value
than the components were trained with, the two steps stop before any work,
naming the setting, both values and both files, and saying to set it back or
train again.
"""

from __future__ import annotations

import io

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_pca.util as util
from moseq2_common.yaml import read_yaml, write_yaml

#: (session key, folder in aggregate_results/, uuid, start time)
SESSIONS = (
    ("cohort/a/depth.avi", "cohort_a", "uuid-a", "2020-01-01T10:00:00.000000-05:00"),
    ("cohort/b/depth.avi", "cohort_b", "uuid-b", "2020-01-02T10:00:00.000000-05:00"),
)

#: What the project trains with, away from the defaults where a test looks for the value.
TRAINED = {"gaussfilter_space": [2.25, 1.5], "min_height": 5, "max_height": 100, "tailfilter_size": [13, 13]}


def set_pca(project, **settings):
    config = read_yaml(str(project / "config.yaml"))
    for key, value in settings.items():
        if value is None:
            config["pca"].pop(key, None)
        else:
            config["pca"][key] = value
    write_yaml(config, str(project / "config.yaml"))


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project of two small sessions, trained with ``TRAINED``."""
    from moseq2_extract.gui import generate_config_command
    from moseq2_pca.gui import train_pca_command

    monkeypatch.setattr(util, "get_env_cpu_and_mem", lambda: (2e9, 2))
    root = tmp_path / "project"
    root.mkdir()
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    rng = np.random.default_rng(0)
    for key, folder, uuid, start in SESSIONS:
        path = root / "aggregate_results" / folder
        path.mkdir(parents=True)
        with h5py.File(path / "results.h5", "w") as f:
            f["frames"] = rng.integers(0, 60, (100, 80, 80)).astype("uint8")
            f["timestamps"] = np.arange(100) * 33.3
            f["metadata/uuid"] = uuid
            f["metadata/acquisition/SessionName"] = key.split("/")[1]
            f["metadata/acquisition/StartTime"] = start
    write_yaml({"sessions": {key: {"recording": key, "folder": f"aggregate_results/{folder}", "uuid": uuid}
                             for key, folder, uuid, _ in SESSIONS}},
               str(root / "moseq2-project.yaml"))
    set_pca(root, cluster_type="local", nworkers=1, overwrite_pca_train=True, overwrite_pca_apply=True,
            dask_cache_path=str(tmp_path / "cache"), **TRAINED)
    train_pca_command(str(root), "pca")
    return root


def stored_settings(components):
    with h5py.File(components, "r") as f:
        return read_yaml(io.StringIO(f.attrs["settings"]))


def test_components_carry_their_training_settings(project):
    from moseq2_pca.steps import TRAINED_SETTINGS

    components = project / "_pca" / "pca.h5"
    stored = stored_settings(components)

    assert set(stored) == set(TRAINED_SETTINGS)
    for key, value in TRAINED.items():
        assert stored[key] == value, key
    assert stored["missing_data"] is False and stored["use_fft"] is False
    assert not (project / "_pca" / "pca.yaml").exists()


def test_apply_pca_uses_the_trained_settings(project, monkeypatch):
    """config.yaml no longer sets the filter or the height range: scoring uses the trained ones."""
    import moseq2_pca.helpers.wrappers as wrappers
    from moseq2_pca.gui import apply_pca_command

    used = {}
    monkeypatch.setattr(wrappers, "apply_pca_dask", lambda **kwargs: used.update(kwargs))
    set_pca(project, gaussfilter_space=None, min_height=None, max_height=None, tailfilter_size=None)

    apply_pca_command(str(project), "pca_scores")

    assert tuple(used["clean_params"]["gaussfilter_space"]) == (2.25, 1.5)
    assert (used["mask_params"]["min_height"], used["mask_params"]["max_height"]) == (5, 100)


def w10(project, setting, trained, configured):
    return (f"The PCA components in {project / '_pca' / 'pca.h5'} were trained with {setting} {trained}, but the "
            f"pca section of {project / 'config.yaml'} sets {configured}. Scores and changepoints must use the "
            f"trained value: set {setting} back to {trained} in {project / 'config.yaml'}, or train PCA again "
            "with these settings.")


@pytest.mark.parametrize("step", ["apply_pca_command", "compute_changepoints_command"])
def test_apply_pca_stops_when_config_filters_differ_from_the_trained_ones(project, monkeypatch, step):
    import moseq2_pca.gui as gui
    import moseq2_pca.helpers.wrappers as wrappers

    def reached(*args, **kwargs):
        raise AssertionError("the step started its work")

    monkeypatch.setattr(wrappers, "load_and_check_data", reached)
    set_pca(project, gaussfilter_space=[1.5, 1], max_height=120)

    with pytest.raises(ValueError) as e:
        getattr(gui, step)(str(project))

    message = str(e.value)
    assert w10(project, "gaussfilter_space", "[2.25, 1.5]", "[1.5, 1]") in message
    assert w10(project, "max_height", "100", "120") in message
    assert "min_height" not in message


def test_the_command_line_stops_with_the_message_and_writes_nothing(project):
    from moseq2_pca.cli import cli

    set_pca(project, tailfilter_size=[9, 9])

    result = CliRunner().invoke(cli, ["apply-pca", "-o", str(project / "_pca"),
                                      "--config-file", str(project / "config.yaml")])

    assert result.exit_code != 0
    assert w10(project, "tailfilter_size", "[13, 13]", "[9, 9]") in " ".join(result.output.split())
    assert not (project / "_pca" / "pca_scores.h5").exists()


def test_components_without_settings_stop_apply_pca(project):
    """A components file from before 1.6.0, which carries no settings, says to train again."""
    from moseq2_pca.gui import apply_pca_command

    with h5py.File(project / "_pca" / "pca.h5", "a") as f:
        del f.attrs["settings"]

    with pytest.raises(ValueError, match="Train PCA again"):
        apply_pca_command(str(project), "pca_scores")
