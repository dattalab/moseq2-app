"""Each PCA step is one function in ``moseq2_pca.steps``, called by both front ends.

``train-pca``, ``apply-pca`` and ``compute-changepoints`` on the command line and
``train_pca_command``, ``apply_pca_command`` and ``compute_changepoints_command``
in the notebook resolve their settings into one options object and hand it to
the same step, which runs the work. A step also runs from a script, on its
options class's defaults, with no ``config.yaml`` at all.

A ``--config-file`` that does not exist stops the command, naming the file,
before anything is written (1.5.0 created an empty one and ran on defaults).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_pca.cli as pca_cli
import moseq2_pca.util as util
from moseq2_common.yaml import read_yaml, write_yaml

COMMANDS = ("train-pca", "apply-pca", "compute-changepoints")

#: What the Extract notebook's PCA cell sets before training (cell 13, a Kinect v2 project).
NOTEBOOK_PCA_SETTINGS = {"gaussfilter_space": (1.5, 1), "medfilter_space": [0],
                         "medfilter_time": [0], "nworkers": 1}


@pytest.fixture
def steps():
    import moseq2_pca.steps

    return moseq2_pca.steps


@pytest.fixture
def options():
    import moseq2_pca.options

    return moseq2_pca.options


def step_classes(options):
    return {"train_pca": options.TrainPCAOptions, "apply_pca": options.ApplyPCAOptions,
            "compute_changepoints": options.ChangepointsOptions}


@pytest.fixture
def called(monkeypatch, steps, options):
    """Each step replaced by a recorder of its arguments."""
    calls = {name: [] for name in step_classes(options)}

    def recorder(name):
        def record(input_dir, output_dir, output_file, opts, **kwargs):
            calls[name].append(opts)
            return getattr(steps, RESULTS[name])(**RESULT_FIELDS[name])
        return record

    for name in calls:
        monkeypatch.setattr(steps, name, recorder(name))
    return calls


RESULTS = {"train_pca": "TrainPCAResult", "apply_pca": "ApplyPCAResult",
           "compute_changepoints": "ChangepointsResult"}
RESULT_FIELDS = {"train_pca": {"components_file": None},
                 "apply_pca": {"scores_file": None},
                 "compute_changepoints": {"changepoints_file": None}}


@pytest.mark.parametrize("command", COMMANDS)
def test_pca_commands_stop_on_a_missing_config_file(tmp_path, command):
    missing = tmp_path / "missing.yaml"
    result = CliRunner().invoke(
        pca_cli.cli,
        [command, "-i", str(tmp_path), "-o", str(tmp_path / "_pca"), "--config-file", str(missing)],
    )
    assert result.exit_code != 0
    assert str(missing) in result.output
    assert not missing.exists()


def test_every_pca_command_calls_its_step(tmp_path, called, options):
    for command, name in zip(COMMANDS, step_classes(options)):
        result = CliRunner().invoke(
            pca_cli.cli, [command, "-i", str(tmp_path), "-o", str(tmp_path / "_pca")],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
    for name, cls in step_classes(options).items():
        assert len(called[name]) == 1, name
        assert type(called[name][0]) is cls, name


def test_the_notebook_functions_call_the_steps(tmp_path, called, options):
    from moseq2_pca.gui import apply_pca_command, compute_changepoints_command, train_pca_command

    from configs import write_bookkeeping

    config = tmp_path / "config.yaml"
    write_yaml({"pca": {"rank": 30}}, str(config))
    write_bookkeeping(tmp_path, ["aggregate_results/s0.h5"], scores="_pca/pca_scores.h5")

    train_pca_command(str(tmp_path), "pca")
    apply_pca_command(str(tmp_path), "pca_scores")
    compute_changepoints_command(str(tmp_path), "changepoints")

    for name, cls in step_classes(options).items():
        assert len(called[name]) == 1, name
        assert type(called[name][0]) is cls, name
    assert called["train_pca"][0].rank == 30


def test_both_front_ends_hand_the_wrapper_the_same_settings(tmp_path, monkeypatch, steps, options):
    from moseq2_extract.gui import generate_config_command
    from moseq2_pca.gui import train_pca_command

    received = []

    def record(input_dir, config_data, output_dir, output_file):
        received.append(dict(config_data))
        return config_data

    monkeypatch.setattr(steps, "train_pca_wrapper", record)

    configs = []
    for front_end in ("cli", "notebook"):
        config = tmp_path / front_end / "config.yaml"
        config.parent.mkdir()
        generate_config_command(str(config), camera_type="k2")
        data = read_yaml(str(config))
        data["pca"].update(NOTEBOOK_PCA_SETTINGS)
        write_yaml(data, str(config))
        configs.append(config)

    from configs import write_bookkeeping, write_session_labels

    # each project lists the same session
    for front_end in ("cli", "notebook"):
        write_bookkeeping(tmp_path / front_end, ["../s0"])
    write_session_labels(tmp_path, ["s0"])
    result = CliRunner().invoke(
        pca_cli.cli,
        ["train-pca", "-o", str(tmp_path / "cli" / "_pca"), "--config-file", str(configs[0])],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    train_pca_command(str(tmp_path / "notebook"), "pca")

    cli, notebook = received
    fields = list(options.TrainPCAOptions().as_dict())
    assert {k: cli[k] for k in fields} == {k: notebook[k] for k in fields}


@pytest.fixture
def extracted(tmp_path):
    """Two small extracted sessions (random 80 x 80 frames, 30 Hz timestamps)."""
    rng = np.random.default_rng(0)
    for i in range(2):
        folder = tmp_path / "aggregate_results" / f"session-{i}"
        folder.mkdir(parents=True)
        with h5py.File(folder / "results_00.h5", "w") as f:
            f["frames"] = rng.integers(0, 60, (200, 80, 80)).astype("uint8")
            f["timestamps"] = np.arange(200) * 33.3
        with h5py.File(folder / "results_00.h5", "a") as f:
            f["metadata/uuid"] = f"uuid-{i}"
    return tmp_path / "aggregate_results"


def test_pca_steps_run_from_a_script_without_a_config_file(tmp_path, extracted, steps, options, monkeypatch):
    monkeypatch.setattr(util, "get_env_cpu_and_mem", lambda: (2e9, 2))
    out = str(tmp_path / "_pca")
    cache = str(tmp_path / "cache")

    trained = steps.train_pca(str(extracted), out, "pca", options.TrainPCAOptions(
        cluster_type="local", nworkers=1, overwrite_pca_train=True, dask_cache_path=cache))
    scores = steps.apply_pca(str(extracted), out, "pca_scores", options.ApplyPCAOptions(
        cluster_type="nodask", overwrite_pca_apply=True, dask_cache_path=cache))
    changepoints = steps.compute_changepoints(str(extracted), out, "changepoints", options.ChangepointsOptions(
        nworkers=1, dask_cache_path=cache))

    assert trained.components_file.is_file()
    assert scores.scores_file.is_file()
    assert changepoints.changepoints_file.is_file()
    assert list(tmp_path.rglob("config.yaml")) == []
