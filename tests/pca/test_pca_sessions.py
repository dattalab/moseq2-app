"""Which sessions the PCA steps read, and in what order.

In a project, ``train-pca``, ``apply-pca`` and ``compute-changepoints`` (and the
notebook's PCA functions) take their sessions from ``moseq2-project.yaml``
through ``moseq2_common.bookkeeping.session_index``, the list the model and
analysis steps use: the sessions ``config.yaml``'s ``exclude`` lists are left
out, and the rest are listed in the order they were recorded (``StartTime``),
then by name. Outside a project, the steps scan their input folder and list
what they find in the same order.

The order matters beyond tidiness: ``compute-changepoints`` draws one random
projection per session from numpy's global generator, in listing order, so a
listing by folder name would tie the changepoints to what the folders are
called.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_pca.util as util
from moseq2_common.yaml import read_yaml, write_yaml

#: (session key, folder in aggregate_results/, uuid, start time). The folders sort opposite to the start times.
SESSIONS = (
    ("cohort/zebra/depth.avi", "a_recorded_last", "uuid-late", "2020-01-03T10:00:00.000000-05:00"),
    ("cohort/yak/depth.avi", "b_recorded_second", "uuid-middle", "2020-01-02T10:00:00.000000-05:00"),
    ("cohort/xerus/depth.avi", "c_recorded_first", "uuid-early", "2020-01-01T10:00:00.000000-05:00"),
)
IN_RECORDING_ORDER = ["uuid-early", "uuid-middle", "uuid-late"]


def write_sessions(root):
    """The three sessions' extraction files under ``root``: small random frames, 30 Hz timestamps."""
    rng = np.random.default_rng(0)
    for key, folder, uuid, start in SESSIONS:
        path = root / folder
        path.mkdir(parents=True)
        with h5py.File(path / "results.h5", "w") as f:
            f["frames"] = rng.integers(0, 60, (200, 80, 80)).astype("uint8")
            f["timestamps"] = np.arange(200) * 33.3
            f["metadata/uuid"] = uuid
            f["metadata/acquisition/SessionName"] = key.split("/")[1]
            f["metadata/acquisition/StartTime"] = start


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project of the three sessions, listed in its bookkeeping file in folder-name order."""
    from moseq2_extract.gui import generate_config_command

    monkeypatch.setattr(util, "get_env_cpu_and_mem", lambda: (2e9, 2))
    root = tmp_path / "project"
    root.mkdir()
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    config = read_yaml(str(root / "config.yaml"))
    config["pca"].update({"cluster_type": "local", "nworkers": 1, "overwrite_pca_train": True,
                          "overwrite_pca_apply": True, "dask_cache_path": str(tmp_path / "cache"),
                          "gaussfilter_space": (1.5, 1), "medfilter_space": [0], "medfilter_time": [0]})
    write_yaml(config, str(root / "config.yaml"))
    write_sessions(root / "aggregate_results")
    write_yaml({"sessions": {key: {"recording": key, "folder": f"aggregate_results/{folder}", "uuid": uuid}
                             for key, folder, uuid, _ in SESSIONS}},
               str(root / "moseq2-project.yaml"))
    return root


def exclude(project, keys):
    config = read_yaml(str(project / "config.yaml"))
    config["exclude"] = list(keys)
    write_yaml(config, str(project / "config.yaml"))


def trained_on(pca_dir):
    """The uuids of the sessions training read, in the order it read them."""
    uuids = {f"{folder}/results.h5": uuid for _, folder, uuid, _ in SESSIONS}
    inputs = read_yaml(str(pca_dir / "pca.yaml"))["inputs"]
    return [next(uuid for tail, uuid in uuids.items() if path.endswith(tail)) for path in inputs]


def keys_in(h5_file, group):
    with h5py.File(h5_file, "r") as f:
        return sorted(f[group].keys())


def run_notebook(project):
    """The notebook's three PCA functions; what they printed."""
    from moseq2_pca.gui import apply_pca_command, compute_changepoints_command, train_pca_command

    output = io.StringIO()
    with redirect_stdout(output):
        train_pca_command(str(project), "pca")
        apply_pca_command(str(project), "pca_scores")
        compute_changepoints_command(str(project), "changepoints")
    return output.getvalue()


def run_command_line(project):
    """The three PCA commands, in the project; what they printed."""
    from moseq2_pca.cli import cli

    pca = str(project / "_pca")
    config = str(project / "config.yaml")
    output = ""
    for command in ("train-pca", "apply-pca", "compute-changepoints"):
        result = CliRunner().invoke(cli, [command, "-o", pca, "--config-file", config], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        output += result.output
    return output


@pytest.mark.parametrize("run", [run_notebook, run_command_line], ids=["notebook", "cli"])
def test_an_excluded_session_is_left_out_of_pca(project, run):
    exclude(project, ["cohort/yak/depth.avi"])

    output = run(project)

    kept = ["uuid-early", "uuid-late"]
    assert trained_on(project / "_pca") == kept
    assert keys_in(project / "_pca" / "pca_scores.h5", "scores") == kept
    assert keys_in(project / "_pca" / "changepoints.h5", "cps") == kept
    # named once by each of the three steps
    assert output.count("Leaving out the 1 sessions under exclude in config.yaml: cohort/yak/depth.avi.") == 3


@pytest.mark.parametrize("run", [run_notebook, run_command_line], ids=["notebook", "cli"])
def test_a_project_s_sessions_are_read_in_recording_order(project, run):
    run(project)

    assert trained_on(project / "_pca") == IN_RECORDING_ORDER


def test_outside_a_project_the_scanned_sessions_are_read_in_recording_order(tmp_path, monkeypatch):
    from moseq2_pca import steps
    from moseq2_pca.options import TrainPCAOptions

    monkeypatch.setattr(util, "get_env_cpu_and_mem", lambda: (2e9, 2))
    write_sessions(tmp_path / "extracted")

    steps.train_pca(str(tmp_path / "extracted"), str(tmp_path / "_pca"), "pca", TrainPCAOptions(
        cluster_type="local", nworkers=1, overwrite_pca_train=True, dask_cache_path=str(tmp_path / "cache")))

    assert trained_on(tmp_path / "_pca") == IN_RECORDING_ORDER


def test_a_project_with_every_session_excluded_stops_pca(project):
    from moseq2_common.bookkeeping import BookkeepingError
    from moseq2_pca.gui import train_pca_command

    exclude(project, [key for key, *_ in SESSIONS])

    with pytest.raises(BookkeepingError, match="Remove at least one session from exclude"):
        train_pca_command(str(project), "pca")
    assert not (project / "_pca" / "pca.h5").exists()


def test_a_typed_input_dir_in_a_project_is_named_as_unread(project, monkeypatch):
    import moseq2_pca.steps as steps
    from moseq2_pca.cli import cli

    read = []
    monkeypatch.setattr(steps, "train_pca_wrapper", lambda sessions, config, *_: read.append(sessions) or config)
    elsewhere = project.parent / "elsewhere"

    result = CliRunner().invoke(cli, ["train-pca", "-i", str(elsewhere), "--config-file",
                                      str(project / "config.yaml")], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert (f"train-pca runs in the project {project}, so it reads the sessions {project / 'moseq2-project.yaml'} "
            f"lists, not --input-dir {elsewhere}.") in result.output
    assert read[0] == [str(project / "aggregate_results" / folder / "results.h5")
                                        for folder in ("c_recorded_first", "b_recorded_second", "a_recorded_last")]
