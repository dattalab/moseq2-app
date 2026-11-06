"""``exclude`` is an ordinary setting of every step after extraction.

Its default is the running project's ``config.yaml`` ``exclude``. A list typed
for one run (``--exclude KEY``, repeatable, on the command line; ``exclude=`` on
the step function) replaces that list for that run, and ``--no-exclude``
(``exclude=[]``) leaves no session out; typing both is an error saying to pick
one. ``exclude=None`` means ``config.yaml``'s list. Each run prints one line
naming the list it used, when the list is not empty or was typed, and the run's
record names the sessions used.

The list applies to a typed ``--input-dir`` too: the extractions found there are
matched to session keys, and when the folder belongs to another project, the
running project's ``config.yaml`` and ``exclude`` hold, not the other project's.
A typed key that names no session stops the run, since a typo would otherwise
quietly keep the session in.

The progress display flags an output computed with a session the current
``exclude`` leaves out.
"""

from __future__ import annotations

import contextlib
import io
import os
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from moseq2_common.yaml import read_yaml, write_yaml

#: (session key, folder in aggregate_results/, uuid, start time).
SESSIONS = (
    ("cohort/ant/depth.avi", "cohort_ant", "uuid-a", "2020-01-01T10:00:00.000000-05:00"),
    ("cohort/bee/depth.avi", "cohort_bee", "uuid-b", "2020-01-02T10:00:00.000000-05:00"),
    ("cohort/cat/depth.avi", "cohort_cat", "uuid-c", "2020-01-03T10:00:00.000000-05:00"),
)
ANT, BEE, CAT = (key for key, *_ in SESSIONS)


def write_session(path, uuid, key, start, frames=20):
    """A small extraction file: frames, 30 Hz timestamps, its uuid and start time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["frames"] = np.zeros((frames, 8, 8), dtype="uint8")
        f["timestamps"] = np.arange(frames) * 1000 / 30
        f["metadata/fps"] = 30.0
        f["metadata/uuid"] = uuid
        f["metadata/acquisition/SessionName"] = key.split("/")[1]
        f["metadata/acquisition/StartTime"] = start


def make_project(root, exclude=None):
    """A project of the three sessions, extracted and listed, with ``exclude`` in its config.yaml."""
    root.mkdir(parents=True)
    config = {"camera_type": "auto", "extract": {}, "pca": {}, "model": {}, "crowd_movies": {}}
    if exclude is not None:
        config["exclude"] = list(exclude)
    write_yaml(config, str(root / "config.yaml"))
    for key, folder, uuid, start in SESSIONS:
        write_session(root / "aggregate_results" / folder / "results.h5", uuid, key, start)
    write_yaml({"sessions": {key: {"recording": key, "folder": f"aggregate_results/{folder}", "uuid": uuid}
                             for key, folder, uuid, _ in SESSIONS},
                "pca": {"scores": "_pca/pca_scores.h5"},
                "models": {"fitted": ["models/model.p"], "current": "models/model.p", "analysis": {}},
                "applied_models": []}, str(root / "moseq2-project.yaml"))
    for name in ("_pca/pca_scores.h5", "models/model.p"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"\0")
    return root


@pytest.fixture
def project(tmp_path):
    """A project whose config.yaml leaves out ``ANT``."""
    return make_project(tmp_path / "project", exclude=[ANT])


@contextmanager
def working_in(folder):
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def invoke(cli, *args, cwd):
    with working_in(cwd):
        return CliRunner().invoke(cli, [str(arg) for arg in args])


def printed(call, *args, **kwargs):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = call(*args, **kwargs)
    return result, stream.getvalue()


def keys(index):
    return [f["key"] for f in index["files"]]


# ------------------------------------------------------------------ the index


def test_none_means_config_yaml_s_list(project):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project

    index, output = printed(session_index, Project.at(project))

    assert keys(index) == [BEE, CAT]
    assert output.count("Leaving out") == 1 and ANT in output


def test_a_list_replaces_config_yaml_s_list(project):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project

    index, output = printed(session_index, Project.at(project), exclude=[BEE])

    assert keys(index) == [ANT, CAT]
    assert list(index["excluded"]) == [BEE]
    assert output.count("Leaving out") == 1 and BEE in output and ANT not in output


def test_an_empty_list_leaves_no_session_out_and_says_so(project):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project

    index, output = printed(session_index, Project.at(project), exclude=[])

    assert keys(index) == [ANT, BEE, CAT]
    assert "excluded" not in index
    assert output.count("Leaving out no session") == 1


def test_nothing_is_printed_when_the_list_is_empty_and_not_given(tmp_path):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project

    root = make_project(tmp_path / "plain")
    _, output = printed(session_index, Project.at(root))

    assert "Leaving out" not in output


def test_a_given_key_that_is_no_session_stops(project):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project
    from moseq2_common.records import StepInputError

    with pytest.raises(StepInputError) as error:
        session_index(Project.at(project), exclude=["cohort/typo/depth.avi"])

    message = str(error.value)
    assert "cohort/typo/depth.avi" in message and BEE in message


def test_groups_given_replace_config_yaml_s_groups(project):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project

    config = read_yaml(str(project / "config.yaml"))
    config["groups"] = {BEE: "saline", CAT: "saline"}
    write_yaml(config, str(project / "config.yaml"))

    index = session_index(Project.at(project), groups={BEE: "drug"})

    assert {f["key"]: f["group"] for f in index["files"]} == {BEE: "drug", CAT: "default"}


# ------------------------------------------------------------------ the command line


@pytest.fixture
def viz(monkeypatch):
    import moseq2_viz.cli as cli

    calls = {}

    def plot_stats(index, *args, **kwargs):
        calls["index"] = index

    monkeypatch.setattr(cli.steps, "plot_stats", plot_stats)
    return cli.cli, calls


def test_typed_exclude_replaces_config_yaml_s_list(project, viz):
    cli, calls = viz
    result = invoke(cli, "plot-stats", "--exclude", BEE, "--exclude", CAT, cwd=project)

    assert result.exit_code == 0, result.output
    assert keys(calls["index"]) == [ANT]
    assert result.output.count("Leaving out") == 1


def test_no_exclude_leaves_no_session_out(project, viz):
    cli, calls = viz
    result = invoke(cli, "plot-stats", "--no-exclude", cwd=project)

    assert result.exit_code == 0, result.output
    assert keys(calls["index"]) == [ANT, BEE, CAT]
    assert result.output.count("Leaving out no session") == 1


def test_both_typed_is_an_error_saying_to_pick_one(project, viz):
    cli, calls = viz
    result = invoke(cli, "plot-stats", "--no-exclude", "--exclude", BEE, cwd=project)

    assert result.exit_code == 2, result.output
    message = " ".join(result.output.split())
    assert "--no-exclude" in message and "--exclude" in message and "one" in message
    assert "index" not in calls


@pytest.mark.parametrize("command, step", [("learn-model", "learn_model"), ("apply-model", "apply_model"),
                                           ("kappa-scan", "kappa_scan")])
def test_model_commands_take_exclude(command, step, project, monkeypatch):
    import moseq2_model.cli as cli

    calls = {}
    monkeypatch.setattr(cli.steps, step, lambda *args, **kwargs: calls.setdefault("kwargs", kwargs))
    args = [command, *(["models/model.p"] if command == "apply-model" else []), "--exclude", CAT]
    result = invoke(cli.cli, *args, cwd=project)

    assert result.exit_code == 0, result.output
    assert keys(calls["kwargs"]["index"]) == [ANT, BEE]


def test_kappa_scan_jobs_carry_a_typed_exclude(tmp_path):
    from moseq2_model.util import create_command_strings

    settings = {"cluster_type": "local", "ncpus": 0}
    typed = create_command_strings("scores.h5", str(tmp_path / "a"), settings, [1000], exclude=[BEE, CAT])
    none = create_command_strings("scores.h5", str(tmp_path / "b"), settings, [1000], exclude=[])
    config = create_command_strings("scores.h5", str(tmp_path / "c"), settings, [1000])

    assert f"--exclude {BEE} --exclude {CAT}" in typed
    assert "--no-exclude" in none
    assert "exclude" not in config


# ------------------------------------------------------------------ PCA, and a typed --input-dir


@pytest.fixture
def pca_wrapper(monkeypatch):
    """Stand in for train-pca's work, recording the sessions it was given."""
    import moseq2_pca.steps as steps

    calls = {}

    def train_pca_wrapper(sessions, *args, **kwargs):
        calls["sessions"] = sessions
        return {}

    monkeypatch.setattr(steps, "train_pca_wrapper", train_pca_wrapper)
    return calls


def _sessions_read(calls):
    from moseq2_pca.util import find_extractions

    return [info["uuid"] for info in find_extractions(calls["sessions"])[1]]


def test_train_pca_step_takes_exclude(project, pca_wrapper):
    from moseq2_common.project import Project
    from moseq2_pca.options import TrainPCAOptions
    from moseq2_pca.steps import train_pca

    train_pca(None, str(project / "_pca"), "pca", TrainPCAOptions(), project=Project.at(project), exclude=[CAT])

    assert _sessions_read(pca_wrapper) == ["uuid-a", "uuid-b"]


def test_a_typed_input_dir_of_another_project_uses_the_running_project_s_exclude(tmp_path, pca_wrapper):
    from moseq2_pca.cli import cli

    other = make_project(tmp_path / "other", exclude=[])
    running = make_project(tmp_path / "running", exclude=[BEE])

    result = invoke(cli, "train-pca", "--input-dir", other / "aggregate_results", cwd=running)

    assert result.exit_code == 0, result.output
    assert _sessions_read(pca_wrapper) == ["uuid-a", "uuid-c"]


def test_a_typed_exclude_applies_to_a_typed_input_dir(tmp_path, pca_wrapper):
    from moseq2_pca.cli import cli

    other = make_project(tmp_path / "other", exclude=[CAT])
    running = make_project(tmp_path / "running", exclude=[BEE])

    result = invoke(cli, "train-pca", "-i", other / "aggregate_results", "--exclude", ANT, cwd=running)

    assert result.exit_code == 0, result.output
    assert _sessions_read(pca_wrapper) == ["uuid-b", "uuid-c"]


# ------------------------------------------------------------------ the record


def test_the_record_names_the_sessions_used(project, monkeypatch):
    import moseq2_model.steps as steps
    from moseq2_common.bookkeeping import project_index
    from moseq2_common.project import Project
    from moseq2_common.records import read_record
    from moseq2_model.options import LearnModelOptions

    def learn_model_wrapper(input_file, dest_file, settings):
        Path(dest_file).write_bytes(b"model")

    monkeypatch.setattr(steps, "learn_model_wrapper", learn_model_wrapper)
    proj = Project.at(project)
    write_session(project / "_pca" / "pca_scores.h5", "scores", "x/y", "2020")
    steps.learn_model(str(project / "_pca" / "pca_scores.h5"), str(project / "models" / "fit.p"),
                      LearnModelOptions(), index=project_index(proj), project=proj, exclude=[CAT])

    notes = read_record(project / "models" / "fit.p")["notes"]
    assert notes["sessions"] == [ANT, BEE]
    assert notes["exclude"] == [CAT]


# ------------------------------------------------------------------ the progress display


def test_progress_flags_an_output_computed_with_a_session_exclude_now_leaves_out(project):
    from moseq2_app.project import stale_outputs
    from moseq2_common.project import Project
    from moseq2_common.records import start_record, write_h5_record

    proj = Project.at(project)
    sessions = [project / "aggregate_results" / folder / "results.h5" for _, folder, _, _ in SESSIONS]
    record = start_record("apply-pca", {}, sessions, project=proj)
    with h5py.File(project / "_pca" / "pca_scores.h5", "w") as f:
        write_h5_record(f, record.finish({}))

    lines = stale_outputs(proj)

    assert any(ANT in line and "exclude" in line and "apply-pca" in line for line in lines), lines
    assert not any(BEE in line for line in lines), lines
