"""PCA's project layer gives each step its project's sessions, files and record.

``moseq2_pca.project`` resolves what a PCA step reads in a project: the
sessions ``moseq2-project.yaml`` lists, less ``exclude``, in the order they were
recorded (or the extractions under a typed folder, named by the project), and
the components and scores the project recorded. It starts the run's record with
the project, so the record describes each input relative to the project folder,
names ``config.yaml`` and notes the sessions used and left out; it hands both to
the file layer and records what the step wrote in ``moseq2-project.yaml``.

Outside any project, the command line names the extractions under its folder
the way the project that owns them does, or by their path when none does, and
filters them by ``--exclude`` itself; nothing is written to an owning project.

An output that already exists with overwrite off is asked about first, before
anything else is printed, and the answer goes back to the step without changing
its options, so the record holds the setting as it was resolved.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_pca.util as util
from moseq2_common.options import read_settings
from moseq2_common.project import Project
from moseq2_common.records import read_record
from moseq2_common.yaml import read_yaml, write_yaml
from moseq2_pca import project as project_steps
from moseq2_pca.options import PCA_STAGE, ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions

#: (session key, folder in aggregate_results/, uuid, start time). The folders sort opposite to the start times.
SESSIONS = (
    ("cohort/zebra/depth.avi", "a_recorded_last", "uuid-late", "2020-01-03T10:00:00.000000-05:00"),
    ("cohort/yak/depth.avi", "b_recorded_second", "uuid-middle", "2020-01-02T10:00:00.000000-05:00"),
    ("cohort/xerus/depth.avi", "c_recorded_first", "uuid-early", "2020-01-01T10:00:00.000000-05:00"),
)
ZEBRA, YAK, XERUS = (key for key, *_ in SESSIONS)
#: The sessions read, in recording order, with ``YAK`` under exclude in config.yaml.
READ = [XERUS, ZEBRA]
READ_FOLDERS = ["c_recorded_first", "a_recorded_last"]

EXCLUDE_LINE = f"Leaving out the 1 sessions under exclude in config.yaml: {YAK}."


def call(step, project, input_dir, output_dir, output_file, opts, **kwargs):
    """``step`` of the project layer in ``project``."""
    return getattr(project_steps, step)(project, input_dir, output_dir, output_file, opts, **kwargs)


def write_extraction(path: Path, uuid: str, name: str, start: str) -> None:
    rng = np.random.default_rng(sum(map(ord, uuid)))
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        f["frames"] = rng.integers(0, 60, (200, 80, 80)).astype("uint8")
        f["timestamps"] = np.arange(200) * 1000 / 30
        f["metadata/fps"] = 30.0
        f["metadata/uuid"] = uuid
        f["metadata/acquisition/SessionName"] = name
        f["metadata/acquisition/StartTime"] = start


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A project of the three sessions, listed in folder-name order, with ``YAK`` under exclude in config.yaml."""
    from moseq2_extract.gui import generate_config_command

    monkeypatch.setattr(util, "get_env_cpu_and_mem", lambda: (2e9, 2))
    root = tmp_path / "project"
    root.mkdir()
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    config = read_yaml(str(root / "config.yaml"))
    config["pca"].update({"cluster_type": "local", "nworkers": 1, "dask_cache_path": str(tmp_path / "cache")})
    config["exclude"] = [YAK]
    write_yaml(config, str(root / "config.yaml"))
    for key, folder, uuid, start in SESSIONS:
        write_extraction(root / "aggregate_results" / folder / "results.h5", uuid, key.split("/")[1], start)
    write_yaml({"sessions": {key: {"recording": key, "folder": f"aggregate_results/{folder}", "uuid": uuid}
                             for key, folder, uuid, _ in SESSIONS},
                "pca": {}, "models": {"fitted": [], "current": None, "analysis": {}}, "applied_models": []},
               str(root / "moseq2-project.yaml"))
    return root


def resolved(cls, root):
    """``cls`` resolved from the project's config.yaml, as the notebook resolves it."""
    return cls.resolve(config=read_settings(root / "config.yaml", cls, stage_classes=PCA_STAGE))


def train(root, output_file="pca", input_dir=None):
    return call("train_pca", Project.at(root), input_dir, str(root / "_pca"), output_file,
                resolved(TrainPCAOptions, root))


def scores(root):
    return call("apply_pca", Project.at(root), None, str(root / "_pca"), "pca_scores",
                ApplyPCAOptions(cluster_type="nodask"))


def recorded(root):
    return read_yaml(str(root / "moseq2-project.yaml"))["pca"]


def extractions(record):
    return [item for item in record["inputs"] if "uuid" in item]


# ------------------------------------------------------------------ in a project


def test_train_pca_in_a_project_reads_the_listed_sessions_and_records_the_components(root, capsys):
    result = train(root)

    assert result.components_file == root / "_pca" / "pca.h5"
    assert recorded(root) == {"components": "_pca/pca.h5"}
    record = read_record(root / "_pca" / "pca.h5")
    assert record["project"] == str(root.resolve())
    assert record["inputs"][0]["relative_path"] == "config.yaml"
    assert [item["relative_path"] for item in record["inputs"][1:]] == [
        f"aggregate_results/{folder}/results.h5" for folder in READ_FOLDERS]
    assert [item["uuid"] for item in extractions(record)] == ["uuid-early", "uuid-late"]
    assert record["notes"]["sessions"] == READ
    assert record["notes"]["exclude"] == [YAK]
    assert capsys.readouterr().out.count(EXCLUDE_LINE) == 1


def test_apply_pca_in_a_project_uses_the_recorded_components(root):
    train(root, output_file="trained")

    result = scores(root)

    assert result.scores_file == root / "_pca" / "pca_scores.h5"
    assert recorded(root) == {"components": "_pca/trained.h5", "scores": "_pca/pca_scores.h5"}
    record = read_record(root / "_pca" / "pca_scores.h5")
    assert [item["relative_path"] for item in record["inputs"][:2]] == ["config.yaml", "_pca/trained.h5"]
    assert [item["relative_path"] for item in extractions(record)] == [
        f"aggregate_results/{folder}/results.h5" for folder in READ_FOLDERS]
    assert record["notes"]["sessions"] == READ


def test_compute_changepoints_in_a_project_uses_the_recorded_components_and_scores(root):
    train(root, output_file="trained")
    scores(root)

    result = call("compute_changepoints", Project.at(root), None, str(root / "_pca"), "changepoints",
                  ChangepointsOptions(nworkers=1, dask_cache_path=str(root.parent / "cache")))

    assert result.changepoints_file == root / "_pca" / "changepoints.h5"
    assert recorded(root)["changepoints"] == "_pca/changepoints.h5"
    record = read_record(root / "_pca" / "changepoints.h5")
    assert [item["relative_path"] for item in record["inputs"][:3]] == [
        "config.yaml", "_pca/trained.h5", "_pca/pca_scores.h5"]
    assert record["notes"]["sessions"] == READ
    assert record["notes"]["exclude"] == [YAK]


def test_a_typed_input_dir_in_a_project_is_named_and_excluded_by_the_project(root, capsys):
    """A copy of the extractions in a folder no project owns: each is the project's session by its uuid, and
    the project's exclude leaves ``YAK`` out."""
    elsewhere = root.parent / "elsewhere"
    shutil.copytree(root / "aggregate_results", elsewhere)

    train(root, input_dir=str(elsewhere))

    record = read_record(root / "_pca" / "pca.h5")
    assert [item["path"] for item in extractions(record)] == [
        str(elsewhere / folder / "results.h5") for folder in READ_FOLDERS]
    assert record["notes"]["sessions"] == READ
    assert record["notes"]["exclude"] == [YAK]
    assert recorded(root) == {"components": "_pca/pca.h5"}
    assert capsys.readouterr().out.count(EXCLUDE_LINE) == 1


# ------------------------------------------------------------------ outside any project, on the command line


@pytest.fixture
def wrapper(monkeypatch):
    """Stand in for train-pca's work: write an empty components file, recording the extractions read."""
    import moseq2_pca.steps as steps

    read = []

    def train_pca_wrapper(sessions, config, output_dir, output_file):
        from moseq2_pca.helpers.data import write_trained_settings
        from moseq2_pca.util import find_extractions

        read.append(find_extractions(sessions)[0])
        path = Path(output_dir) / f"{output_file}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            write_trained_settings(f, config)
        return {**config, "pca_file": str(path)}

    monkeypatch.setattr(steps, "train_pca_wrapper", train_pca_wrapper)
    return read


def train_pca_command_line(*args, cwd, input=None):
    """``moseq2-pca train-pca`` with ``args``, run from ``cwd``."""
    from moseq2_pca.cli import cli

    before = os.getcwd()
    os.chdir(cwd)
    try:
        return CliRunner().invoke(cli, ["train-pca", *[str(arg) for arg in args]], input=input,
                                  catch_exceptions=False)
    finally:
        os.chdir(before)


def test_a_folder_no_project_owns_is_named_by_path_and_filtered_on_the_command_line(tmp_path, wrapper):
    plain, work = tmp_path / "plain", tmp_path / "work"
    for key, folder, uuid, start in SESSIONS:
        write_extraction(plain / folder / "results.h5", uuid, key.split("/")[1], start)
    work.mkdir()
    left_out = "b_recorded_second/results.h5"

    result = train_pca_command_line("-i", plain, "--exclude", left_out, cwd=work)

    assert result.exit_code == 0, result.output
    assert f"Leaving out the 1 session given for this run: {left_out}." in result.output.splitlines()
    assert wrapper == [[str(plain / folder / "results.h5") for folder in READ_FOLDERS]]
    record = read_record(work / "_pca" / "pca.h5")
    assert record["project"] is None
    assert record["notes"]["sessions"] == [f"{folder}/results.h5" for folder in READ_FOLDERS]
    assert record["notes"]["exclude"] == [left_out]
    assert list(tmp_path.rglob("moseq2-project.yaml")) == []


#: The extraction the owning project lists, and the one beside it that it does not.
LISTED = ("cohort/xerus/depth.avi", "x", "uuid-x", "2020-01-01T10:00:00.000000-05:00")
UNLISTED = ("y", "uuid-y", "2020-01-02T10:00:00.000000-05:00")
UNLISTED_NAME = "y/results.h5"


@pytest.mark.parametrize("left_out", [None, LISTED[0], UNLISTED_NAME], ids=["none", "listed", "unlisted"])
def test_a_folder_another_project_partly_owns_is_named_by_owner_and_path(tmp_path, wrapper, left_out):
    """The owning project names the extraction it lists, the other is named by its path, and ``--exclude``
    takes either name. The run is outside any project, so the owner's exclude (which names its session) is not
    applied and nothing is written to the owner."""
    owner, work = tmp_path / "owner", tmp_path / "work"
    key, folder, uuid, start = LISTED
    write_extraction(owner / "aggregate_results" / folder / "results.h5", uuid, "xerus", start)
    write_extraction(owner / "aggregate_results" / UNLISTED[0] / "results.h5", UNLISTED[1], "other", UNLISTED[2])
    write_yaml({"camera_type": "auto", "pca": {}, "exclude": [key]}, str(owner / "config.yaml"))
    write_yaml({"sessions": {key: {"recording": key, "folder": f"aggregate_results/{folder}", "uuid": uuid}},
                "pca": {}, "models": {"fitted": [], "current": None, "analysis": {}}, "applied_models": []},
               str(owner / "moseq2-project.yaml"))
    owner_files = {path: path.read_bytes() for path in (owner / "config.yaml", owner / "moseq2-project.yaml")}
    work.mkdir()
    typed = ["--exclude", left_out] if left_out else []

    result = train_pca_command_line("-i", owner / "aggregate_results", *typed, cwd=work)

    assert result.exit_code == 0, result.output
    names = {key: owner / "aggregate_results" / folder / "results.h5",
             UNLISTED_NAME: owner / "aggregate_results" / UNLISTED[0] / "results.h5"}
    used = [name for name in names if name != left_out]
    assert wrapper == [[str(names[name]) for name in used]]
    record = read_record(work / "_pca" / "pca.h5")
    assert record["project"] is None
    assert record["notes"]["sessions"] == used
    assert record["notes"]["exclude"] == ([left_out] if left_out else [])
    assert {path: path.read_bytes() for path in owner_files} == owner_files
    assert not (owner / "_pca").exists()


# ------------------------------------------------------------------ an output that already exists


def question(path: Path) -> str:
    return f"The file {path} already exists.\nWould you like to overwrite it? [y -> yes, n -> no]\n\n"


def test_answering_yes_replaces_and_records_the_options_as_resolved(root, wrapper):
    existing = root / "_pca" / "pca.h5"
    existing.parent.mkdir()
    existing.write_bytes(b"an earlier run's components")

    result = train_pca_command_line("--config-file", root / "config.yaml", cwd=root, input="y\n")

    assert result.exit_code == 0, result.output
    assert result.output.startswith(question(existing)), result.output
    assert result.output.count("already exists") == 1
    assert result.output.count(EXCLUDE_LINE) == 1
    assert existing.read_bytes() != b"an earlier run's components"
    record = read_record(existing)
    assert record["options"]["overwrite_pca_train"] is False
    assert record["notes"]["sessions"] == READ


def test_answering_no_keeps_the_existing_components(root, wrapper):
    existing = root / "_pca" / "pca.h5"
    existing.parent.mkdir()
    existing.write_bytes(b"an earlier run's components")
    bookkeeping = (root / "moseq2-project.yaml").read_bytes()

    result = train_pca_command_line("--config-file", root / "config.yaml", cwd=root, input="n\n")

    assert result.exit_code == 0, result.output
    assert result.output == question(existing)
    assert existing.read_bytes() == b"an earlier run's components"
    assert (root / "moseq2-project.yaml").read_bytes() == bookkeeping
    assert wrapper == []


def test_the_notebook_asks_the_same_question(root, wrapper, monkeypatch, capsys):
    from moseq2_pca.gui import apply_pca_command

    train(root)
    existing = root / "_pca" / "pca_scores.h5"
    existing.write_bytes(b"an earlier run's scores")
    capsys.readouterr()
    monkeypatch.setattr("builtins.input", lambda *prompt: "n")

    assert apply_pca_command(str(root)) == "PCA Scores have not been computed"

    assert capsys.readouterr().out.count(question(existing)) == 1
    assert existing.read_bytes() == b"an earlier run's scores"
