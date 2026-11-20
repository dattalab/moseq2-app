"""PCA's file layer runs on the extractions it is given and knows nothing of projects.

``moseq2_pca.steps`` takes the extractions as (session name, extraction ``.h5``)
pairs, an output folder and name, and an options object. Run on extractions that
sit inside a project folder, from that folder, a step still reads none of the
project's files and writes nothing but its outputs: the extractions here sit in
a trap project (``tests/layers.py``) whose ``moseq2-project.yaml`` lists them under
other names, whose ``config.yaml`` leaves one of them out and sets other filters,
and whose settings would stop scores and changepoints if a step compared the
components with them. An output that already exists, with overwrite off, is a
decision the step raises before anything else; neither layer asks it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import h5py
import numpy as np
import pytest

import moseq2_pca
import moseq2_pca.util as util
from layers import listing, trap_project

from moseq2_common.project import OverwriteNeeded
from moseq2_common.records import read_record
from moseq2_pca import steps
from moseq2_pca.helpers.data import read_trained_settings
from moseq2_pca.options import ApplyPCAOptions, ChangepointsOptions, TrainPCAOptions

#: (name the step is given, the trap's key for it, folder in aggregate_results/, uuid, start time). The second
#: was recorded first, so a step that sorted the sessions would read them in the other order.
SESSIONS = (
    ("a", "trap/ant/depth.avi", "ant", "uuid-a", "2020-01-02T10:00:00.000000-05:00"),
    ("b", "trap/bee/depth.avi", "bee", "uuid-b", "2020-01-01T10:00:00.000000-05:00"),
)
STEPS = ("train_pca", "apply_pca", "compute_changepoints")
OUTPUT_FILES = {"train_pca": "pca", "apply_pca": "pca_scores", "compute_changepoints": "changepoints"}


def call(step, sessions, output_dir, output_file, opts, **kwargs):
    """``step`` of the file layer, on ``sessions``: (session name, extraction ``.h5``) pairs."""
    return getattr(steps, step)(sessions, output_dir, output_file, opts, **kwargs)


def options(step):
    """Options for ``step`` that run quickly on one worker: Dask for training and changepoints, none for scores."""
    if step == "train_pca":
        return TrainPCAOptions(cluster_type="local", nworkers=1)
    if step == "apply_pca":
        return ApplyPCAOptions(cluster_type="nodask")
    return ChangepointsOptions(nworkers=1)


def write_extraction(path: Path, uuid: str, name: str, start: str) -> None:
    rng = np.random.default_rng(len(uuid) + ord(name[0]))
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        f["frames"] = rng.integers(0, 60, (200, 80, 80)).astype("uint8")
        f["timestamps"] = np.arange(200) * 1000 / 30
        f["metadata/fps"] = 30.0
        f["metadata/uuid"] = uuid
        f["metadata/acquisition/SessionName"] = name
        f["metadata/acquisition/StartTime"] = start


@pytest.fixture
def trapped(tmp_path, monkeypatch):
    """Two extractions inside a trap project, the working folder the trap; (the trap, the pairs to give)."""
    monkeypatch.setattr(util, "get_env_cpu_and_mem", lambda: (2e9, 2))
    folder = tmp_path / "trap"
    pairs = []
    for name, _, where, uuid, start in SESSIONS:
        h5 = folder / "aggregate_results" / where / "results.h5"
        write_extraction(h5, uuid, name, start)
        pairs.append((name, str(h5)))
    trap = trap_project(folder, sessions={key: f"aggregate_results/{where}" for _, key, where, *_ in SESSIONS},
                        exclude=[SESSIONS[0][1]])
    monkeypatch.chdir(folder)
    return trap, pairs


def run_through(step, pairs, out):
    """The file layer's steps in order, up to and including ``step``; its result."""
    components, scores = str(out / "pca.h5"), str(out / "pca_scores.h5")
    for each in STEPS[:STEPS.index(step) + 1]:
        files = {} if each == "train_pca" else {"components_file": components}
        if each == "compute_changepoints":
            files["scores_file"] = scores
        result = call(each, pairs, str(out), OUTPUT_FILES[each], options(each), **files)
    return result


@pytest.mark.parametrize("step", STEPS)
def test_each_step_on_plain_files_reads_and_writes_no_project_file(trapped, step):
    trap, pairs = trapped
    out = trap.folder / "standalone_pca"
    before = listing(trap.folder)

    run_through(step, pairs, out)

    record = read_record(out / f"{OUTPUT_FILES[step]}.h5")
    assert record["project"] is None
    assert [item["relative_path"] for item in record["inputs"]] == [None] * len(record["inputs"])
    assert record["notes"]["sessions"] == ["a", "b"]
    assert record["notes"]["exclude"] == []
    assert [item["path"] for item in record["inputs"] if "uuid" in item] == [h5 for _, h5 in pairs]
    if step == "train_pca":
        trained = read_trained_settings(str(out / "pca.h5"))
        assert trained["gaussfilter_space"] == list(TrainPCAOptions().gaussfilter_space)
        assert trained["medfilter_space"] == list(TrainPCAOptions().medfilter_space)
    assert trap.unchanged()
    new = listing(trap.folder) - before
    assert new and all(path == out or out in path.parents for path in new), sorted(new)


@pytest.mark.parametrize("step, option", [("train_pca", "overwrite_pca_train"), ("apply_pca", "overwrite_pca_apply")])
def test_an_existing_output_is_a_decision_raised_before_anything_is_printed(trapped, capsys, step, option):
    _, pairs = trapped
    out = Path.cwd() / "_pca"
    out.mkdir()
    existing = out / f"{OUTPUT_FILES[step]}.h5"
    existing.write_bytes(b"an earlier run's output")

    with pytest.raises(OverwriteNeeded) as decision:
        call(step, pairs, str(out), OUTPUT_FILES[step], options(step))

    assert decision.value.output == existing
    assert decision.value.option == option
    assert decision.value.keyword == "replace_existing"
    assert capsys.readouterr() == ("", "")
    assert existing.read_bytes() == b"an earlier run's output"

    kept = call(step, pairs, str(out), OUTPUT_FILES[step], options(step), replace_existing=False)

    assert kept == (steps.TrainPCAResult(None) if step == "train_pca" else steps.ApplyPCAResult(None))
    assert capsys.readouterr() == ("", "")
    assert existing.read_bytes() == b"an earlier run's output"


@pytest.mark.parametrize("step", STEPS)
def test_a_step_given_no_extractions_says_what_to_pass(tmp_path, step):
    out = tmp_path / "_pca"

    with pytest.raises(ValueError) as error:
        call(step, [], str(out), OUTPUT_FILES[step], options(step))

    assert str(error.value) == (
        f"{step} was given no extractions to read. Pass each session as a (session name, path to its extraction "
        ".h5) pair."
    )
    assert not out.exists()


def _module(name: str) -> ast.Module:
    return ast.parse((Path(moseq2_pca.__file__).parent / f"{name}.py").read_text())


#: What reads or writes a project's files: none of it is imported by the file layer.
PROJECT_NAMES = {"session_index", "sessions_folder", "extraction_keys", "check_given_exclude", "load", "update",
                 "relative_path", "read_config", "read_exclude", "find_project"}


def test_the_file_layer_reaches_no_project_file():
    module = _module("steps")
    imported = {alias.asname or alias.name for node in ast.walk(module)
                if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    imported |= {alias.name for node in ast.walk(module)
                 if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert not imported & PROJECT_NAMES, sorted(imported & PROJECT_NAMES)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            arguments = {arg.arg for arg in [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]}
            assert not arguments & {"project", "exclude"}, node.name


def test_neither_layer_asks():
    for name in ("steps", "project"):
        calls = [node for node in ast.walk(_module(name))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input"]
        assert calls == [], name
