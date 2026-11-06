"""Every step with two front ends gives one result through either.

The same project, copied twice, is taken through the pipeline once on the
command line and once through the notebook's functions: extraction of every
recording, aggregation, the three PCA steps, fitting a model, applying it and
picking the best model. After each step the two outputs must hold the same
data, and their records the same run, apart from the times, the front end and
the command typed. Each step's random draws are seeded alike on both sides (the
sessions' uuids, PCA's compressed SVD, the changepoints' random projections), so
a difference is a difference in what the front ends handed the step.

The ``.h5`` outputs are compared dataset by dataset, the model files key by key,
and the records field by field with each project's folder written as
``<project>``.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import uuid as uuid_module
from pathlib import Path

import h5py
import joblib
import numpy as np
import pytest
from click.testing import CliRunner

from moseq2_common.yaml import read_yaml, write_yaml

RECORDINGS = ("saline_example_0", "amphetamine_example_0")
FRONT_ENDS = ("cli", "notebook")

#: Record fields that describe when or how a run was started, not what it did.
NOT_COMPARED = {"started", "ended", "duration_seconds", "front_end", "command", "fingerprint"}


@contextlib.contextmanager
def working_in(folder):
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def cli(program, *args, cwd):
    """Run a command as a person in ``cwd`` types it; it must succeed. (The PCA commands' ``-o`` defaults to
    ``_pca`` in the folder the program started in, which is not ``cwd`` inside a test, so the tests type it.)"""
    module = __import__(f"moseq2_{program}.cli", fromlist=["cli"])
    with working_in(cwd):
        result = CliRunner().invoke(module.cli, [str(a) for a in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture(scope="module")
def projects(tmp_path_factory, raw_sessions):
    """Two copies of one project: two recordings and a config.yaml set for a small, quick run."""
    from moseq2_extract.gui import generate_config_command

    base = tmp_path_factory.mktemp("one_front_end")
    roots = {}
    for front_end in FRONT_ENDS:
        root = base / front_end
        root.mkdir()
        for name in RECORDINGS:
            shutil.copytree(Path(raw_sessions) / name, root / name)
        generate_config_command(str(root / "config.yaml"), camera_type="k2")
        config = read_yaml(str(root / "config.yaml"))
        config["extract"]["num_frames"] = 120
        config["pca"].update({"cluster_type": "local", "nworkers": 1, "overwrite_pca_train": True,
                              "overwrite_pca_apply": True, "dask_cache_path": str(root / "dask_cache")})
        config["model"].update({"num_iter": 5, "npcs": 5, "kappa": 100, "robust": False})
        write_yaml(config, str(root / "config.yaml"))
        roots[front_end] = root
    return roots


@pytest.fixture(autouse=True)
def seeded(monkeypatch):
    """The same random draws on both sides: uuids in extraction order, the SVD's and numpy's global seed."""
    import dask.array.linalg as linalg

    import moseq2_extract.helpers.wrappers as extract_wrappers
    import moseq2_pca.util as pca_util

    counter = iter(range(10 ** 6))
    monkeypatch.setattr(extract_wrappers, "uuid4", lambda: uuid_module.UUID(int=next(counter)))
    svd = linalg.svd_compressed
    monkeypatch.setattr(linalg, "svd_compressed", lambda *a, **k: svd(*a, **{**k, "seed": 0}))
    monkeypatch.setattr(pca_util, "get_env_cpu_and_mem", lambda: (2e9, 2))

    def reseed():
        np.random.seed(0)
        nonlocal counter
        counter = iter(range(10 ** 6))

    return reseed


def both(seeded, projects, cli_side, notebook_side):
    """Run one step on each side, with the same seeds."""
    for front_end, run in (("cli", cli_side), ("notebook", notebook_side)):
        seeded()
        with contextlib.redirect_stdout(io.StringIO()):
            run(projects[front_end])


# ------------------------------------------------------------------ comparing


def plain(value, root):
    """``value`` with ``root`` written as ``<project>``, arrays as lists."""
    if isinstance(value, dict):
        return {str(k): plain(v, root) for k, v in value.items() if k not in NOT_COMPARED}
    if isinstance(value, (list, tuple)):
        return [plain(v, root) for v in value]
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if isinstance(value, str):
        return value.replace(str(Path(root).resolve()), "<project>").replace(str(root), "<project>")
    return value


def same(a, b, where):
    """Assert two values read from the two sides are the same data."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(a) == sorted(b), f"{where}: keys {sorted(a)} != {sorted(b)}"
        for key in a:
            same(a[key], b[key], f"{where}/{key}")
    elif isinstance(a, (list, tuple)) and not (len(a) and isinstance(a[0], (int, float, np.number))):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f"{where}: {a} != {b}"
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{i}]")
    elif isinstance(a, (np.ndarray, list, tuple, float, np.number)) and not isinstance(a, (bool, np.bool_)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, f"{where}: shape {a.shape} != {b.shape}"
        if a.dtype.kind in "fc":
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=True, err_msg=where)
        else:
            np.testing.assert_array_equal(a, b, err_msg=where)
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"


def h5_contents(path, root):
    """Every dataset of an ``.h5`` (strings with the project folder written as ``<project>``), and its record."""
    from moseq2_common.records import read_record

    found = {}
    with h5py.File(path, "r") as f:
        def visit(name, node):
            if isinstance(node, h5py.Dataset):
                value = node[()]
                if isinstance(value, np.ndarray) and value.dtype.kind in "SOU":
                    value = [plain(v, root) for v in value.ravel().tolist()]
                found[name] = plain(value, root) if isinstance(value, (bytes, str)) else value
        f.visititems(visit)
    return found, plain(read_record(path), root)


def same_h5(projects, relative):
    sides = {front_end: h5_contents(projects[front_end] / relative, projects[front_end]) for front_end in FRONT_ENDS}
    (cli_data, cli_record), (nb_data, nb_record) = sides["cli"], sides["notebook"]
    same(cli_data, nb_data, f"{relative} data")
    same(cli_record, nb_record, f"{relative} record")


def same_record(projects, relative):
    from moseq2_common.records import read_record

    same(*[plain(read_record(projects[f] / relative), projects[f]) for f in FRONT_ENDS], f"{relative} record")


def bookkeeping(projects):
    same(*[read_yaml(str(projects[f] / "moseq2-project.yaml")) for f in FRONT_ENDS], "moseq2-project.yaml")


# ------------------------------------------------------------------ the steps, in order


def test_extract(projects, seeded):
    from moseq2_extract.gui import extract_found_sessions

    both(seeded, projects,
         lambda root: cli("extract", "batch-extract", root, "--config-file", root / "config.yaml",
                                  "--skip-completed", cwd=root),
         lambda root: extract_found_sessions(str(root), str(root / "config.yaml"), extract_all=True,
                                             skip_extracted=True))
    for name in RECORDINGS:
        same_h5(projects, Path("aggregate_results") / name / "results.h5")


def test_aggregate_results(projects, seeded):
    from moseq2_extract.gui import aggregate_extract_results_command

    both(seeded, projects, lambda root: cli("extract", "aggregate-results", cwd=root),
         lambda root: aggregate_extract_results_command(str(root)))
    bookkeeping(projects)


def test_train_pca(projects, seeded):
    from moseq2_pca.gui import train_pca_command

    both(seeded, projects, lambda root: cli("pca", "train-pca", "-o", root / "_pca", "--config-file", root / "config.yaml",
                             cwd=root),
         lambda root: train_pca_command(str(root)))
    same_h5(projects, "_pca/pca.h5")
    bookkeeping(projects)


def test_apply_pca(projects, seeded):
    from moseq2_pca.gui import apply_pca_command

    both(seeded, projects, lambda root: cli("pca", "apply-pca", "-o", root / "_pca", "--config-file", root / "config.yaml",
                             cwd=root),
         lambda root: apply_pca_command(str(root)))
    same_h5(projects, "_pca/pca_scores.h5")
    bookkeeping(projects)


def test_compute_changepoints(projects, seeded):
    from moseq2_pca.gui import compute_changepoints_command

    both(seeded, projects,
         lambda root: cli("pca", "compute-changepoints", "-o", root / "_pca", "--config-file", root / "config.yaml",
                             cwd=root),
         lambda root: compute_changepoints_command(str(root)))
    same_h5(projects, "_pca/changepoints.h5")
    bookkeeping(projects)


def model_contents(path, root):
    """A model file's contents, but the fitted model object (its parameters are compared under their own key)."""
    return plain({k: v for k, v in joblib.load(path).items() if k != "model"}, root)


def test_learn_model(projects, seeded):
    from moseq2_model.gui import learn_model_command

    both(seeded, projects,
         lambda root: cli("model", "learn-model", root / "_pca" / "pca_scores.h5", root / "models" / "model.p",
                          "--config-file", root / "config.yaml", cwd=root),
         lambda root: learn_model_command(str(root)))
    same(*[model_contents(projects[f] / "models" / "model.p", projects[f]) for f in FRONT_ENDS], "model.p")
    same_record(projects, "models/model.p")
    bookkeeping(projects)


def test_apply_model(projects, seeded):
    from moseq2_model.gui import apply_model_command

    both(seeded, projects,
         lambda root: cli("model", "apply-model", root / "models" / "model.p", root / "_pca" / "pca_scores.h5",
                          root / "applied_models" / "model.p", cwd=root),
         lambda root: apply_model_command(str(root), str(root / "models" / "model.p")))
    same(*[model_contents(projects[f] / "applied_models" / "model.p", projects[f]) for f in FRONT_ENDS],
         "applied_models/model.p")
    same_record(projects, "applied_models/model.p")
    bookkeeping(projects)


def test_get_best_model(projects, seeded, monkeypatch):
    import matplotlib.figure

    from moseq2_viz.gui import get_best_fit_model

    monkeypatch.setattr(matplotlib.figure.Figure, "show", lambda *a, **k: None)
    both(seeded, projects,
         lambda root: cli("viz", "get-best-model", root / "models", root / "_pca" / "changepoints.h5",
                          root / "plots" / "model_vs_pc_changepoints", cwd=root),
         lambda root: get_best_fit_model(str(root)))
    same_record(projects, "plots/model_vs_pc_changepoints")
    bookkeeping(projects)
