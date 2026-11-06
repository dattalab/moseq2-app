"""Dask's messages go to the command's log file, not the console.

``train-pca``, ``apply-pca`` and ``compute-changepoints`` run under dask, and
distributed logs freely to the console: worker memory warnings, port
fallbacks, shutdown chatter. Users could not tell those harmless messages from
a real failure. Each command already writes errors to its own log
(``train.log``, ``scores.log``, ``changepoints.log``); dask's messages below
ERROR now go there too, real errors still reach the console, and the console
says once where the log is.

Each test runs the command's wrapper on a few small sessions with a real local
dask cluster and replaces the computation itself with one that logs the way
distributed does.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import sys

import h5py
import numpy as np
import pytest

import moseq2_pca.helpers.wrappers as wrappers
import moseq2_pca.util as util
from moseq2_pca.cli import apply_pca, compute_changepoints, train_pca
from moseq2_pca.helpers.data import write_trained_settings

#: A warning distributed logs without rate limiting (its "Unmanaged memory use is
#: high" and garbage-collection warnings are let through once per ten seconds,
#: so a second test would not see them).
HARMLESS = ("Removing worker 'tcp://127.0.0.1:2' caused the cluster to lose already "
            "computed task(s), which will be recomputed elsewhere")
FAILURE = "Worker tcp://127.0.0.1:1 failed: this one is real"


@pytest.fixture(autouse=True)
def two_gigabytes(monkeypatch):
    """Size the local workers from fixed numbers rather than this machine's."""
    monkeypatch.setattr(util, "get_env_cpu_and_mem", lambda: (2e9, 2))


@pytest.fixture
def sessions(tmp_path):
    for uuid in ("uuid-a", "uuid-b"):
        folder = tmp_path / "input" / uuid
        folder.mkdir(parents=True)
        with h5py.File(folder / "results_00.h5", "w") as f:
            f["frames"] = np.random.default_rng(0).integers(0, 50, (20, 8, 8)).astype("uint8")
            f["timestamps"] = np.arange(20) * 33.3
        with h5py.File(folder / "results_00.h5", "a") as f:
            f["metadata/uuid"] = uuid
    return tmp_path / "input"


@pytest.fixture
def busy_port():
    """A port something else is listening on, so distributed falls back and warns."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        yield str(s.getsockname()[1])


def defaults(command, tmp_path, port, **extra):
    config = {p.name: p.default for p in command.params if not p.required}
    config.update(cluster_type="local", nworkers=1, dask_port=port,
                  dask_cache_path=str(tmp_path / "cache"), **extra)
    return config


#: What distributed logs at ERROR when the progress bar stops following a
#: computation: the scheduler's feed to it is cancelled. Nothing has failed.
CANCELLED = "feed cancelled at shutdown"


def log_like_distributed():
    """What distributed does during a run: chatter, a warning, a cancelled feed and one real error."""
    logging.getLogger("distributed.scheduler").warning(HARMLESS)
    logging.getLogger("distributed.scheduler").info("Receive client connection")
    try:
        raise asyncio.CancelledError()
    except asyncio.CancelledError:
        logging.getLogger("distributed.scheduler").exception(CANCELLED)
    logging.getLogger("distributed.core").error(FAILURE)


def check(capfd, log_file, port_warning=True):
    out, err = capfd.readouterr()
    console = out + err
    assert HARMLESS not in console
    assert "already in use" not in console
    assert "Receive client connection" not in console
    assert CANCELLED not in console and "CancelledError" not in console
    assert FAILURE in console
    assert str(log_file) in console
    written = log_file.read_text()
    assert HARMLESS in written
    if port_warning:
        assert "already in use" in written
    assert FAILURE in written
    assert CANCELLED in written


def test_train_pca(tmp_path, sessions, busy_port, capfd, monkeypatch):
    def train(**kwargs):
        log_like_distributed()
        return {"components": np.eye(3, 64), "singular_values": np.ones(3),
                "explained_variance": np.ones(3), "explained_variance_ratio": np.ones(3) / 3,
                "mean": np.zeros(64)}

    monkeypatch.setattr(wrappers, "train_pca_dask", train)
    out = tmp_path / "_pca"
    wrappers.train_pca_wrapper(str(sessions), defaults(train_pca, tmp_path, busy_port), str(out), "pca")
    # only Linux refuses the busy port; elsewhere distributed binds it and has nothing to warn about
    check(capfd, out / "train.log", port_warning=sys.platform.startswith("linux"))


@pytest.fixture
def trained(tmp_path):
    out = tmp_path / "_pca"
    out.mkdir()
    with h5py.File(out / "pca.h5", "w") as f:
        f["components"] = np.eye(3, 64, dtype="float32")
        write_trained_settings(f, {"use_fft": False, "missing_data": False, "tailfilter_shape": "ellipse",
                                   "tailfilter_size": [9, 9], "gaussfilter_space": [1.5, 1], "gaussfilter_time": 0,
                                   "medfilter_time": [0], "medfilter_space": [0], "mask_height_threshold": 5,
                                   "mask_threshold": -16, "min_height": 10, "max_height": 120})
    return out


def test_apply_pca(tmp_path, sessions, trained, busy_port, capfd, monkeypatch):
    monkeypatch.setattr(wrappers, "apply_pca_dask", lambda **kwargs: log_like_distributed())
    config = defaults(apply_pca, tmp_path, busy_port, pca_file=str(trained / "pca.h5"),
                      overwrite_pca_apply=True)
    wrappers.apply_pca_wrapper(str(sessions), config, str(trained), "pca_scores")
    # apply-pca ignores every UserWarning, the port warning among them
    check(capfd, trained / "scores.log", port_warning=False)


def test_compute_changepoints(tmp_path, sessions, trained, busy_port, capfd, monkeypatch):
    def changepoints(save_file, **kwargs):
        log_like_distributed()
        with h5py.File(f"{save_file}.h5", "w") as f:
            f["cps/uuid-a"] = np.array([[1], [5], [9]])

    monkeypatch.setattr(wrappers, "get_changepoints_dask", changepoints)
    config = defaults(compute_changepoints, tmp_path, busy_port, pca_file=str(trained / "pca.h5"))
    wrappers.compute_changepoints_wrapper(str(sessions), config, str(trained), "changepoints")
    # compute-changepoints ignores every UserWarning, the port warning among them
    check(capfd, trained / "changepoints.log", port_warning=False)


def test_logging_is_restored_afterwards(tmp_path, sessions, busy_port, monkeypatch):
    """After the command, distributed's loggers have the handlers they had before."""
    names = ("distributed", "distributed.client", "distributed.gc", "dask", "tornado", "bokeh")
    before = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).propagate) for n in names}
    def trained(**kwargs):
        log_like_distributed()
        return {"components": np.zeros((4, 64)), "explained_variance_ratio": np.full(4, 0.25)}

    monkeypatch.setattr(wrappers, "train_pca_dask", trained)
    wrappers.train_pca_wrapper(str(sessions), defaults(train_pca, tmp_path, busy_port),
                               str(tmp_path / "_pca"), "pca")
    after = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).propagate) for n in names}
    assert after == before
    logging.getLogger("distributed.worker.memory").warning("after the command")
    log_file = tmp_path / "_pca" / "train.log"
    assert "after the command" not in (log_file.read_text() if log_file.exists() else "")
