"""train-pca says before it starts when the training data will not fit in its workers.

``train_on_subset`` defaults to 1, so a large project trains on every frame:
178 sessions of about 36,000 frames at 80 x 80 is about 160 GB as float32,
and the run died hours in (moseq2-app#193). How big the training data is,
and how much memory the dask workers will have, are both known before any
work starts.

Training holds every training frame, cleaned and as float32, in the workers'
memory at once: the mean over all frames has to finish before any frame can
be centred. What does not fit, dask spills to disk at ``dask_cache_path``,
which is slower but finishes, so the check warns, however far over, with the
estimate and what to change, in numbers. It stops only when the disk under
``dask_cache_path`` has less free space than would be spilled, since then the
run would fail partway.

Each test trains on three small sessions (600 frames of 80 x 80, 15.36 MB as
float32) with the workers' memory faked, and stops the run where dask would
start.
"""

from __future__ import annotations

import re
import shutil

import h5py
import numpy as np
import pytest

import moseq2_pca.helpers.wrappers as wrappers
import moseq2_pca.util as util
from moseq2_pca.cli import train_pca
from moseq2_pca.util import write_yaml

FRAMES = 200
SESSIONS = 3
ESTIMATE = SESSIONS * FRAMES * 80 * 80 * 4


class ReachedDask(Exception):
    """Raised in place of starting dask: the run got past every check."""


@pytest.fixture
def sessions(tmp_path):
    for i in range(SESSIONS):
        folder = tmp_path / "input" / f"session-{i}"
        folder.mkdir(parents=True)
        with h5py.File(folder / "results_00.h5", "w") as f:
            f["frames"] = np.zeros((FRAMES, 80, 80), dtype="uint8")
            f["timestamps"] = np.arange(FRAMES) * 33.3
        write_yaml({"uuid": f"uuid-{i}"}, str(folder / "results_00.yaml"))
    return tmp_path / "input"


@pytest.fixture
def workers_have(monkeypatch):
    """Fake the memory the local workers are given, and stop where dask would start."""
    def fake(memory):
        monkeypatch.setattr(util, "get_env_cpu_and_mem", lambda: (memory, 4))

    def stop(*args, **kwargs):
        raise ReachedDask

    monkeypatch.setattr(wrappers, "initialize_dask", stop)
    return fake


def config(tmp_path, **extra):
    defaults = {p.name: p.default for p in train_pca.params if not p.required}
    return {**defaults, "cluster_type": "local", "nworkers": 1, "overwrite_pca_train": True,
            "dask_cache_path": str(tmp_path / "cache"), **extra}


def train(sessions, tmp_path, cfg):
    return wrappers.train_pca_wrapper(str(sessions), cfg, str(tmp_path / "_pca"), "pca")


def memory_warnings(recwarn):
    return [w for w in recwarn if "the dask workers have" in str(w.message)]


def suggested_subset(message):
    return float(re.search(r"train_on_subset to ([0-9.]+)", message).group(1))


def warned(sessions, tmp_path, cfg):
    """Train, expecting a memory warning and the run to go on to dask; the warning's text."""
    with pytest.warns(Warning) as record, pytest.raises(ReachedDask):
        train(sessions, tmp_path, cfg)
    messages = [str(w.message) for w in record if "the dask workers have" in str(w.message)]
    assert messages
    return "\n".join(messages)


def suggested_slurm(message):
    match = re.search(r"cluster_type slurm, nworkers (\d+), memory (\S+) per worker", message)
    return int(match.group(1)), match.group(2)


def test_training_that_fits_says_nothing(tmp_path, sessions, workers_have, recwarn):
    workers_have(ESTIMATE * 2)
    with pytest.raises(ReachedDask):
        train(sessions, tmp_path, config(tmp_path))
    assert not memory_warnings(recwarn)


def test_a_little_over_warns_and_goes_on(tmp_path, sessions, workers_have):
    """At 1.5 times what the workers have, dask spills to disk: slower, but it finishes."""
    workers_have(ESTIMATE / 1.5)
    with pytest.warns(Warning) as record, pytest.raises(ReachedDask):
        train(sessions, tmp_path, config(tmp_path))
    message = "\n".join(str(w.message) for w in record)
    assert "15.36 MB" in message
    assert "10.24 MB" in message
    assert "600 frames" in message
    assert str(tmp_path / "cache") in message
    assert suggested_subset(message) < 1
    assert suggested_slurm(message)


def test_far_over_with_disk_room_only_warns(tmp_path, sessions, workers_have):
    """At 50 times what the workers have, with room on disk for the spill, the run goes on."""
    workers_have(ESTIMATE / 50)
    message = warned(sessions, tmp_path, config(tmp_path))
    assert "15.36 MB" in message and "307.20 kB" in message
    assert "50.0 times" in message
    assert suggested_subset(message) < 1


def test_the_suggested_subset_fits(tmp_path, sessions, workers_have, recwarn):
    workers_have(ESTIMATE / 5)
    subset = suggested_subset(warned(sessions, tmp_path, config(tmp_path)))
    with pytest.raises(ReachedDask):
        train(sessions, tmp_path, config(tmp_path, train_on_subset=subset))
    assert not memory_warnings(recwarn)


def test_the_suggested_slurm_workers_fit(tmp_path, sessions, workers_have, recwarn):
    workers_have(ESTIMATE / 5)
    nworkers, memory = suggested_slurm(warned(sessions, tmp_path, config(tmp_path, memory="4MB")))
    assert memory == "4MB" and nworkers == 5
    with pytest.raises(ReachedDask):
        train(sessions, tmp_path, config(tmp_path, cluster_type="slurm", nworkers=nworkers,
                                         memory=memory))
    assert not memory_warnings(recwarn)


def test_no_disk_room_to_spill_to_stops(tmp_path, sessions, workers_have, monkeypatch):
    """At 1.5 times, dask would spill 5.12 MB; with 1 MB free on the cache's disk the run would fail."""
    workers_have(ESTIMATE / 1.5)
    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda path: real(path)._replace(free=1_000_000))
    with pytest.raises(RuntimeError) as e:
        train(sessions, tmp_path, config(tmp_path))
    message = str(e.value)
    assert "5.12 MB" in message and "1.00 MB free" in message
    assert str(tmp_path / "cache") in message
    assert suggested_subset(message) < 1
    assert not (tmp_path / "_pca" / "pca.yaml").exists()


def test_far_over_without_disk_room_stops(tmp_path, sessions, workers_have, monkeypatch):
    """At 5 times, dask would spill 12.29 MB; with 10 MB free the run would fail."""
    workers_have(ESTIMATE / 5)
    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda path: real(path)._replace(free=10_000_000))
    with pytest.raises(RuntimeError) as e:
        train(sessions, tmp_path, config(tmp_path))
    assert "12.29 MB" in str(e.value) and "10.00 MB free" in str(e.value)


def test_slurm_workers_that_are_too_small(tmp_path, sessions, workers_have):
    """On SLURM the workers have nworkers times memory."""
    workers_have(ESTIMATE * 100)
    message = warned(sessions, tmp_path, config(tmp_path, cluster_type="slurm", nworkers=1,
                                                  memory="3MB"))
    assert "15.36 MB" in message
    assert suggested_slurm(message) == (6, "3MB")


def test_the_estimate_counts_the_subset(tmp_path, sessions, workers_have, recwarn):
    """train_on_subset 0.1 trains on 60 frames, 1.536 MB, which fits in 2 MB."""
    workers_have(2e6)
    with pytest.raises(ReachedDask):
        train(sessions, tmp_path, config(tmp_path, train_on_subset=0.1))
    assert not memory_warnings(recwarn)


def test_the_notebook_cell_gets_the_same_check(tmp_path, sessions, workers_have):
    """The notebook's PCA cell calls train_pca_command, which reads the project's config.yaml."""
    from moseq2_pca.gui import train_pca_command

    workers_have(ESTIMATE / 5)
    config_file = tmp_path / "config.yaml"
    write_yaml({"cluster_type": "local", "nworkers": 1, "overwrite_pca_train": True,
                "dask_cache_path": str(tmp_path / "cache")}, str(config_file))
    progress = {"train_data_dir": str(sessions), "config_file": str(config_file)}
    with pytest.warns(Warning) as record, pytest.raises(ReachedDask):
        train_pca_command(progress, str(tmp_path / "_pca"), "pca")
    message = "\n".join(str(w.message) for w in record if "the dask workers have" in str(w.message))
    assert "15.36 MB" in message
    assert suggested_subset(message) < 1
