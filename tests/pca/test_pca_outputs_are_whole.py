"""A PCA step's output appears under its final name only when it is complete, with its record.

``train-pca``, ``apply-pca`` and ``compute-changepoints`` write their ``.h5``
(and figures) under temporary names, store the run's record in the ``.h5``, and
rename them into place at the end (``moseq2_common.records.atomic_output``). A
step stopped partway (Ctrl-C, a killed job) leaves the previous output as it
was and nothing half-written under the final name. 1.5.0 wrote straight to the
final name, and the record was added afterwards.
"""

from __future__ import annotations

import builtins

import h5py
import numpy as np
import pytest

import moseq2_pca.util as util
from moseq2_common.yaml import read_yaml, write_yaml

SESSIONS = (("cohort/xerus/depth.avi", "cohort_xerus", "uuid-x"), ("cohort/yak/depth.avi", "cohort_yak", "uuid-y"))


@pytest.fixture
def project(tmp_path, monkeypatch):
    from moseq2_extract.gui import generate_config_command

    monkeypatch.setattr(util, "get_env_cpu_and_mem", lambda: (2e9, 2))
    root = tmp_path / "project"
    root.mkdir()
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    config = read_yaml(str(root / "config.yaml"))
    config["pca"].update({"cluster_type": "local", "nworkers": 1, "dask_cache_path": str(tmp_path / "cache")})
    write_yaml(config, str(root / "config.yaml"))
    rng = np.random.default_rng(0)
    for key, folder, uuid in SESSIONS:
        path = root / "aggregate_results" / folder / "results.h5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as f:
            f["frames"] = rng.integers(0, 60, (200, 80, 80)).astype("uint8")
            f["timestamps"] = np.arange(200) * 1000 / 30
            f["metadata/fps"] = 30.0
            f["metadata/uuid"] = uuid
            f["metadata/acquisition/SessionName"] = key.split("/")[1]
    write_yaml({"sessions": {key: {"recording": key, "folder": f"aggregate_results/{folder}", "uuid": uuid}
                             for key, folder, uuid in SESSIONS}}, str(root / "moseq2-project.yaml"))
    return root


def run(project, step, **opts):
    from moseq2_common.project import Project
    from moseq2_pca import options, steps

    out = str(project / "_pca")
    if step == "train":
        return steps.train_pca(None, out, "pca", options.TrainPCAOptions(
            cluster_type="local", nworkers=1, overwrite_pca_train=True, dask_cache_path=str(project / "cache")),
            project=Project.at(project))
    return steps.apply_pca(None, out, "pca_scores", options.ApplyPCAOptions(cluster_type="nodask", **opts),
                           components_file=str(project / "_pca" / "pca.h5"), project=Project.at(project))


def interrupt_the_record(monkeypatch):
    """Stop the step (Ctrl-C) as it stores its record, after its arrays are written."""
    import moseq2_pca.steps as steps

    def stopped(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(steps, "write_h5_record", stopped)


def leftovers(project):
    return sorted(p.name for p in (project / "_pca").iterdir() if "partial" in p.name)


@pytest.mark.parametrize("step, output", [("train", "pca.h5"), ("apply", "pca_scores.h5")])
def test_an_interrupted_pca_step_leaves_the_previous_output(project, monkeypatch, step, output):
    run(project, "train")
    if step == "apply":
        run(project, "apply", overwrite_pca_apply=True)
    final = project / "_pca" / output
    before = final.read_bytes()

    interrupt_the_record(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        run(project, step, overwrite_pca_apply=True)

    assert final.read_bytes() == before
    assert leftovers(project) == []


def test_a_completed_pca_output_carries_its_record(project):
    from moseq2_common.records import read_record

    run(project, "train")
    run(project, "apply", overwrite_pca_apply=True)
    for name, step in (("pca.h5", "train-pca"), ("pca_scores.h5", "apply-pca")):
        assert read_record(project / "_pca" / name)["step"] == step
    figures = sorted(p.name for p in (project / "_pca").glob("pca_*.p*"))
    assert figures == ["pca_components.pdf", "pca_components.png", "pca_scree.pdf", "pca_scree.png"]
    assert leftovers(project) == []


def test_declining_to_overwrite_keeps_the_existing_scores(project, monkeypatch):
    run(project, "train")
    run(project, "apply", overwrite_pca_apply=True)
    final = project / "_pca" / "pca_scores.h5"
    before = final.read_bytes()
    monkeypatch.setattr(builtins, "input", lambda *args: "n")
    result = run(project, "apply", overwrite_pca_apply=False)
    assert result.scores_file is None
    assert final.read_bytes() == before
    assert leftovers(project) == []
