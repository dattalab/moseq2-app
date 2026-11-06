"""What the progress display says, and what each step checks before its work.

The notebooks' project cell (``moseq2_app.project.progress``) prints what the
project holds and names every output older than what it was built from: PC
scores computed before one of their sessions was re-extracted, a model fit
before the scores were computed again, an output computed from a file that no
longer exists, and PCA outputs or a model computed before a session was added
to the project. It finds them by comparing the inputs an output's record holds
with the project's sessions and the inputs as they are now: an input the
pipeline wrote by the run its record names, any other by its size and content,
never by a file time. The message is for display; no step reads it or a record
to decide what to run.

Each step checks its inputs before any work (``moseq2_common.records.check_inputs``
and the step's own checks): a file it needs that does not exist, is empty or
cannot be read stops it with a message naming the step, the file and what to do,
and so does an extraction whose frames and timestamps disagree in number. 1.5.0
went on and failed later, inside a library, with a message naming neither.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_pca.util as pca_util
from moseq2_common.yaml import read_yaml, write_yaml

#: (session key, folder in aggregate_results/, uuid, start time).
SESSIONS = (
    ("cohort/xerus/depth.avi", "cohort_xerus", "uuid-x", "2020-01-01T10:00:00.000000-05:00"),
    ("cohort/yak/depth.avi", "cohort_yak", "uuid-y", "2020-01-02T10:00:00.000000-05:00"),
)


def write_session(path, uuid, key, start, *, frames=200, timestamps=None, seed=0):
    """A small extraction file: random 80 x 80 frames, 30 Hz timestamps, and the record of the extraction
    that wrote it."""
    from moseq2_common.records import start_record, write_h5_record

    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["frames"] = rng.integers(0, 60, (frames, 80, 80)).astype("uint8")
        f["timestamps"] = np.arange(frames if timestamps is None else timestamps) * 1000 / 30
        f["metadata/fps"] = 30.0
        f["metadata/uuid"] = uuid
        f["metadata/acquisition/SessionName"] = key.split("/")[1]
        f["metadata/acquisition/StartTime"] = start
        write_h5_record(f, start_record("extract", {}, []).finish({}))


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project of two extracted sessions, listed in its moseq2-project.yaml."""
    from moseq2_extract.gui import generate_config_command

    monkeypatch.setattr(pca_util, "get_env_cpu_and_mem", lambda: (2e9, 2))
    root = tmp_path / "project"
    root.mkdir()
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    config = read_yaml(str(root / "config.yaml"))
    config["pca"].update({"cluster_type": "local", "nworkers": 1, "overwrite_pca_train": True,
                          "overwrite_pca_apply": True, "dask_cache_path": str(tmp_path / "cache")})
    write_yaml(config, str(root / "config.yaml"))
    for i, (key, folder, uuid, start) in enumerate(SESSIONS):
        write_session(root / "aggregate_results" / folder / "results.h5", uuid, key, start, seed=i)
    write_yaml({"sessions": {key: {"recording": key, "folder": f"aggregate_results/{folder}", "uuid": uuid}
                             for key, folder, uuid, _ in SESSIONS}},
               str(root / "moseq2-project.yaml"))
    return root


def no_dask(monkeypatch):
    """Fail the test if a PCA step starts its work (its Dask cluster)."""
    import moseq2_pca.helpers.wrappers as wrappers

    def started(*args, **kwargs):
        raise AssertionError("the step started its work")

    monkeypatch.setattr(wrappers, "initialize_dask", started)


# ------------------------------------------------------------------ the progress display


def scores_recorded_from_sessions(project, sessions=SESSIONS):
    """PC scores in the project's _pca/, with the record apply-pca writes: each of ``sessions``' extractions
    an input."""
    from moseq2_common.bookkeeping import update
    from moseq2_common.project import Project
    from moseq2_common.records import start_record, write_h5_record

    proj = Project.at(project)
    scores = project / "_pca" / "pca_scores.h5"
    scores.parent.mkdir()
    sessions = [project / "aggregate_results" / folder / "results.h5" for _, folder, _, _ in sessions]
    record = start_record("apply-pca", {}, [proj.config_file, *sessions], project=proj)
    with h5py.File(scores, "w") as f:
        f["scores/uuid-x"] = np.zeros((200, 10))
        write_h5_record(f, record.finish({}))

    def change(bookkeeping):
        bookkeeping.pca["scores"] = "_pca/pca_scores.h5"

    update(proj, change)
    return scores


def test_progress_names_pca_scores_older_than_a_reextraction(project):
    """Re-extracting a session after apply-pca makes the scores older than it; the display says so, naming
    the scores, the session, when it was re-extracted and the step to run again. A config.yaml edit is not
    a re-extraction and is not named."""
    from moseq2_app.project import progress
    from moseq2_common.project import Project

    scores_recorded_from_sessions(project)
    assert "again" not in progress(Project.at(project))

    key, folder, uuid, start = SESSIONS[1]
    h5 = project / "aggregate_results" / folder / "results.h5"
    write_session(h5, uuid, key, start, seed=7)
    with open(project / "config.yaml", "a") as f:
        f.write("# edited\n")
    when = extracted_on(h5)

    shown = progress(Project.at(project))
    assert (f"_pca/pca_scores.h5 was computed before {key} was re-extracted ({when}); run apply-pca again "
            "to include it.") in shown, shown
    assert f"before {SESSIONS[0][0]}" not in shown, shown
    assert "config.yaml was" not in shown and shown.count("again") == 1, shown


def extracted_on(h5):
    """When the extraction that wrote ``h5`` ended, as the display gives it."""
    from datetime import datetime

    from moseq2_common.records import run_of

    return datetime.fromisoformat(run_of(h5)["ended"]).strftime("%Y-%m-%d %H:%M")


def test_a_reextraction_with_the_same_frames_is_named(project):
    """Re-extracting a session writes it again even when its frames come out the same: the scores were
    computed from the earlier run, and the display names it."""
    from moseq2_app.project import progress
    from moseq2_common.project import Project

    scores_recorded_from_sessions(project)
    key, folder, uuid, start = SESSIONS[0]
    h5 = project / "aggregate_results" / folder / "results.h5"
    write_session(h5, uuid, key, start, seed=0)

    shown = progress(Project.at(project))
    assert (f"_pca/pca_scores.h5 was computed before {key} was re-extracted ({extracted_on(h5)}); run "
            "apply-pca again to include it.") in shown, shown


def test_a_project_copied_with_new_file_times_is_not_out_of_date(project, tmp_path):
    """Copied with ``shutil.copytree`` and every file time moved a day on, nothing is named."""
    from moseq2_app.project import progress
    from moseq2_common.project import Project

    scores = scores_recorded_from_sessions(project)
    model_recorded_from(project, scores, SESSIONS)
    assert "again" not in progress(Project.at(project))

    moved = tmp_path / "moved"
    shutil.copytree(project, moved)
    for folder, _, names in os.walk(moved):
        for name in names:
            path = os.path.join(folder, name)
            stat = os.stat(path)
            os.utime(path, (stat.st_atime + 86400, stat.st_mtime + 86400))
    shown = progress(Project.at(moved))
    assert "again" not in shown and "no longer exists" not in shown, shown


def test_the_project_cell_prints_the_progress(project, capsys):
    from moseq2_app.project import start

    scores_recorded_from_sessions(project)
    key, folder, uuid, startt = SESSIONS[0]
    write_session(project / "aggregate_results" / folder / "results.h5", uuid, key, startt, seed=9)
    start(str(project))
    assert f"before {key} was re-extracted" in capsys.readouterr().out


def model_recorded_from(project, scores, sessions):
    """A model in the project's models/, with the record learn-model writes: the scores and each of
    ``sessions``' extractions its inputs."""
    from moseq2_common.bookkeeping import update
    from moseq2_common.project import Project
    from moseq2_common.records import start_record, write_record

    proj = Project.at(project)
    model = project / "models" / "model.p"
    model.parent.mkdir()
    model.write_bytes(b"a model")
    extractions = [project / "aggregate_results" / folder / "results.h5" for _, folder, _, _ in sessions]
    record = start_record("learn-model", {}, [proj.config_file, scores, *extractions], project=proj)
    write_record(model, record.finish({"files": [str(model)]}))

    def change(bookkeeping):
        bookkeeping.models["fitted"] = ["models/model.p"]
        bookkeeping.models["current"] = "models/model.p"

    update(proj, change)
    return model


def test_progress_names_outputs_computed_before_a_session_was_added(project):
    """PC scores and a model computed from the first session only, after which the second was aggregated
    into the project: each names the session it leaves out and the step to run again. Nothing is compared
    by file time."""
    from moseq2_app.project import progress
    from moseq2_common.project import Project

    scores = scores_recorded_from_sessions(project, SESSIONS[:1])
    model_recorded_from(project, scores, SESSIONS[:1])
    added = SESSIONS[1][0]

    shown = progress(Project.at(project))
    assert (f"_pca/pca_scores.h5 was computed before {added} was added to the project; run apply-pca again "
            "to include it.") in shown, shown
    assert (f"models/model.p was computed before {added} was added to the project; run learn-model again "
            "to include it.") in shown, shown
    assert f"before {SESSIONS[0][0]}" not in shown and shown.count("again") == 2, shown


def test_progress_leaves_out_a_session_listed_under_exclude(project):
    """A session config.yaml's exclude lists is meant to be left out; scores without it are not named."""
    from moseq2_app.project import progress
    from moseq2_common.project import Project

    scores_recorded_from_sessions(project, SESSIONS[:1])
    config = read_yaml(str(project / "config.yaml"))
    config["exclude"] = [SESSIONS[1][0]]
    write_yaml(config, str(project / "config.yaml"))
    assert "again" not in progress(Project.at(project))


def test_progress_names_a_session_extraction_that_no_longer_exists(project):
    """A session's extraction deleted after apply-pca and learn-model: both outputs name the session whose
    extraction they were computed from and is gone, rather than passing over it."""
    from moseq2_app.project import progress
    from moseq2_common.project import Project

    scores = scores_recorded_from_sessions(project)
    model_recorded_from(project, scores, SESSIONS)
    key, folder, _, _ = SESSIONS[1]
    (project / "aggregate_results" / folder / "results.h5").unlink()

    shown = progress(Project.at(project))
    for stored, step in (("_pca/pca_scores.h5", "apply-pca"), ("models/model.p", "learn-model")):
        assert (f"{stored} was computed from the extraction of {key} (aggregate_results/{folder}/results.h5), "
                "which no longer exists; extract that session again, or list its recording under exclude in "
                f"config.yaml, then run {step} again.") in shown, shown
    assert shown.count("no longer exists") == 2, shown


def test_progress_names_another_recorded_input_that_no_longer_exists(project):
    """The scores a model was fit on, deleted: the model names them."""
    from moseq2_app.project import progress
    from moseq2_common.project import Project

    scores = scores_recorded_from_sessions(project)
    model_recorded_from(project, scores, SESSIONS)
    scores.unlink()

    shown = progress(Project.at(project))
    assert ("models/model.p was computed from _pca/pca_scores.h5, which no longer exists; run learn-model "
            "again.") in shown, shown
    assert shown.count("no longer exists") == 1, shown


# ------------------------------------------------------------------ frame counts


def test_a_frame_count_that_disagrees_with_the_timestamps_stops_before_work(project, monkeypatch):
    """A session whose extraction holds 200 frames and 199 timestamps cannot have its dropped frames placed;
    train-pca stops before it starts its work, naming the file and both counts."""
    from moseq2_common.project import Project
    from moseq2_pca import steps
    from moseq2_pca.options import TrainPCAOptions

    key, folder, uuid, start = SESSIONS[1]
    h5 = project / "aggregate_results" / folder / "results.h5"
    write_session(h5, uuid, key, start, timestamps=199)
    no_dask(monkeypatch)
    with pytest.raises(ValueError) as error:
        steps.train_pca(None, str(project / "_pca"), "pca", TrainPCAOptions(cluster_type="local", nworkers=1),
                        project=Project.at(project))
    assert str(error.value) == (
        f"The extraction {h5} holds 200 frames (/frames) and 199 timestamps (/timestamps). Each frame needs "
        "one timestamp, so its dropped frames cannot be placed and its timing is unknown. Extract that "
        "session again, then run this step again."
    )
    assert not (project / "_pca" / "pca.h5").exists()


def test_extraction_stops_on_a_frame_count_mismatch_before_writing(tmp_path, raw_sessions):
    """The recording's video and its timestamps disagree: extraction stops before writing anything into the
    session's folder."""
    from moseq2_extract.cli import cli

    root = tmp_path / "project"
    session = root / "saline_example_0"
    session.mkdir(parents=True)
    source = Path(raw_sessions) / "saline_example_0"
    for name in ("depth.avi", "metadata.json"):
        shutil.copy(source / name, session / name)
    lines = (source / "depth_ts.txt").read_text().splitlines(keepends=True)
    (session / "depth_ts.txt").write_text("".join(lines[:-1]))
    CliRunner().invoke(cli, ["generate-config", "--output-file", str(root / "config.yaml")])
    result = CliRunner().invoke(cli, ["extract", str(session / "depth.avi"), "--config-file",
                                      str(root / "config.yaml")])
    assert isinstance(result.exception, ValueError), result.output
    assert "Frame count mismatch" in str(result.exception)
    assert not (root / "aggregate_results").exists()


# ------------------------------------------------------------------ empty and unreadable inputs


def test_an_empty_input_file_stops_before_work(tmp_path):
    """An empty recording stops extraction before any work, with a message naming the step, the file and
    what to do."""
    from moseq2_extract.cli import cli

    root = tmp_path / "project"
    session = root / "session_1"
    session.mkdir(parents=True)
    (session / "depth.avi").write_bytes(b"")
    (session / "metadata.json").write_text("{}")
    CliRunner().invoke(cli, ["generate-config", "--output-file", str(root / "config.yaml")])
    result = CliRunner().invoke(cli, ["extract", str(session / "depth.avi"), "--config-file",
                                      str(root / "config.yaml")])
    assert result.exit_code != 0
    message = str(result.exception) if not isinstance(result.exception, SystemExit) else result.output
    assert message.strip().endswith(
        f"extract needs {session / 'depth.avi'}, which is empty. Check that it is the recording you meant "
        "and that it was copied completely, or copy it again from where it was recorded."
    ), message
    assert not (root / "aggregate_results").exists()


def test_an_empty_session_extraction_stops_pca_before_work(project, monkeypatch):
    from moseq2_common.project import Project
    from moseq2_pca import steps
    from moseq2_pca.options import TrainPCAOptions

    h5 = project / "aggregate_results" / SESSIONS[0][1] / "results.h5"
    h5.write_bytes(b"")
    no_dask(monkeypatch)
    with pytest.raises(ValueError) as error:
        steps.train_pca(None, str(project / "_pca"), "pca", TrainPCAOptions(), project=Project.at(project))
    assert str(error.value) == (
        f"train-pca needs {h5}, which is empty. Extract that session again (moseq2-extract extract, or the "
        "notebook's extraction cell), then run this step again; or, to leave the session out, list its "
        "recording under exclude in config.yaml (the group widget's exclude column)."
    )


def test_an_unreadable_components_file_stops_apply_pca_before_work(project, monkeypatch):
    from moseq2_common.project import Project
    from moseq2_pca import steps
    from moseq2_pca.options import ApplyPCAOptions

    components = project / "_pca" / "pca.h5"
    components.parent.mkdir()
    components.write_bytes(b"not an h5 file")
    no_dask(monkeypatch)
    with pytest.raises(ValueError) as error:
        steps.apply_pca(None, str(project / "_pca"), "pca_scores", ApplyPCAOptions(),
                        components_file=str(components), project=Project.at(project))
    message = str(error.value)
    assert message.startswith(f"apply-pca needs {components}, which could not be read: "), message
    assert message.endswith(
        "Train PCA again (moseq2-pca train-pca, or the notebook's PCA cell), or give the components file "
        "it wrote with --pca-file."
    ), message
    assert not (project / "_pca" / "pca_scores.h5").exists()


def test_a_missing_scores_file_stops_learn_model_before_work(tmp_path):
    from moseq2_model import steps
    from moseq2_model.options import LearnModelOptions

    scores = tmp_path / "_pca" / "pca_scores.h5"
    with pytest.raises(ValueError) as error:
        steps.learn_model(str(scores), str(tmp_path / "models" / "model.p"), LearnModelOptions())
    assert str(error.value) == (
        f"learn-model needs {scores}, which does not exist. Compute the PC scores again (moseq2-pca "
        "apply-pca, or the notebook's scores cell), or give the scores file it wrote."
    )
    assert not (tmp_path / "models").exists()
