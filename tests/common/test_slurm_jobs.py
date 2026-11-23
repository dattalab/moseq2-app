"""What a SLURM job runs is fixed when it is submitted.

``moseq2 extract`` with ``computing.extract.cluster_type: slurm`` and
``kappa-scan`` with ``cluster_type: slurm`` write one ``sbatch`` line per job
into a script, and each job runs its step from a settings file of its own
(``moseq2_common.slurm.job_settings_file``): a session's in its folder in
``aggregate_results/``, which ``moseq2 run-job`` reads, and a kappa-scan fit's
beside its model, passed as ``--config-file``. The file holds the step's
settings in ``config.yaml``'s shape and a ``job:`` block naming the project and
the session (with the uuid the submitter gave it, its recording and the cores
the job asks for) or the model, so a later edit to ``config.yaml`` does not
change a queued job. Every path in a ``--wrap`` command is quoted for the
shell, so a project under ``C:\\Users\\Jane Doe\\OneDrive`` or
``/home/jane doe`` reaches the job whole.

Nothing is submitted: the script is read, and each command is run in the test
with the step's work function replaced.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import h5py
import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_extract.steps as extract_steps
from moseq2_app.cli import cli as moseq2_cli
from moseq2_common import slurm
from moseq2_common.bookkeeping import load
from moseq2_common.project import Project
from moseq2_common.yaml import read_yaml, write_yaml
from extract_projects import recorded_extraction

WINDOWS_PROJECT = r"C:\Users\Jane Doe\OneDrive\proj"
WINDOWS_PYTHON = r"C:\Users\Jane Doe\miniconda3\envs\moseq2\python.exe"


def _words(command: str) -> list[str]:
    """``command`` split as a POSIX shell splits it, with ``;`` its own word."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def _wrapped(line: str) -> list[str]:
    """The words of the command an ``sbatch ... --wrap`` line runs."""
    args = shlex.split(line)
    return _words(args[args.index("--wrap") + 1])


def _run(cli, *args, cwd=None):
    before = os.getcwd()
    if cwd is not None:
        os.chdir(cwd)
    try:
        result = CliRunner().invoke(cli, [str(a) for a in args], catch_exceptions=False)
    finally:
        os.chdir(before)
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture
def project(tmp_path):
    """A project in a folder with a space in its name: two recordings and a config.yaml set to SLURM."""
    folder = tmp_path / "jane doe" / "my project"
    for name in ("session 1", "session_2"):
        session = folder / name
        session.mkdir(parents=True)
        (session / "depth.avi").write_bytes(b"\0")  # a placeholder: the work functions are stubbed, and an empty input stops a step
        (session / "metadata.json").write_text("{}")
    (folder / "flip.npz").write_bytes(b"\0")  # a placeholder head/tail classifier, so nothing is downloaded
    config = folder / "config.yaml"
    _run(extract_cli.cli, "generate-config", "-o", config)
    data = read_yaml(str(config))
    data["extract"].update(crop_size=[90, 90], flip_classifier=str(folder / "flip.npz"))
    # a time per job given, since `auto` reads each placeholder recording's length
    data["computing"]["extract"].update(cluster_type="slurm", wall_time="3:00:00")
    write_yaml(data, str(config))
    return Project.at(folder)


@pytest.fixture
def received(monkeypatch):
    """Each job run in the test: the settings and uuid its extraction was given."""
    runs = []
    monkeypatch.setattr(
        extract_steps, "extract_wrapper",
        lambda f, out, config, num_frames=None, skip=False, uuid=None, record=None:
        runs.append({"recording": f, "folder": out, "config": config, "uuid": uuid}),
    )
    popen, system = subprocess.Popen, os.system

    class NoSbatch(popen):
        def __init__(self, args, *rest, **kwargs):
            if "sbatch" in str(args):
                pytest.fail("moseq2 extract submitted a job")
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", NoSbatch)
    monkeypatch.setattr(os, "system", lambda command: pytest.fail("moseq2 extract submitted a job")
                        if "sbatch" in str(command) else system(command))
    return runs


def _extract(project, *recordings):
    """``moseq2 extract`` in ``project``; the lines of the script it wrote."""
    _run(moseq2_cli, "extract", *recordings, cwd=project.folder)
    return [line for line in (Path(project.folder) / "extract_out.sh").read_text().splitlines() if line.strip()]


def _run_job(line):
    """Run the extraction one sbatch line's job would run."""
    words = _wrapped(line)
    assert words[:4] == [sys.executable, "-m", "moseq2_app", "run-job"], words
    _run(moseq2_cli, *words[3:words.index(";")])


def _session_folder(project, key):
    from moseq2_extract.recordings import session_outputs

    return session_outputs(project, key).folder


def test_slurm_commands_quote_paths_with_spaces(project, received, tmp_path):
    """The ``--wrap`` command parses back to the same paths, for a POSIX and a Windows project path."""
    lines = _extract(project)

    assert len(lines) == 2
    for line in lines:
        words = _wrapped(line)
        assert words[:4] == [sys.executable, "-m", "moseq2_app", "run-job"] and words[5:] == [";"], words
        job_file = Path(words[4])
        assert job_file.is_file()
        assert " " in str(job_file)
        job = read_yaml(str(job_file))
        assert (Path(job["job"]["project"]) / job["job"]["recording"]).is_file()

    windows = "\\".join([WINDOWS_PROJECT, "aggregate_results", "session_1", "extract-job.yaml"])
    line = slurm.sbatch_command(slurm.command_line(WINDOWS_PYTHON, "-m", "moseq2_app", "run-job", windows) + "; ",
                                ncpus=1, memory="5GB", partition="short", wall_time="3:00:00",
                                prefix="module load x; ")
    assert _wrapped(line) == ["module", "load", "x", ";", WINDOWS_PYTHON, "-m", "moseq2_app", "run-job", windows,
                              ";"]

    from moseq2_model.util import create_command_strings

    scores = "\\".join([WINDOWS_PROJECT, "_pca", "pca_scores.h5"])
    models = tmp_path / "jane doe" / "models"
    config = dict(npcs=10, num_iter=50, index=None, separate_trans=False, robust=False, e_step=False,
                  hold_out=False, max_states=20, ncpus=0, cluster_type="slurm", memory="5GB", partition="short",
                  wall_time="3:00:00", prefix="")
    line = create_command_strings(scores, str(models), config, [100]).splitlines()[-1]
    words = _wrapped(line)
    assert words[:4] == ["moseq2-model", "learn-model", scores, str(models / "model-000-100.p")]
    assert Path(words[words.index("--config-file") + 1]) == slurm.job_settings_file(models, "model-000-100")


def test_job_files_are_written_only_for_submitted_recordings(project, received):
    """A recording the plan leaves as it is, up to date, gets no job and no job file."""
    write_yaml({"session 1/depth.avi": {"crop_size": [110, 110]}}, str(project.session_config_file))
    recorded_extraction(project, "session_2/depth.avi")

    lines = _extract(project)

    assert len(lines) == 1
    assert sorted(Path(project.folder).rglob("*-job.yaml")) == [
        slurm.job_settings_file(_session_folder(project, "session 1/depth.avi"), "extract")]
    assert list(Path(project.folder).rglob("config.yaml")) == [project.config_file]
    job = read_yaml(str(slurm.job_settings_file(_session_folder(project, "session 1/depth.avi"), "extract")))
    assert job["extract"]["crop_size"] == [110, 110]
    assert job["job"]["project"] == str(Path(project.folder).absolute())
    assert job["job"]["session"] == "session 1/depth.avi"


def test_a_queued_job_is_unaffected_by_a_later_config_edit(project, received):
    lines = _extract(project)
    data = read_yaml(str(project.config_file))
    data["extract"]["crop_size"] = [70, 70]
    write_yaml(data, str(project.config_file))

    for line in lines:
        _run_job(line)

    assert len(received) == 2
    assert [tuple(run["config"]["crop_size"]) for run in received] == [(90, 90), (90, 90)]


def test_a_submitted_session_has_its_uuid_before_the_job_runs(project, received):
    """The submitter gives each session its uuid in moseq2-project.yaml and its job file; the job extracts with it."""
    lines = _extract(project)

    sessions = load(project).sessions
    assert sorted(sessions) == ["session 1/depth.avi", "session_2/depth.avi"]
    for key, session in sessions.items():
        job = read_yaml(str(slurm.job_settings_file(_session_folder(project, key), "extract")))
        assert job["job"] == {"project": str(Path(project.folder).absolute()), "session": key,
                              "uuid": session.uuid, "recording": key, "cpus": 4}

    for line in lines:
        _run_job(line)
    by_recording = {Path(run["recording"]).parent.name: run["uuid"] for run in received}
    assert by_recording == {"session 1": sessions["session 1/depth.avi"].uuid,
                            "session_2": sessions["session_2/depth.avi"].uuid}

    # submitting again keeps each session's uuid
    _extract(project)
    assert {k: s.uuid for k, s in load(project).sessions.items()} == {k: s.uuid for k, s in sessions.items()}


def test_get_best_model_registers_kappa_scan_models(tmp_path, monkeypatch):
    """The fits a kappa scan's jobs wrote enter moseq2-project.yaml when get-best-model compares them."""
    import moseq2_viz.steps as viz_steps
    from moseq2_viz.options import BestModelOptions

    project = Project.at(tmp_path)
    write_yaml({"sessions": {}, "models": {"fitted": ["models/model.p"], "current": "models/model.p"}},
               str(project.bookkeeping_file))
    models = tmp_path / "models"
    models.mkdir()
    for name in ("model.p", "model-000-10.p", "model-001-1000.p"):
        (models / name).touch()
    (models / "model-000-10-job.yaml").touch()
    best = str(models / "model-001-1000.p")
    monkeypatch.setattr(viz_steps, "get_best_fit_model_wrapper",
                        lambda *args: ({"best model - duration (mean match)": best}, None))

    h5py.File(tmp_path / "changepoints.h5", "w").close()  # a placeholder: the comparison is stubbed
    viz_steps.get_best_model(str(models), str(tmp_path / "changepoints.h5"), None, BestModelOptions(),
                             project=project)

    assert load(project).models["fitted"] == ["models/model.p", "models/model-000-10.p", "models/model-001-1000.p"]
    assert load(project).models["current"] == "models/model-001-1000.p"
