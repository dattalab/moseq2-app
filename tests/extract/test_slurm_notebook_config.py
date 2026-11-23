"""The Extract step's SLURM path extracts with the project's config file.

With ``cluster_type: slurm`` in config.yaml's ``computing.extract`` block, the
notebook's Extract step writes each recording's job settings file in its
session folder and one ``sbatch`` line per recording, running ``moseq2 run-job``
on that file, into ``extract_out.sh`` instead of extracting. Each job has to
extract with the project's config file, and the arena tool's settings from the
recording's entry when the session config has one. ``threads`` and
``compress_threads`` are the CPUs the job asks for, whatever either file holds.
A config written by ``generate-config`` names no config file of its own, and
no ``--config-file`` reaches the written lines.

Nothing is submitted and nothing is extracted while the lines are written: each
written job is then run through ``moseq2 run-job`` with ``extract_wrapper``
recorded, so what is checked is what the job would extract with.

``split_command`` (``tests/commands.py``) splits a command line written with a
platform's paths back into its arguments.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_extract.steps as extract_steps
from commands import split_command
from moseq2_app.cli import cli
from moseq2_common.yaml import read_yaml, write_yaml

SESSION = "session_1"
#: The fixture's head/tail classifier, so the project's extraction downloads nothing.
CLASSIFIER = "flip_classifier_k2_c57_10to13weeks.npz"
#: The CPUs, and so the threads, each cluster job asks for.
JOB_CPUS = 4


def run(group, *args):
    result = CliRunner().invoke(group, [str(a) for a in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture
def project(tmp_path, raw_sessions):
    """One recording and a config from ``generate-config``, set to SLURM, naming the fixture's head/tail
    classifier, and with values changed from their defaults."""
    session = tmp_path / SESSION
    session.mkdir()
    (session / "depth.avi").write_bytes(b"\0")  # a placeholder: the work functions are stubbed, and an empty input stops a step
    (session / "metadata.json").write_text("{}")
    shutil.copy(Path(raw_sessions) / CLASSIFIER, tmp_path / CLASSIFIER)

    config = tmp_path / "config.yaml"
    run(extract_cli.cli, "generate-config", "-o", str(config))
    data = read_yaml(str(config))
    # paths are not settings: config.yaml names no config file of its own
    assert "config_file" not in data["extract"]
    data["extract"].update(compress_threads=20, crop_size=[90, 90], flip_classifier=str(tmp_path / CLASSIFIER))
    # a placeholder recording has no length to measure a job's time from
    data["computing"]["extract"].update(cluster_type="slurm", wall_time="1:00:00")
    write_yaml(data, str(config))
    return tmp_path, config


@pytest.fixture
def jobs(monkeypatch):
    """Run the notebook's Extract step on SLURM, which writes the jobs and extracts nothing, then run each
    written job through ``moseq2 run-job``; the written lines and the config each job received."""
    received = []
    monkeypatch.setattr(
        extract_steps,
        "extract_wrapper",
        lambda f, out, config, num_frames=None, skip=False, uuid=None, record=None: received.append(config),
    )

    def go(root, config):
        from moseq2_app.notebook import extract

        before = os.getcwd()
        os.chdir(root)
        try:
            widget = extract()
            widget.start()
            state = widget.wait()["state"]
        finally:
            os.chdir(before)
        assert state == "finished", widget.problem
        assert received == [], "the Extract step extracted a recording itself on SLURM"
        lines = [line for line in (root / "extract_out.sh").read_text().splitlines() if line.strip()]
        for line in lines:
            words = shlex.split(line)
            wrapped = shlex.split(words[words.index("--wrap") + 1].rstrip().removesuffix(";"))
            assert wrapped[1:4] == ["-m", "moseq2_app", "run-job"], line
            run(cli, "run-job", wrapped[4])
        return lines, received

    return go


def test_notebook_slurm_jobs_read_the_config_file(project, jobs):
    root, config = project
    commands, received = jobs(root, config)

    assert len(commands) == 1
    assert "--config-file" not in commands[0]
    assert len(received) == 1
    assert received[0]["compress_threads"] == JOB_CPUS
    assert tuple(received[0]["crop_size"]) == (90, 90)


def test_notebook_slurm_jobs_read_the_session_config(project, jobs):
    """With a session config from the arena tool, each job reads the settings
    the tool sets from its own entry, and the others from the config file;
    ``compress_threads`` from neither."""
    root, config = project
    session_config = root / "session_config.yaml"
    write_yaml({f"{SESSION}/depth.avi": {"compress_threads": 15, "crop_size": [110, 110]}}, str(session_config))

    commands, received = jobs(root, config)

    assert len(received) == 1
    assert tuple(received[0]["crop_size"]) == (110, 110)
    assert received[0]["compress_threads"] == JOB_CPUS


def test_notebook_slurm_jobs_for_a_recording_missing_from_the_session_config(
    project, jobs
):
    """A recording the session config has no entry for is extracted with the
    config file's values."""
    root, config = project
    session_config = root / "session_config.yaml"
    write_yaml({"another_session/depth.avi": {"crop_size": [110, 110]}}, str(session_config))

    commands, received = jobs(root, config)

    assert len(received) == 1
    assert received[0]["compress_threads"] == JOB_CPUS
    assert tuple(received[0]["crop_size"]) == (90, 90)


#: The arguments of a command line, as a Windows runner's temporary folder makes them.
WINDOWS_CONFIG = r"C:\Users\runneradmin\AppData\Local\Temp\pytest-0\project\session_0\proc\config.yaml"
WINDOWS_RECORDING = r"C:\Users\runneradmin\AppData\Local\Temp\pytest-0\project\session_0\depth.dat"


def test_a_written_command_splits_to_its_windows_paths():
    """On Windows a command line's paths are written with backslashes, which a
    POSIX split reads as escapes and drops."""
    command = f" --config-file {WINDOWS_CONFIG} --num-frames 30 {WINDOWS_RECORDING}"
    assert split_command(command, windows=True) == [
        "--config-file", WINDOWS_CONFIG, "--num-frames", "30", WINDOWS_RECORDING,
    ]


def test_a_quoted_windows_path_loses_only_its_quotes():
    spaced = r"C:\Users\run neradmin\depth.avi"
    assert split_command(f' --num-frames 30 "{spaced}"', windows=True) == [
        "--num-frames", "30", spaced,
    ]


def test_a_written_command_splits_to_its_posix_paths():
    command = " --config-file /tmp/pytest-0/project/config.yaml /tmp/pytest-0/project/depth.avi"
    assert split_command(command, windows=False) == [
        "--config-file", "/tmp/pytest-0/project/config.yaml", "/tmp/pytest-0/project/depth.avi",
    ]
