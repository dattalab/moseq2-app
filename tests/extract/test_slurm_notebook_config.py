"""The extraction notebook's SLURM path extracts with the notebook's config file.

With ``cluster_type: slurm`` in the config, the notebook's extraction cell
(``extract_found_sessions``) writes one ``moseq2-extract extract`` command per
recording into ``extract_out.sh`` instead of extracting. Each of those jobs has
to read the config file the notebook was given, whether or not the arena tool
wrote a session config. A config written by ``generate-config`` holds
``config_file: null``, which must not reach the commands as ``--config-file
None``.

Nothing is submitted: the submission is patched to fail the test, and each
written command is run through ``extract`` with ``extract_wrapper`` recorded,
so what is checked is what the job would extract with.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_extract.steps as extract_steps
from commands import split_command
from moseq2_common.yaml import read_yaml, write_yaml
from moseq2_extract.gui import extract_found_sessions

SESSION = "session_1"


def run(*args):
    result = CliRunner().invoke(extract_cli.cli, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture
def project(tmp_path):
    """One recording and a config from ``generate-config``, set to SLURM and
    with a value changed from its default."""
    session = tmp_path / SESSION
    session.mkdir()
    (session / "depth.avi").write_bytes(b"\0")  # a placeholder: the work functions are stubbed, and an empty input stops a step
    (session / "metadata.json").write_text("{}")

    config = tmp_path / "config.yaml"
    run("generate-config", "-o", str(config))
    data = read_yaml(str(config))
    # paths are not settings: config.yaml names no config file of its own
    assert "config_file" not in data["extract"]
    data["extract"].update(cluster_type="slurm", compress_threads=20, crop_size=[90, 90])
    write_yaml(data, str(config))
    return tmp_path, config


@pytest.fixture
def jobs(monkeypatch):
    """Run the notebook's SLURM extraction without submitting, then run each
    written command through ``extract`` and return the config each received."""
    monkeypatch.setattr(
        "moseq2_extract.helpers.extract.system",
        lambda *_: pytest.fail("extract_found_sessions submitted a job"),
    )
    received = []
    monkeypatch.setattr(
        extract_steps,
        "extract_wrapper",
        lambda f, out, config, num_frames=None, skip=False, uuid=None, record=None: received.append(config),
    )

    def go(root, config):
        extract_found_sessions(str(root), str(config), ".avi")
        script = (root / "extract_out.sh").read_text()
        commands = [
            line.split("moseq2-extract extract", 1)[1].split(";", 1)[0]
            for line in script.splitlines()
            if line.strip()
        ]
        for command in commands:
            run("extract", *split_command(command))
        return commands, received

    return go


def test_notebook_slurm_jobs_read_the_config_file(project, jobs):
    root, config = project
    commands, received = jobs(root, config)

    assert len(commands) == 1
    assert "--config-file None" not in commands[0]
    assert len(received) == 1
    assert received[0]["compress_threads"] == 20
    assert tuple(received[0]["crop_size"]) == (90, 90)


def test_notebook_slurm_jobs_read_the_session_config(project, jobs):
    """With a session config from the arena tool, each job reads the settings
    the tool sets from its own entry, and the others from the config file."""
    root, config = project
    session_config = root / "session_config.yaml"
    write_yaml({f"{SESSION}/depth.avi": {"compress_threads": 15, "crop_size": [110, 110]}}, str(session_config))

    commands, received = jobs(root, config)

    assert len(received) == 1
    assert tuple(received[0]["crop_size"]) == (110, 110)
    assert received[0]["compress_threads"] == 20


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
    assert received[0]["compress_threads"] == 20
    assert tuple(received[0]["crop_size"]) == (90, 90)


#: A written command after ``moseq2-extract extract``, as a Windows runner's
#: temporary folder makes it.
WINDOWS_CONFIG = r"C:\Users\runneradmin\AppData\Local\Temp\pytest-0\project\session_0\proc\config.yaml"
WINDOWS_RECORDING = r"C:\Users\runneradmin\AppData\Local\Temp\pytest-0\project\session_0\depth.dat"


def test_a_written_command_splits_to_its_windows_paths():
    """The tests above split each written command before running it, and on
    Windows its paths are written with backslashes, which a POSIX split
    reads as escapes and drops."""
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
