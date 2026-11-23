"""Options typed on the command line outrank the config file.

``moseq2 standalone extract FILE --config-file FILE`` merges the options typed
with the config file's. An option typed on the command line wins, including
when the typed value happens to equal the option's default (moseq2-extract#24):
``--crop-size 80 80`` over a config holding ``[120, 120]`` extracts at 80. An
option not typed takes the config file's value where it has one.

``moseq2 extract`` in a project saves the settings typed before the run: an
arena setting typed with recordings named goes to those recordings' entries in
``session_config.yaml``, every other setting to ``config.yaml``. Each recording
is then extracted with the settings the arena tool sets from its entry, and
every other setting from ``config.yaml``, so what was typed reaches the
extraction, and ``--num-frames`` with it (moseq2-extract#158). ``threads`` and
``compress_threads`` are one-run switches that neither file sets: a local run
uses the ``cores`` of ``computing.extract`` for both, a cluster job the 4 CPUs
it asks for.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

import moseq2_extract.steps as extract_steps
from moseq2_app.cli import cli
from moseq2_common.slurm import job_settings_file
from moseq2_common.yaml import read_yaml, write_yaml

#: The fixture's head/tail classifier, so a project extraction downloads nothing.
CLASSIFIER = "flip_classifier_k2_c57_10to13weeks.npz"
#: The cores ``computing.extract`` gives a local run.
CORES = 2
#: The CPUs, and so the threads, a cluster job asks for.
JOB_CPUS = 4


def run(*args, cwd=None):
    """``moseq2 ARGS``, from ``cwd`` as the working folder when given."""
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
def received(monkeypatch):
    """What ``extract`` hands its wrapper, instead of running it."""
    calls = []
    monkeypatch.setattr(
        extract_steps, "extract_wrapper", lambda f, out, config, **kw: calls.append(config)
    )
    return calls


@pytest.mark.parametrize(
    "typed, expected",
    [
        (("--crop-size", "80", "80"), (80, 80)),
        (("--crop-size", "100", "100"), (100, 100)),
        ((), (120, 120)),
    ],
    ids=["typed-default", "typed-other", "not-typed"],
)
def test_extract_crop_size(typed, expected, tmp_path, received):
    recording = tmp_path / "depth.avi"
    recording.write_bytes(b"\0")  # a placeholder: the work functions are stubbed, and an empty input stops a step
    config = tmp_path / "config.yaml"
    write_yaml({"extract": {"crop_size": [120, 120]}}, str(config))

    run("standalone", "extract", str(recording), "--config-file", str(config), *typed)

    assert tuple(received[0]["crop_size"]) == expected


SESSION = "session_1"
KEY = f"{SESSION}/depth.avi"


@pytest.fixture
def project(tmp_path, raw_sessions):
    """A project of one recording, whose config.yaml and session_config.yaml disagree with the defaults and
    with each other. ``crop_size`` and ``manual_set_depth_range`` are settings the arena tool sets, so the
    entry's apply; ``frame_trim`` is not, so config.yaml's applies. Both files give ``compress_threads``,
    which neither sets. Extraction runs locally on ``CORES`` cores, with the fixture's head/tail classifier."""
    session = tmp_path / SESSION
    session.mkdir()
    (session / "depth.avi").write_bytes(b"\0")  # a placeholder: the work functions are stubbed, and an empty input stops a step
    (session / "metadata.json").write_text("{}")
    shutil.copy(Path(raw_sessions) / CLASSIFIER, tmp_path / CLASSIFIER)

    session_config = tmp_path / "session_config.yaml"
    write_yaml(
        {
            KEY: {
                "crop_size": [120, 120],
                "frame_trim": [5, 5],
                "compress_threads": 15,
                "bg_roi_depth_range": [550, 650],
                "manual_set_depth_range": False,
            }
        },
        str(session_config),
    )
    config = tmp_path / "config.yaml"
    write_yaml(
        {
            "extract": {
                "crop_size": [90, 90],
                "frame_trim": [0, 0],
                "compress_threads": 20,
                "flip_classifier": str(tmp_path / CLASSIFIER),
            },
            "computing": {"extract": {"cluster_type": "local", "cores": CORES, "wall_time": "1:00:00"}},
        },
        str(config),
    )
    return tmp_path, config


@pytest.fixture
def extracted(monkeypatch):
    """What each extraction hands ``extract_wrapper``, instead of running it."""
    calls = []

    def record(input_file, output_dir, config_data, num_frames=None, skip=False, uuid=None, record=None):
        calls.append(
            {"config": config_data, "num_frames": num_frames, "output_dir": output_dir}
        )

    monkeypatch.setattr(extract_steps, "extract_wrapper", record)
    return calls


def batch(root, *typed):
    """``moseq2 extract`` in the project ``root``, with ``typed`` (flags, and recordings named)."""
    return run("extract", "--extensions", ".avi", *typed, cwd=root)


def test_batch_extract_typed_options_reach_each_recording(project, extracted):
    root, config = project
    batch(
        root,
        "--num-frames",
        "50",
        "--frame-trim",
        "10",
        "10",
        "--crop-size",
        "80",
        "80",
        KEY,
    )

    assert len(extracted) == 1
    call = extracted[0]
    assert call["num_frames"] == 50
    assert tuple(call["config"]["frame_trim"]) == (10, 10)
    assert tuple(call["config"]["crop_size"]) == (80, 80)
    # saved where they belong: the arena setting in the recording's entry, the others in config.yaml
    assert read_yaml(str(root / "session_config.yaml"))[KEY]["crop_size"] == [80, 80]
    saved = read_yaml(str(config))["extract"]
    assert (saved["num_frames"], saved["frame_trim"], saved["crop_size"]) == (50, [10, 10], [90, 90])
    # a one-run switch neither file sets: the run's cores
    assert (call["config"]["threads"], call["config"]["compress_threads"]) == (CORES, CORES)


def test_batch_extract_without_typed_options_uses_the_session_config_for_the_arena_tool_settings(
    project, extracted
):
    root, _ = project
    batch(root)

    call = extracted[0]
    assert call["num_frames"] is None
    assert tuple(call["config"]["crop_size"]) == (120, 120)
    assert call["config"]["manual_set_depth_range"] is False
    assert tuple(call["config"]["frame_trim"]) == (0, 0)
    assert call["config"]["compress_threads"] == CORES


def test_batch_extract_typed_depth_range_is_manual(project, extracted):
    root, _ = project
    batch(root, "--bg-roi-depth-range", "600", "800", KEY)

    call = extracted[0]
    assert tuple(call["config"]["bg_roi_depth_range"]) == (600, 800)
    assert call["config"]["manual_set_depth_range"] is True


# ``moseq2 extract`` on SLURM (``computing.extract.cluster_type: slurm``) writes
# each recording's job settings file in its session folder and one ``sbatch``
# line per recording into ``extract_out.sh`` instead of extracting. The options
# typed on its command line reach each of those extractions with the same
# precedence as the local path (moseq2-extract#158). Each test runs the written
# command, ``moseq2 run-job JOB_FILE``, with ``extract_wrapper`` recorded instead
# of run, so what is checked is what the job would extract with.


def jobs_in(script: Path) -> list[str]:
    """The job settings file each ``sbatch`` line of ``script`` runs ``moseq2 run-job`` on."""
    files = []
    for line in script.read_text().splitlines():
        if not line.strip():
            continue
        words = shlex.split(line)
        wrapped = shlex.split(words[words.index("--wrap") + 1].rstrip().removesuffix(";"))
        assert wrapped[1:4] == ["-m", "moseq2_app", "run-job"], line
        files.append(wrapped[4])
    return files


@pytest.fixture
def slurm_extracted(monkeypatch, received):
    """Run ``moseq2 extract`` on SLURM, which submits nothing, then run each written job through
    ``moseq2 run-job`` and return what it received: one dict per recording, the config and ``num_frames``."""
    nums = []
    monkeypatch.setattr(
        extract_steps,
        "extract_wrapper",
        lambda f, out, config, num_frames=None, skip=False, uuid=None, record=None: (
            received.append(config) or nums.append(num_frames)
        ),
    )

    def go(root, *typed):
        data = read_yaml(str(root / "config.yaml"))
        data["computing"]["extract"]["cluster_type"] = "slurm"
        write_yaml(data, str(root / "config.yaml"))
        batch(root, *typed)
        assert received == [], "moseq2 extract on SLURM extracted a recording itself"
        for job_file in jobs_in(root / "extract_out.sh"):
            run("run-job", job_file)
        return [
            {"config": c, "num_frames": n} for c, n in zip(received, nums)
        ]

    return go


def test_slurm_batch_extract_typed_options_reach_each_recording(project, slurm_extracted):
    root, _ = project
    calls = slurm_extracted(
        root,
        "--num-frames",
        "50",
        "--frame-trim",
        "10",
        "10",
        "--crop-size",
        "80",
        "80",
        KEY,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["num_frames"] == 50
    assert tuple(call["config"]["frame_trim"]) == (10, 10)
    assert tuple(call["config"]["crop_size"]) == (80, 80)
    # a one-run switch neither file sets: the CPUs the job asks for
    assert (call["config"]["threads"], call["config"]["compress_threads"]) == (JOB_CPUS, JOB_CPUS)


def test_slurm_batch_extract_without_typed_options_uses_the_session_config_for_the_arena_tool_settings(
    project, slurm_extracted
):
    root, _ = project
    call = slurm_extracted(root)[0]

    assert call["num_frames"] is None
    assert tuple(call["config"]["crop_size"]) == (120, 120)
    assert call["config"]["manual_set_depth_range"] is False
    assert tuple(call["config"]["frame_trim"]) == (0, 0)
    assert call["config"]["compress_threads"] == JOB_CPUS


def test_slurm_batch_extract_typed_depth_range_is_manual(project, slurm_extracted):
    root, _ = project
    call = slurm_extracted(root, "--bg-roi-depth-range", "600", "800", KEY)[0]

    assert tuple(call["config"]["bg_roi_depth_range"]) == (600, 800)
    assert call["config"]["manual_set_depth_range"] is True


def test_slurm_batch_extract_typed_options_without_a_session_config(
    project, slurm_extracted
):
    """With no session config, each recording is extracted from the config file,
    and the typed options still apply over it."""
    root, _ = project
    (root / "session_config.yaml").unlink()

    call = slurm_extracted(
        root, "--num-frames", "50", "--crop-size", "80", "80"
    )[0]

    assert call["num_frames"] == 50
    assert tuple(call["config"]["crop_size"]) == (80, 80)
    # not a setting either file sets: the CPUs the job asks for
    assert call["config"]["compress_threads"] == JOB_CPUS


def test_slurm_batch_extract_without_typed_options_or_session_config(
    project, slurm_extracted
):
    root, _ = project
    (root / "session_config.yaml").unlink()

    call = slurm_extracted(root)[0]

    assert call["num_frames"] is None
    assert tuple(call["config"]["crop_size"]) == (90, 90)
    assert call["config"]["compress_threads"] == JOB_CPUS


def test_slurm_job_config_holds_the_entry_and_names_no_session_config(project, slurm_extracted):
    """Each job's settings file already holds the recording's entry, so the job does not lay the
    project's session_config.yaml over the typed options again."""
    root, _ = project
    slurm_extracted(root, "--crop-size", "80", "80", KEY)
    job_file = read_yaml(str(job_settings_file(root / "aggregate_results" / SESSION, "extract")))
    job_config = job_file["extract"]

    assert "session_config_path" not in job_file
    assert tuple(job_config["crop_size"]) == (80, 80)
    assert job_config["manual_set_depth_range"] is False


# ``moseq2 extract RECORDING`` in a project resolves the recording's entry in
# session_config.yaml by the same rule as a run of every recording and the
# notebook: the arena tool's settings from the entry, every other setting from
# the config file, and typed options saved first.


def test_extract_file_uses_the_recordings_session_settings(project, received):
    root, _ = project
    run("extract", KEY, cwd=root)

    settings = received[0]
    assert tuple(settings["crop_size"]) == (120, 120)
    assert tuple(settings["bg_roi_depth_range"]) == (550, 650)
    assert settings["manual_set_depth_range"] is False
    # not settings the arena tool sets: the config file's values
    assert tuple(settings["frame_trim"]) == (0, 0)
    # a one-run switch neither file sets: the run's cores
    assert settings["compress_threads"] == CORES


def test_extract_file_typed_options_win_over_the_session_settings(project, received):
    root, _ = project
    run("extract", KEY, "--crop-size", "80", "80", "--bg-roi-depth-range", "600", "800", cwd=root)

    settings = received[0]
    assert tuple(settings["crop_size"]) == (80, 80)
    assert tuple(settings["bg_roi_depth_range"]) == (600, 800)
    assert settings["manual_set_depth_range"] is True


def test_extract_file_without_a_session_entry_uses_the_config_file(project, received):
    root, _ = project
    other = root / "session_2"
    other.mkdir()
    (other / "depth.avi").write_bytes(b"\0")  # a placeholder: the work functions are stubbed, and an empty input stops a step
    run("extract", "session_2/depth.avi", cwd=root)

    settings = received[0]
    assert tuple(settings["crop_size"]) == (90, 90)
    assert settings["compress_threads"] == CORES


def test_threads_help_says_it_is_for_reading_the_recording():
    """On ``moseq2 standalone extract`` ``--threads`` reads the recording; ``--compress-threads`` compresses it.
    ``moseq2 extract`` takes neither: a local run uses its ``--cores`` for both."""
    standalone = {p.name: p for p in cli.commands["standalone"].commands["extract"].params}
    assert "read" in standalone["threads"].help
    assert "compress" not in standalone["threads"].help
    assert "compress" in standalone["compress_threads"].help

    in_project = {p.name for p in cli.commands["extract"].params}
    assert "cores" in in_project
    assert not {"threads", "compress_threads"} & in_project
