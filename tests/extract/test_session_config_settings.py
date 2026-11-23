"""What a recording is extracted with when the arena tool has saved an entry for it.

The arena tool saves each recording's entry in the project's
``session_config.yaml``, keyed by session (the recording's path relative to the
project), holding only the settings the tool sets (``ARENA_TOOL_SETTINGS``).
Extraction takes those from the entry and every other setting from
``config.yaml``, so a ``frame_trim``, ``num_frames`` or flip classifier set in
``config.yaml`` after the tool ran reaches every recording (moseq2-extract#158).
Options typed on ``moseq2 extract``'s command line are saved first, an arena
setting typed with recordings named to their entries, so they win over both.
``compress_threads`` is a one-run switch: neither file sets it, and a local run
uses its ``cores``.

Every path is covered: the project's extraction of named recordings
(``moseq2_extract.project.extract``), ``moseq2 extract`` run locally, and the
job settings file in each session folder that ``moseq2 extract`` and the
notebook's Extract step write on SLURM. Nothing is submitted; each job is run
through ``moseq2 run-job`` with ``extract_wrapper`` recorded, so what is checked
is what the job would extract with.
"""

from __future__ import annotations

import os
from copy import deepcopy

import pytest
from click.testing import CliRunner

import moseq2_extract.steps as extract_steps
from moseq2_app.cli import cli
from moseq2_common.project import Project
from moseq2_common.slurm import job_settings_file
from moseq2_common.yaml import read_yaml, write_yaml

from configs import read_settings
from slurm_fakes import fake_slurm, job_file_of, submitted_lines  # noqa: F401 -- the fixture
from moseq2_extract import project as extraction
from moseq2_extract.gui import generate_config_command
from moseq2_extract.session_config import ARENA_TOOL_SETTINGS
from moseq2_extract.util import detect_and_set_camera_parameters

SESSIONS = ("session_0", "session_1")
#: The cores ``computing.extract`` gives a local run, which reads and compresses with them.
CORES = 2


def run(*args, cwd):
    """``moseq2 ARGS`` with ``cwd`` as the working folder; its result."""
    before = os.getcwd()
    os.chdir(cwd)
    try:
        result = CliRunner().invoke(cli, [str(a) for a in args], catch_exceptions=False)
    finally:
        os.chdir(before)
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture
def project(tmp_path):
    """Two recordings, a config from ``generate-config`` and a
    ``session_config.yaml`` as the arena tool writes it: per recording the tool's
    settings, with detected camera settings and a tuned crop size and depth
    range. ``config.yaml`` is then edited the way the reporter of
    moseq2-extract#158 and the notebook's flip-classifier cells edit it."""
    root = tmp_path / "project"
    for name in SESSIONS:
        (root / name).mkdir(parents=True)
        (root / name / "depth.dat").write_bytes(b"\0")  # a placeholder: the work functions are stubbed, and an empty input stops a step
        (root / name / "metadata.json").write_text("{}")

    config = root / "config.yaml"
    generate_config_command(str(config), camera_type="k2")
    data = read_yaml(str(config))
    extract = data["extract"]

    entries = {}
    for name in SESSIONS:
        entry = detect_and_set_camera_parameters({**deepcopy(extract), "camera_type": data["camera_type"]},
                                                 str(root / name / "depth.dat"))
        entry["crop_size"] = [100, 100]
        entry["bg_roi_depth_range"] = [600, 700]
        entry["manual_set_depth_range"] = True
        entries[f"{name}/depth.dat"] = {k: v for k, v in entry.items() if k in ARENA_TOOL_SETTINGS}
    session_config = root / "session_config.yaml"
    write_yaml(entries, str(session_config))

    extract["frame_trim"] = [100, 0]
    extract["num_frames"] = 50
    extract["flip_classifier"] = str(root / "flip_classifier.npz")
    extract["crop_size"] = [80, 80]
    extract["bg_roi_depth_range"] = [650, 750]
    data["computing"]["extract"]["cores"] = CORES
    write_yaml(data, str(config))
    return root, config, session_config


@pytest.fixture
def local(monkeypatch):
    """What each local extraction hands ``extract_wrapper``, instead of running it."""
    calls = {}

    def record(input_file, output_dir, config_data, num_frames=None, skip=False, uuid=None, record=None):
        calls[input_file] = {"config": config_data, "num_frames": num_frames}
        return output_dir

    monkeypatch.setattr(extract_steps, "extract_wrapper", record)
    return calls


def job_files(root):
    """The job settings file each job submitted in the project in ``root`` runs ``moseq2 run-job`` on."""
    return [job_file_of(line) for line in submitted_lines(Project.at(root))]


@pytest.fixture
def slurm(monkeypatch, request):
    """Submit the SLURM jobs (to the stand-ins in ``tests/fake_slurm``, which hold them in the queue), run each
    through ``moseq2 run-job`` and return what each received, keyed by recording, plus each written
    job settings file in its session folder."""
    calls = {}

    def go(root, write_commands):
        # asked for only here, so a test that runs locally too needs no fake cluster
        request.getfixturevalue("fake_slurm").hold()
        write_commands()
        # stood in for only now, so a local extraction in the same test reaches the local fixture's
        monkeypatch.setattr(
            extract_steps,
            "extract_wrapper",
            lambda f, out, config, num_frames=None, skip=False, uuid=None, record=None: calls.update(
                {f: {"config": config, "num_frames": num_frames}}
            ),
        )
        files = job_files(root)
        assert sorted(files) == sorted(str(job_settings_file(root / "aggregate_results" / name, "extract"))
                                       for name in SESSIONS)
        for job_file in files:
            run("run-job", job_file, cwd=root)
        written = {name: read_settings(job_settings_file(root / "aggregate_results" / name, "extract")) for name in SESSIONS}
        return calls, written

    return go


def recording(root, name):
    return str(root / name / "depth.dat")


def assert_config_yaml_settings_with_tuned_arena(config, root):
    """Settings outside the arena tool's come from config.yaml; the tool's from the entry."""
    assert tuple(config["frame_trim"]) == (100, 0)
    assert config["flip_classifier"] == str(root / "flip_classifier.npz")
    assert tuple(config["crop_size"]) == (100, 100)
    assert tuple(config["bg_roi_depth_range"]) == (600, 700)
    assert config["manual_set_depth_range"] is True
    # detected for the recording by the arena tool, kept from the entry
    assert config["pixel_format"] == "gray16le"


# ------------------------------------------------------------------ R3: one recording named


def test_extract_command_takes_settings_set_in_config_yaml_after_the_tool(project, local):
    """R3/R5: a flip classifier and a frame_trim set in config.yaml after the
    arena tool saved the entry reach the extraction; the crop size and depth
    range tuned in the tool still win over config.yaml's."""
    root, _, _ = project

    extraction.extract(Project.at(root), recordings=["session_0/depth.dat"])

    assert sorted(local) == [recording(root, "session_0")]

    assert_config_yaml_settings_with_tuned_arena(local[recording(root, "session_0")]["config"], root)


def test_a_recording_without_an_entry_takes_config_yaml(project, local):
    root, _, session_config = project
    entries = read_yaml(str(session_config))
    del entries["session_1/depth.dat"]
    write_yaml(entries, str(session_config))

    extraction.extract(Project.at(root), recordings=["session_1/depth.dat"])

    received = local[recording(root, "session_1")]["config"]
    assert tuple(received["crop_size"]) == (80, 80)
    assert tuple(received["bg_roi_depth_range"]) == (650, 750)
    assert tuple(received["frame_trim"]) == (100, 0)


# ------------------------------------------------------------------ B1: moseq2 extract, local and SLURM


def test_local_batch_extract_takes_frame_trim_and_num_frames_from_config_yaml(project, local):
    """B1, moseq2-extract#158 as reported: frame_trim and num_frames edited in
    config.yaml reach every recording that has an entry."""
    root, _, _ = project

    run("extract", "--extensions", ".dat", cwd=root)

    assert sorted(local) == [recording(root, name) for name in SESSIONS]
    for name in SESSIONS:
        call = local[recording(root, name)]
        assert call["num_frames"] == 50
        assert_config_yaml_settings_with_tuned_arena(call["config"], root)


def test_slurm_batch_extract_writes_config_yaml_settings_into_each_job(project, slurm):
    """B1 on SLURM: each job's settings file holds config.yaml's frame_trim and
    num_frames over the recording's tuned arena settings."""
    root, _, _ = project

    calls, written = slurm(
        root,
        lambda: run("extract", "--extensions", ".dat", "--cluster-type", "slurm", cwd=root),
    )

    for name in SESSIONS:
        assert tuple(written[name]["frame_trim"]) == (100, 0)
        assert written[name]["num_frames"] == 50
        assert tuple(written[name]["crop_size"]) == (100, 100)
        call = calls[recording(root, name)]
        assert call["num_frames"] == 50
        assert_config_yaml_settings_with_tuned_arena(call["config"], root)
    # the settings travel in each job's settings file, not on its command line
    assert not any("--num-frames" in line for line in submitted_lines(Project.at(root)))


def test_notebook_slurm_path_takes_frame_trim_from_config_yaml(project, slurm):
    root, config, _ = project
    data = read_yaml(str(config))
    data["computing"]["extract"]["cluster_type"] = "slurm"
    write_yaml(data, str(config))

    def notebook():
        from moseq2_app.notebook import extract

        before = os.getcwd()
        os.chdir(root)
        try:
            widget = extract()
            widget.start()
            ran = widget.wait()
            assert (ran["state"], ran["where"]) == ("running", "the cluster"), widget.problem
        finally:
            os.chdir(before)

    calls, written = slurm(root, notebook)

    for name in SESSIONS:
        assert tuple(written[name]["frame_trim"]) == (100, 0)
        assert_config_yaml_settings_with_tuned_arena(calls[recording(root, name)]["config"], root)


# ------------------------------------------------------------------ old files, typed options, grouping


def test_an_entry_key_the_tool_does_not_set_takes_config_yaml_and_is_named(project, local):
    """An entry holding a setting the arena tool does not set (compress_threads, frame_trim, typed in
    by hand) is read as it is, not rewritten: those settings do not come from the entry, and
    one line names them; the tool's own settings in it still apply."""
    root, _, session_config = project
    entries = read_yaml(str(session_config))
    entries["session_0/depth.dat"].update(compress_threads=15, frame_trim=[5, 5], noise_tolerance=40)
    write_yaml(entries, str(session_config))
    before = session_config.read_text()

    out = run("extract", "--extensions", ".dat", cwd=root).stderr

    received = local[recording(root, "session_0")]["config"]
    assert received["compress_threads"] == CORES
    assert tuple(received["frame_trim"]) == (100, 0)
    assert received["noise_tolerance"] == 40
    assert session_config.read_text() == before
    message = (
        f"{session_config}'s entry for session_0/depth.dat sets `compress_threads`, `frame_trim`, which the arena tool does "
        "not set, so extraction ignores them and takes them from config.yaml's extract section. Delete those "
        "lines from the entry to silence this message."
    )
    assert out.count(message) == 1


@pytest.mark.parametrize("cluster", ["local", "slurm"])
def test_typed_options_win_over_both(project, local, slurm, cluster):
    """#421: an option typed on ``moseq2 extract``'s command line wins over config.yaml
    and the entry: it is saved first, an arena setting typed with the recordings named to their entries."""
    root, _, _ = project
    typed = ("--frame-trim", "7", "7", "--crop-size", "90", "90", "--num-frames", "20",
             *(f"{name}/depth.dat" for name in SESSIONS))
    args = ("extract", "--extensions", ".dat")

    if cluster == "local":
        run(*args, *typed, cwd=root)
        calls = local
    else:
        calls, _ = slurm(root, lambda: run(*args, "--cluster-type", "slurm", *typed, cwd=root))

    for name in SESSIONS:
        call = calls[recording(root, name)]
        assert tuple(call["config"]["frame_trim"]) == (7, 7)
        assert tuple(call["config"]["crop_size"]) == (90, 90)
        assert call["num_frames"] == 20
        assert call["config"]["flip_classifier"] == str(root / "flip_classifier.npz")
