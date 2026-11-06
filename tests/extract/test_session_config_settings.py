"""What a recording is extracted with when the arena tool has saved an entry for it.

The arena tool saves each recording's entry in the project's
``session_config.yaml``, keyed by session (the recording's path relative to the
project), holding only the settings the tool sets (``ARENA_TOOL_SETTINGS``).
Extraction takes those from the entry and every other setting from
``config.yaml``, so a ``frame_trim``, ``num_frames`` or flip classifier set in
``config.yaml`` after the tool ran reaches every recording (moseq2-extract#158).
Options typed on ``batch-extract``'s command line win over both.

Every path is covered: the notebook's local extraction and ``batch-extract``
(both through ``extract_command``), and the per-recording ``config.yaml`` in its session folder
that ``batch-extract --cluster-type slurm`` and the notebook's SLURM path write.
Nothing is submitted; each SLURM command is run through ``extract`` with
``extract_wrapper`` recorded, so what is checked is what the job would extract with.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_extract.steps as extract_steps
from commands import split_command
from moseq2_common.slurm import job_settings_file
from moseq2_common.yaml import read_yaml, write_yaml

from configs import read_settings
from moseq2_extract.gui import extract_command, extract_found_sessions, generate_config_command
from moseq2_extract.session_config import ARENA_TOOL_SETTINGS
from moseq2_extract.util import detect_and_set_camera_parameters

SESSIONS = ("session_0", "session_1")


def run(*args):
    result = CliRunner().invoke(extract_cli.cli, list(args), catch_exceptions=False)
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


@pytest.fixture
def slurm(monkeypatch):
    """Write the SLURM commands without submitting, run each through ``extract``
    and return what each received, keyed by recording, plus each written
    job ``config.yaml`` in its session folder."""
    monkeypatch.setattr(
        "moseq2_extract.helpers.extract.system",
        lambda *_: pytest.fail("a job was submitted"),
    )
    calls = {}

    def go(root, write_commands):
        write_commands()
        # stood in for only now, so a local extraction in the same test reaches the local fixture's
        monkeypatch.setattr(
            extract_steps,
            "extract_wrapper",
            lambda f, out, config, num_frames=None, skip=False, uuid=None, record=None: calls.update(
                {f: {"config": config, "num_frames": num_frames}}
            ),
        )
        script = (root / "extract_out.sh").read_text()
        for line in filter(str.strip, script.splitlines()):
            command = line.split("moseq2-extract extract", 1)[1].split(";", 1)[0]
            run("extract", *split_command(command))
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


# ------------------------------------------------------------------ R3: extract_command


def test_extract_command_takes_settings_set_in_config_yaml_after_the_tool(project, local):
    """R3/R5: a flip classifier and a frame_trim set in config.yaml after the
    arena tool saved the entry reach the extraction; the crop size and depth
    range tuned in the tool still win over config.yaml's."""
    root, config, session_config = project

    extract_command(recording(root, "session_0"), None, str(config))

    assert_config_yaml_settings_with_tuned_arena(local[recording(root, "session_0")]["config"], root)


def test_a_recording_without_an_entry_takes_config_yaml(project, local):
    root, config, session_config = project
    entries = read_yaml(str(session_config))
    del entries["session_1/depth.dat"]
    write_yaml(entries, str(session_config))

    extract_command(recording(root, "session_1"), None, str(config))

    received = local[recording(root, "session_1")]["config"]
    assert tuple(received["crop_size"]) == (80, 80)
    assert tuple(received["bg_roi_depth_range"]) == (650, 750)
    assert tuple(received["frame_trim"]) == (100, 0)


# ------------------------------------------------------------------ B1: batch-extract, local and SLURM


def test_local_batch_extract_takes_frame_trim_and_num_frames_from_config_yaml(project, local):
    """B1, moseq2-extract#158 as reported: frame_trim and num_frames edited in
    config.yaml reach every recording that has an entry."""
    root, config, session_config = project

    run("batch-extract", str(root), "--config-file", str(config), "--extensions", ".dat")

    assert sorted(local) == [recording(root, name) for name in SESSIONS]
    for name in SESSIONS:
        call = local[recording(root, name)]
        assert call["num_frames"] == 50
        assert_config_yaml_settings_with_tuned_arena(call["config"], root)


def test_slurm_batch_extract_writes_config_yaml_settings_into_each_job(project, slurm):
    """B1 on SLURM: each job's config.yaml holds config.yaml's frame_trim and
    num_frames over the recording's tuned arena settings."""
    root, config, session_config = project

    calls, written = slurm(
        root,
        lambda: run(
            "batch-extract", str(root), "--config-file", str(config),
            "--extensions", ".dat", "--cluster-type", "slurm",
        ),
    )

    for name in SESSIONS:
        assert tuple(written[name]["frame_trim"]) == (100, 0)
        assert written[name]["num_frames"] == 50
        assert tuple(written[name]["crop_size"]) == (100, 100)
        call = calls[recording(root, name)]
        assert call["num_frames"] == 50
        assert_config_yaml_settings_with_tuned_arena(call["config"], root)
    assert "--num-frames 50" in (root / "extract_out.sh").read_text()


def test_notebook_slurm_path_takes_frame_trim_from_config_yaml(project, slurm):
    root, config, session_config = project
    data = read_yaml(str(config))
    data["extract"]["cluster_type"] = "slurm"
    write_yaml(data, str(config))

    calls, written = slurm(root, lambda: extract_found_sessions(str(root), str(config), [".dat"]))

    for name in SESSIONS:
        assert tuple(written[name]["frame_trim"]) == (100, 0)
        assert_config_yaml_settings_with_tuned_arena(calls[recording(root, name)]["config"], root)


# ------------------------------------------------------------------ old files, typed options, grouping


def test_an_entry_key_the_tool_does_not_set_takes_config_yaml_and_is_named(project, local):
    """An entry holding a setting the arena tool does not set (compress_threads, frame_trim, typed in
    by hand) is read as it is, not rewritten: those settings come from config.yaml and
    one line names them; the tool's own settings in it still apply."""
    root, config, session_config = project
    entries = read_yaml(str(session_config))
    entries["session_0/depth.dat"].update(compress_threads=15, frame_trim=[5, 5], noise_tolerance=40)
    write_yaml(entries, str(session_config))
    before = session_config.read_text()

    out = run("batch-extract", str(root), "--config-file", str(config), "--extensions", ".dat").output

    received = local[recording(root, "session_0")]["config"]
    assert received["compress_threads"] == read_yaml(str(config))["extract"]["compress_threads"]
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
    """#421: an option typed on batch-extract's command line wins over config.yaml
    and the entry."""
    root, config, session_config = project
    typed = ("--frame-trim", "7", "7", "--crop-size", "90", "90", "--num-frames", "20")
    args = ("batch-extract", str(root), "--config-file", str(config), "--extensions", ".dat")

    if cluster == "local":
        out = run(*args, *typed).output
        calls = local
    else:
        holder = {}
        calls, _ = slurm(root, lambda: holder.update(out=run(*args, "--cluster-type", "slurm", *typed).output))
        out = holder["out"]

    for name in SESSIONS:
        call = calls[recording(root, name)]
        assert tuple(call["config"]["frame_trim"]) == (7, 7)
        assert tuple(call["config"]["crop_size"]) == (90, 90)
        assert call["num_frames"] == 20
        assert call["config"]["flip_classifier"] == str(root / "flip_classifier.npz")
