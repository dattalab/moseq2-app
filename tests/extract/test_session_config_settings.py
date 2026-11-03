"""What a recording is extracted with when the arena tool has saved an entry for it.

The arena tool writes ``session_config.yaml`` with one entry per recording, a
copy of ``config.yaml`` taken when the tool first listed the recording plus the
values tuned in the tool. Extraction takes from that entry only the settings the
tool sets (``ARENA_TOOL_SETTINGS``) and every other setting from ``config.yaml``,
so a ``frame_trim``, ``num_frames`` or flip classifier set in ``config.yaml``
after the tool ran reaches every recording (moseq2-extract#158). Options typed
on ``batch-extract``'s command line win over both. Where an entry holds a
different value for a setting outside the list, one message per setting names
the recordings, both values and their files, which was used and how to change it.

Every path is covered: the notebook's local extraction and ``batch-extract``
(both through ``extract_command``), and the per-recording ``proc/config.yaml``
that ``batch-extract --cluster-type slurm`` and the notebook's SLURM path write.
Nothing is submitted; each SLURM command is run through ``extract`` with
``extract_wrapper`` recorded, so what is checked is what the job would extract with.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_extract.gui as extract_gui
from commands import split_command
from moseq2_common.yaml import read_yaml, write_yaml
from moseq2_extract.gui import extract_command, extract_found_sessions, generate_config_command
from moseq2_extract.util import detect_and_set_camera_parameters

SESSIONS = ("session_0", "session_1")


def run(*args):
    result = CliRunner().invoke(extract_cli.cli, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


@pytest.fixture
def project(tmp_path):
    """Two recordings, a config from ``generate-config`` and a
    ``session_config.yaml`` as the arena tool writes it: per recording a full
    copy of the config with detected camera settings and a tuned crop size and
    depth range. ``config.yaml`` is then edited the way the reporter of
    moseq2-extract#158 and the notebook's flip-classifier cells edit it."""
    root = tmp_path / "project"
    for name in SESSIONS:
        (root / name).mkdir(parents=True)
        (root / name / "depth.dat").touch()
        (root / name / "metadata.json").write_text("{}")

    config = root / "config.yaml"
    generate_config_command(str(config), camera_type="k2")
    data = read_yaml(str(config))
    data["output_dir"] = "proc"

    entries = {}
    for name in SESSIONS:
        entry = detect_and_set_camera_parameters(deepcopy(data), str(root / name / "depth.dat"))
        entry["crop_size"] = [100, 100]
        entry["bg_roi_depth_range"] = [600, 700]
        entry["manual_set_depth_range"] = True
        entries[name] = entry
    session_config = root / "session_config.yaml"
    write_yaml(entries, str(session_config))

    data["session_config_path"] = str(session_config)
    data["frame_trim"] = [100, 0]
    data["num_frames"] = 50
    data["flip_classifier"] = str(root / "flip_classifier.npz")
    data["crop_size"] = [80, 80]
    data["bg_roi_depth_range"] = [650, 750]
    write_yaml(data, str(config))
    return root, config, session_config


@pytest.fixture
def local(monkeypatch):
    """What each local extraction hands ``extract_wrapper``, instead of running it."""
    calls = {}

    def record(input_file, output_dir, config_data, num_frames=None, skip=False):
        calls[input_file] = {"config": config_data, "num_frames": num_frames}
        return output_dir

    monkeypatch.setattr(extract_gui, "extract_wrapper", record)
    return calls


@pytest.fixture
def slurm(monkeypatch):
    """Write the SLURM commands without submitting, run each through ``extract``
    and return what each received, keyed by recording, plus each written
    ``proc/config.yaml``."""
    monkeypatch.setattr(
        "moseq2_extract.helpers.extract.system",
        lambda *_: pytest.fail("a job was submitted"),
    )
    calls = {}
    monkeypatch.setattr(
        extract_cli,
        "extract_wrapper",
        lambda f, out, config, num_frames=None, skip=False: calls.update(
            {f: {"config": config, "num_frames": num_frames}}
        ),
    )

    def go(root, write_commands):
        write_commands()
        script = (root / "extract_out.sh").read_text()
        for line in filter(str.strip, script.splitlines()):
            command = line.split("moseq2-extract extract", 1)[1].split(";", 1)[0]
            run("extract", *split_command(command))
        written = {name: read_yaml(str(root / name / "proc" / "config.yaml")) for name in SESSIONS}
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


def expected_message(root, config, session_config):
    """The message for the ``project`` fixture: the two recordings differ from
    config.yaml in flip_classifier and frame_trim, the same way. num_frames
    differs too but is not named: it is passed to the extraction on its own
    (from config.yaml or the command line) and never came from an entry."""
    names = ", ".join(SESSIONS)
    return (
        f"{session_config} holds values for these recordings that differ from {config}. "
        "Extraction takes them from config.yaml: session_config.yaml supplies only the "
        "settings the arena tool sets (arena detection, mouse height, crop size, the "
        "cleaning filters, noise tolerance, the tracking model, flip smoothing, "
        "compression, chunk overlap and the camera formats).\n"
        f'  flip_classifier: config.yaml\'s "{root / "flip_classifier.npz"}" is used. '
        f"session_config.yaml holds null for {names}. "
        "To change flip_classifier, set it in config.yaml, or type --flip-classifier "
        "on batch-extract's command line.\n"
        "  frame_trim: config.yaml's [100, 0] is used. "
        f"session_config.yaml holds [0, 0] for {names}. "
        "To change frame_trim, set it in config.yaml, or type --frame-trim "
        "on batch-extract's command line."
    )


# ------------------------------------------------------------------ R3: extract_command


def test_extract_command_takes_settings_set_in_config_yaml_after_the_tool(project, local, capsys):
    """R3/R5: a flip classifier and a frame_trim set in config.yaml after the
    arena tool saved the entry reach the extraction; the crop size and depth
    range tuned in the tool still win over config.yaml's."""
    root, config, session_config = project

    extract_command(recording(root, "session_0"), None, str(config))

    assert_config_yaml_settings_with_tuned_arena(local[recording(root, "session_0")]["config"], root)
    out = capsys.readouterr().out
    assert "flip_classifier: config.yaml's" in out
    assert "frame_trim: config.yaml's [100, 0] is used" in out
    assert "session_config.yaml holds [0, 0] for session_0." in out
    # the tool's own settings are not reported
    assert "crop_size" not in out
    assert "bg_roi_depth_range" not in out


def test_a_recording_without_an_entry_takes_config_yaml(project, local, capsys):
    root, config, session_config = project
    entries = read_yaml(str(session_config))
    del entries["session_1"]
    write_yaml(entries, str(session_config))

    extract_command(recording(root, "session_1"), None, str(config))

    received = local[recording(root, "session_1")]["config"]
    assert tuple(received["crop_size"]) == (80, 80)
    assert tuple(received["bg_roi_depth_range"]) == (650, 750)
    assert tuple(received["frame_trim"]) == (100, 0)
    assert "differ from" not in capsys.readouterr().out


def test_no_message_when_the_entry_agrees_with_config_yaml(project, local, capsys):
    root, config, session_config = project
    data = read_yaml(str(config))
    entries = read_yaml(str(session_config))
    for entry in entries.values():
        for key in ("frame_trim", "num_frames", "flip_classifier"):
            entry[key] = data[key]
    write_yaml(entries, str(session_config))

    extract_command(recording(root, "session_0"), None, str(config))

    assert "differ from" not in capsys.readouterr().out


# ------------------------------------------------------------------ B1: batch-extract, local and SLURM


def test_local_batch_extract_takes_frame_trim_and_num_frames_from_config_yaml(project, local, capsys):
    """B1, moseq2-extract#158 as reported: frame_trim and num_frames edited in
    config.yaml reach every recording that has an entry; the differing settings
    are reported once, grouped, for both recordings."""
    root, config, session_config = project
    capsys.readouterr()

    run("batch-extract", str(root), "--config-file", str(config), "--extensions", ".dat")

    assert sorted(local) == [recording(root, name) for name in SESSIONS]
    for name in SESSIONS:
        call = local[recording(root, name)]
        assert call["num_frames"] == 50
        assert_config_yaml_settings_with_tuned_arena(call["config"], root)


def test_local_batch_extract_reports_each_setting_once(project, local):
    root, config, session_config = project

    out = run("batch-extract", str(root), "--config-file", str(config), "--extensions", ".dat").output

    assert expected_message(root, config, session_config) in out
    assert out.count("frame_trim: config.yaml's") == 1


def test_slurm_batch_extract_writes_config_yaml_settings_into_each_job(project, slurm):
    """B1 on SLURM: each job's proc/config.yaml holds config.yaml's frame_trim and
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


def test_slurm_batch_extract_reports_each_setting_once(project, monkeypatch):
    monkeypatch.setattr("moseq2_extract.helpers.extract.system", lambda *_: pytest.fail("submitted"))
    root, config, session_config = project

    out = run(
        "batch-extract", str(root), "--config-file", str(config),
        "--extensions", ".dat", "--cluster-type", "slurm",
    ).output

    assert expected_message(root, config, session_config) in out


def test_notebook_slurm_path_takes_frame_trim_from_config_yaml(project, slurm):
    root, config, session_config = project
    data = read_yaml(str(config))
    data["cluster_type"] = "slurm"
    write_yaml(data, str(config))

    calls, written = slurm(root, lambda: extract_found_sessions(str(root), str(config), [".dat"]))

    for name in SESSIONS:
        assert tuple(written[name]["frame_trim"]) == (100, 0)
        assert_config_yaml_settings_with_tuned_arena(calls[recording(root, name)]["config"], root)


# ------------------------------------------------------------------ old files, typed options, grouping


def test_an_old_full_copy_entry_resolves_by_the_same_rule(project, local):
    """A session_config.yaml an existing project already has: a full copy of an
    older config.yaml, with settings the tool never wrote (fps, frame_trim) set
    by hand. It is read as it is, not rewritten, and the same rule applies."""
    root, config, session_config = project
    entries = read_yaml(str(session_config))
    entries["session_0"].update(fps=15, frame_trim=[5, 5], noise_tolerance=40)
    write_yaml(entries, str(session_config))
    before = session_config.read_text()

    run("batch-extract", str(root), "--config-file", str(config), "--extensions", ".dat")

    received = local[recording(root, "session_0")]["config"]
    assert received["fps"] == read_yaml(str(config))["fps"]
    assert tuple(received["frame_trim"]) == (100, 0)
    assert received["noise_tolerance"] == 40
    assert session_config.read_text() == before


def test_recordings_with_different_entry_values_are_grouped_by_value(project, local):
    root, config, session_config = project
    entries = read_yaml(str(session_config))
    entries["session_1"]["frame_trim"] = [5, 5]
    write_yaml(entries, str(session_config))

    out = run("batch-extract", str(root), "--config-file", str(config), "--extensions", ".dat").output

    assert (
        "  frame_trim: config.yaml's [100, 0] is used. session_config.yaml holds "
        "[0, 0] for session_0 and [5, 5] for session_1. To change frame_trim"
    ) in out


@pytest.mark.parametrize("cluster", ["local", "slurm"])
def test_typed_options_win_over_both(project, local, slurm, cluster):
    """#421: an option typed on batch-extract's command line wins over config.yaml
    and the entry, and a setting typed is not reported as differing."""
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
    assert "frame_trim:" not in out
    assert "flip_classifier: config.yaml's" in out
