"""``session_config.yaml``: a fixed name in the project folder, keyed by session.

Each recording's arena-tool settings live in ``session_config.yaml`` in the
project folder, found without a pointer in ``config.yaml``. An entry is keyed by
its session, the recording's path relative to the project folder
(``saline_example_0/depth.avi``), so a project copied or moved elsewhere keeps
its entries. An entry holds only the settings the arena tool saved for that
recording; the tool writes it when Save is pressed for that recording, and
opening the tool writes nothing.
"""

from __future__ import annotations

import shutil

import pytest

import moseq2_extract.steps as extract_steps
from moseq2_common.yaml import read_yaml, write_yaml
from moseq2_extract.gui import extract_command, generate_config_command

from widgets import buttons_of, click

SESSIONS = ("saline_example_0", "saline_example_1")


@pytest.fixture
def project(tmp_path, raw_sessions):
    """Two recordings and a ``config.yaml`` from ``generate-config``."""
    root = tmp_path / "project"
    for name in SESSIONS:
        shutil.copytree(f"{raw_sessions}/{name}", root / name)
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    return root


@pytest.fixture
def received(monkeypatch):
    """The settings each extraction hands ``extract_wrapper``, by recording, instead of extracting."""
    calls = {}

    def record(input_file, output_dir, config, num_frames=None, skip=False):
        calls[input_file] = config
        return output_dir

    monkeypatch.setattr(extract_steps, "extract_wrapper", record)
    return calls


def recording(root, name="saline_example_0"):
    return str(root / name / "depth.avi")


def open_arena_tool(root):
    """The notebook's arena cell."""
    import panel as pn

    from moseq2_app.roi.widget import ArenaMaskWidget

    pn.extension()
    return ArenaMaskWidget(str(root))


def test_session_config_is_found_without_a_pointer(project, received):
    """config.yaml names no session_config.yaml, and the entry's max_height reaches extraction."""
    write_yaml({"saline_example_0/depth.avi": {"max_height": 100}}, str(project / "session_config.yaml"))
    assert "session_config_path" not in read_yaml(str(project / "config.yaml"))

    extract_command(recording(project), None, str(project / "config.yaml"))
    extract_command(recording(project, "saline_example_1"), None, str(project / "config.yaml"))

    assert received[recording(project)]["max_height"] == 100
    # a recording without an entry takes config.yaml's
    assert received[recording(project, "saline_example_1")]["max_height"] == 120


def test_the_command_line_finds_the_entry_too(project, received):
    from click.testing import CliRunner

    from moseq2_extract.cli import cli

    write_yaml({"saline_example_0/depth.avi": {"max_height": 100}}, str(project / "session_config.yaml"))

    result = CliRunner().invoke(cli, ["extract", recording(project), "--config-file", str(project / "config.yaml")],
                                catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert received[recording(project)]["max_height"] == 100


def test_a_moved_project_keeps_its_arena_settings(project, tmp_path, received):
    """Copied elsewhere and the original deleted, the project resolves the same options."""
    write_yaml({"saline_example_0/depth.avi": {"max_height": 100, "crop_size": [90, 90]}},
               str(project / "session_config.yaml"))
    extract_command(recording(project), None, str(project / "config.yaml"))
    before = received.pop(recording(project))

    moved = tmp_path / "elsewhere" / "moved"
    shutil.copytree(project, moved)
    shutil.rmtree(project)
    extract_command(recording(moved), None, str(moved / "config.yaml"))

    after = received[recording(moved)]
    assert after == before
    assert after["max_height"] == 100
    assert tuple(after["crop_size"]) == (90, 90)


def test_an_entry_setting_the_arena_tool_does_not_set_is_ignored_and_named(project, received, capsys):
    """An entry holds the arena tool's settings only; any other key is ignored with one line naming it."""
    write_yaml({"saline_example_0/depth.avi": {"max_height": 100, "frame_trim": [5, 5]}},
               str(project / "session_config.yaml"))

    extract_command(recording(project), None, str(project / "config.yaml"))

    config = received[recording(project)]
    assert config["max_height"] == 100
    assert tuple(config["frame_trim"]) == (0, 0)
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "frame_trim" in line]
    assert len(lines) == 1
    assert "saline_example_0/depth.avi" in lines[0]
    assert str(project / "session_config.yaml") in lines[0]


def test_an_unreadable_session_config_is_an_error(project, received):
    (project / "session_config.yaml").write_text("saline_example_0/depth.avi: [unclosed\n")

    with pytest.raises(Exception, match="session_config.yaml"):
        extract_command(recording(project), None, str(project / "config.yaml"))
    assert received == {}


def test_opening_the_arena_tool_writes_nothing(project):
    config_before = (project / "config.yaml").read_bytes()

    widget = open_arena_tool(project)

    assert list(widget.sessions) == ["saline_example_0/depth.avi", "saline_example_1/depth.avi"]
    assert not (project / "session_config.yaml").exists()
    assert (project / "config.yaml").read_bytes() == config_before


def test_saving_one_recording_writes_only_its_entry(project):
    from moseq2_extract.session_config import ARENA_TOOL_SETTINGS

    config_before = (project / "config.yaml").read_bytes()
    widget = open_arena_tool(project)
    buttons = buttons_of(widget.view)
    widget.session_data.crop_size = 100
    click(buttons["Save session parameters"])
    first = (project / "session_config.yaml").read_text()
    saved = read_yaml(str(project / "session_config.yaml"))
    assert list(saved) == ["saline_example_0/depth.avi"]
    assert set(saved["saline_example_0/depth.avi"]) <= set(ARENA_TOOL_SETTINGS)
    assert saved["saline_example_0/depth.avi"]["crop_size"] == [100, 100]

    widget.session_data.path = "saline_example_1/depth.avi"
    widget.session_data.noise_tolerance = 40
    click(buttons["Save session parameters"])

    after = (project / "session_config.yaml").read_text()
    assert first in after
    entries = read_yaml(str(project / "session_config.yaml"))
    assert entries["saline_example_0/depth.avi"] == saved["saline_example_0/depth.avi"]
    assert entries["saline_example_1/depth.avi"]["noise_tolerance"] == 40
    assert (project / "config.yaml").read_bytes() == config_before


def test_reopening_the_arena_tool_shows_the_saved_values(project):
    widget = open_arena_tool(project)
    widget.session_data.crop_size = 100
    click(buttons_of(widget.view)["Save session parameters"])

    reopened = open_arena_tool(project)

    assert reopened.session_data.path == "saline_example_0/depth.avi"
    assert reopened.session_data.crop_size == 100
