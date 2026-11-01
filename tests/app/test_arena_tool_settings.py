"""The settings the arena tool saves for a recording are the ones extraction takes from its entry.

``ARENA_TOOL_SETTINGS`` in moseq2-extract lists them, and extraction takes only
those from a recording's ``session_config.yaml`` entry and every other setting
from ``config.yaml``. The list and the tool's ``set_session_config_vars`` must
agree: a key the tool writes but the list lacks would be tuned in the tool and
then ignored, and a key the list holds but the tool never writes would be frozen
at the value copied when the tool first listed the recording.

The second test follows the notebook: the
arena cell runs, a recording is tuned and saved, then ``config.yaml`` gains a
flip classifier and a ``frame_trim`` and the arena cell runs again before the
extraction cell.
"""

from __future__ import annotations

import shutil

import pytest

import moseq2_extract.gui as extract_gui
from moseq2_app.util import update_config
from moseq2_extract.gui import extract_command, generate_config_command
from moseq2_extract.util import read_yaml

from widgets import buttons_of, click


class _Tuned:
    """Stands in for the tool's ``ArenaMaskData``: every value the tool shows."""

    path = "session"
    mouse_height = (10, 100)
    spatial_filter = "[3]"
    temporal_filter = "[0]"

    def __getattr__(self, name):
        return 1


def test_the_tool_saves_exactly_the_shared_list():
    from moseq2_app.roi.widget import ArenaMaskWidget
    from moseq2_extract.session_config import ARENA_TOOL_SETTINGS

    widget = ArenaMaskWidget.__new__(ArenaMaskWidget)
    widget.session_config = {"session": {}}
    widget.session_data = _Tuned()

    widget.set_session_config_vars()

    assert sorted(widget.session_config["session"]) == sorted(ARENA_TOOL_SETTINGS)


def open_arena_tool(root):
    """The notebook's arena cell."""
    import panel as pn

    from moseq2_app.roi.widget import ArenaMaskWidget

    pn.extension()
    with update_config(str(root / "config.yaml")) as config:
        config["output_dir"] = "proc"
    return ArenaMaskWidget(str(root), str(root / "config.yaml"), str(root / "session_config.yaml"))


@pytest.fixture
def project(tmp_path, raw_sessions):
    root = tmp_path / "project"
    for name in ("saline_example_0", "saline_example_1"):
        shutil.copytree(f"{raw_sessions}/{name}", root / name)
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    return root


def test_config_yaml_changes_after_the_tool_reach_extraction(project, monkeypatch, capsys, raw_sessions):
    widget = open_arena_tool(project)
    data = widget.session_data
    assert data.path == "saline_example_0"
    data.crop_size = 100
    data.depth_range = (600, 700)
    click(buttons_of(widget.view)["Save session parameters"])

    classifier = f"{raw_sessions}/flip_classifier_k2_c57_10to13weeks.npz"
    with update_config(str(project / "config.yaml")) as config:
        config["flip_classifier"] = classifier
        config["frame_trim"] = [100, 0]
        config["crop_size"] = [120, 120]
        config["bg_roi_depth_range"] = [650, 750]
    open_arena_tool(project)
    capsys.readouterr()

    received = {}
    monkeypatch.setattr(
        extract_gui,
        "extract_wrapper",
        lambda input_file, output_dir, config, num_frames=None, skip=False: received.update(config),
    )
    extract_command(str(project / "saline_example_0" / "depth.avi"), None, str(project / "config.yaml"))

    assert received["flip_classifier"] == classifier
    assert tuple(received["frame_trim"]) == (100, 0)
    # tuned in the tool
    assert tuple(received["crop_size"]) == (100, 100)
    assert tuple(received["bg_roi_depth_range"]) == (600, 700)
    assert received["manual_set_depth_range"] is True

    out = capsys.readouterr().out
    assert f'flip_classifier: config.yaml\'s "{classifier}" is used.' in out
    assert "frame_trim: config.yaml's [100, 0] is used." in out
    assert "for saline_example_0." in out
    # the entry on disk is not rewritten by extraction
    assert read_yaml(str(project / "session_config.yaml"))["saline_example_0"]["flip_classifier"] is None
