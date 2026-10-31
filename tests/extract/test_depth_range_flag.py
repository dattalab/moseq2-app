"""A floor depth range typed on the command line.

Without ``manual_set_depth_range``, finding the arena replaces
``bg_roi_depth_range`` with a range detected from the background
(moseq2-extract#121, #125). A range a user types as ``--bg-roi-depth-range``
is used as given. A range that only comes from the config file still follows
``manual_set_depth_range``, because a generated config holds a range nobody
chose (``[550, 650]`` for the Azure).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
from moseq2_extract._yaml import write_yaml


@pytest.fixture
def received(monkeypatch):
    """What each command hands its wrapper, instead of running it."""
    calls = []
    monkeypatch.setattr(extract_cli, "get_roi_wrapper", lambda f, config, out: calls.append(config))
    monkeypatch.setattr(extract_cli, "extract_wrapper", lambda f, out, config, **kw: calls.append(config))
    return calls


def run(*args):
    result = CliRunner().invoke(extract_cli.cli, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", ["find-roi", "extract"])
def test_a_typed_range_is_used_as_given(command, tmp_path, received):
    recording = tmp_path / "depth.avi"
    recording.touch()
    run(command, str(recording), "--bg-roi-depth-range", "600", "800")
    assert received[0]["manual_set_depth_range"] is True
    assert tuple(received[0]["bg_roi_depth_range"]) == (600, 800)


@pytest.mark.parametrize("command", ["find-roi", "extract"])
def test_without_a_typed_range_the_range_is_detected(command, tmp_path, received):
    recording = tmp_path / "depth.avi"
    recording.touch()
    run(command, str(recording))
    assert received[0]["manual_set_depth_range"] is False


def test_a_range_from_the_config_file_follows_the_flag(tmp_path, received):
    recording = tmp_path / "depth.avi"
    recording.touch()
    config = tmp_path / "config.yaml"
    write_yaml({"bg_roi_depth_range": [550, 650]}, str(config))
    run("find-roi", str(recording), "--config-file", str(config))
    assert received[0]["manual_set_depth_range"] is False
    assert tuple(received[0]["bg_roi_depth_range"]) == (550, 650)
