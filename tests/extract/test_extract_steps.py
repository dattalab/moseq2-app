"""Each extraction step is one function, called by both front ends.

``find-roi``, ``extract``, ``batch-extract`` and ``aggregate-results`` on the
command line and the Extract notebook's functions resolve their settings into
one options object (``moseq2_extract.options``) and hand it to the same step in
``moseq2_extract.steps``; ``generate-config`` and ``download-flip-file`` call
``moseq2_extract.setup``. A ``--config-file`` that does not exist stops the
command, naming the file.

``extract_chunk`` takes every setting the options declare from its caller, with
no default of its own, and the arena tool saves exactly the settings
``ExtractOptions`` marks as the tool's.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
from moseq2_common.yaml import write_yaml


@pytest.fixture
def steps():
    import moseq2_extract.steps

    return moseq2_extract.steps


@pytest.fixture
def setup():
    import moseq2_extract.setup

    return moseq2_extract.setup


@pytest.fixture
def options():
    import moseq2_extract.options

    return moseq2_extract.options


@pytest.fixture
def called(monkeypatch, steps, setup):
    """Each step replaced by a recorder of its options (or its arguments, for setup)."""
    calls = {name: [] for name in ("find_roi", "extract", "batch_extract", "aggregate_results",
                                   "generate_config", "download_flip_file")}

    def find_roi(input_file, output_dir, opts, **kwargs):
        calls["find_roi"].append(opts)
        return steps.FindRoiResult(roi_files=[])

    def extract(input_file, output_dir, opts, **kwargs):
        calls["extract"].append(opts)
        return steps.ExtractResult(h5_file=None)

    def batch_extract(input_dir, opts, **kwargs):
        calls["batch_extract"].append(opts)
        return steps.BatchExtractResult(found=[], extracted=[], failures=[], script=None)

    def aggregate_results(input_dir, output_dir, opts):
        calls["aggregate_results"].append(opts)
        return steps.AggregateResult(index_file=None)

    def generate_config(output_file, camera_type, *, overwrite):
        calls["generate_config"].append((output_file, camera_type))
        return output_file

    def download_flip_file(config_file, output_dir, selection):
        calls["download_flip_file"].append(selection)
        return None

    for name, function in (("find_roi", find_roi), ("extract", extract), ("batch_extract", batch_extract),
                           ("aggregate_results", aggregate_results)):
        monkeypatch.setattr(steps, name, function)
    monkeypatch.setattr(setup, "generate_config", generate_config)
    monkeypatch.setattr(setup, "download_flip_file", download_flip_file)
    return calls


@pytest.fixture
def project(tmp_path):
    """A recording file (empty: the steps are stood in for) and a config.yaml."""
    recording = tmp_path / "session" / "depth.dat"
    recording.parent.mkdir()
    recording.write_bytes(b"")
    config = tmp_path / "config.yaml"
    write_yaml({"cluster_type": "local", "noise_tolerance": 20}, str(config))
    return tmp_path, recording, config


def invoke(*args):
    result = CliRunner().invoke(extract_cli.cli, [str(a) for a in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def test_every_extract_command_calls_its_step(project, called, options):
    root, recording, config = project

    invoke("find-roi", recording, "--config-file", config)
    invoke("extract", recording, "--config-file", config)
    invoke("batch-extract", root, "--config-file", config)
    invoke("aggregate-results", "-i", root, "-o", root / "aggregate_results")
    invoke("generate-config", "-o", root / "new_config.yaml")
    invoke("download-flip-file", config, "--output-dir", root)

    expected = {"find_roi": options.RoiOptions, "extract": options.ExtractOptions,
                "batch_extract": options.BatchExtractOptions, "aggregate_results": options.AggregateOptions}
    for name, cls in expected.items():
        assert len(called[name]) == 1, name
        assert type(called[name][0]) is cls, name
    assert called["extract"][0].noise_tolerance == 20
    assert called["generate_config"] == [(str(root / "new_config.yaml"), "k2")]
    assert called["download_flip_file"] == [None]


def test_the_notebook_functions_call_the_steps(project, called, options):
    from moseq2_extract.gui import (aggregate_extract_results_command, download_flip_command, extract_command,
                                    extract_found_sessions, generate_config_command)

    root, recording, config = project

    extract_found_sessions(str(root), str(config), [".dat"])
    extract_command(str(recording), None, str(config))
    aggregate_extract_results_command(str(root), "{session_name}", "aggregate_results/")
    generate_config_command(str(root / "new_config.yaml"), camera_type="azure")
    download_flip_command(str(root), str(config), selection=1)

    expected = {"extract": options.ExtractOptions, "batch_extract": options.BatchExtractOptions,
                "aggregate_results": options.AggregateOptions}
    for name, cls in expected.items():
        assert len(called[name]) == 1, name
        assert type(called[name][0]) is cls, name
    assert called["extract"][0].noise_tolerance == 20
    assert called["generate_config"] == [(str(root / "new_config.yaml"), "azure")]
    assert called["download_flip_file"] == [1]


@pytest.mark.parametrize("command", ["extract", "find-roi", "batch-extract"])
def test_extract_commands_stop_on_a_missing_config_file(project, command):
    root, recording, _ = project
    missing = root / "missing.yaml"
    target = root if command == "batch-extract" else recording
    result = CliRunner().invoke(extract_cli.cli, [command, str(target), "--config-file", str(missing)])
    assert result.exit_code != 0
    assert str(missing) in result.output
    assert not missing.exists()


def test_extract_chunk_has_no_default_for_a_declared_option(options):
    from moseq2_extract.extract.extract import extract_chunk

    declared = {f.name for f in dataclasses.fields(options.ExtractOptions)}
    parameters = inspect.signature(extract_chunk).parameters
    with_default = sorted(name for name, p in parameters.items()
                          if name in declared and p.default is not inspect.Parameter.empty)
    assert with_default == []


def test_arena_tool_settings_are_the_arena_fields(options):
    """The keys the arena tool's Save writes (as in tests/app/test_arena_tool_settings.py)."""
    from moseq2_app.roi.widget import ArenaMaskWidget
    from moseq2_common.options import arena_fields

    class Tuned:
        path = "session"
        mouse_height = (10, 100)
        spatial_filter = "[3]"
        temporal_filter = "[0]"

        def __getattr__(self, name):
            return 1

    widget = ArenaMaskWidget.__new__(ArenaMaskWidget)
    widget.session_config = {"session": {}}
    widget.session_data = Tuned()
    widget.set_session_config_vars()

    assert set(arena_fields(options.ExtractOptions)) == set(widget.session_config["session"])
