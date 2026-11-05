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
    write_yaml({"extract": {"cluster_type": "local", "noise_tolerance": 20}}, str(config))
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
    assert called["generate_config"] == [(str(root / "new_config.yaml"), "auto")]
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


def test_the_aggregate_function_takes_its_defaults_from_the_options_class(project, called, monkeypatch):
    """A default changed in ``AggregateOptions`` reaches the notebook's aggregation."""
    import moseq2_extract.gui as gui

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Changed(gui.AggregateOptions):
        mouse_threshold: float = 5.0

    monkeypatch.setattr(gui, "AggregateOptions", Changed)
    root, _, _ = project
    gui.aggregate_extract_results_command(str(root), "{session_name}", "aggregate_results/")
    gui.aggregate_extract_results_command(str(root), "{session_name}", "aggregate_results/", mouse_threshold=2.0)

    assert [(o.format, o.mouse_threshold) for o in called["aggregate_results"]] == [
        ("{session_name}", 5.0), ("{session_name}", 2.0)]


def test_batch_extract_hands_the_step_the_typed_extraction_options(project, monkeypatch, steps):
    """Every extraction option typed on batch-extract's command line, and only those, reaches
    each recording; num_frames, skip_completed and the cluster type travel in the options. A
    typed depth range is used as given (``RoiOptions.resolve``), and --output-file names no
    file in a batch."""
    root, _, config = project
    seen = {}

    def batch_extract(input_dir, opts, **kwargs):
        seen.update(kwargs, opts=opts)
        return steps.BatchExtractResult(found=[], extracted=[], failures=[], script=None)

    monkeypatch.setattr(steps, "batch_extract", batch_extract)
    invoke("batch-extract", root, "--config-file", config, "--bg-roi-depth-range", 600, 700, "--fps", 25,
           "--noise-tolerance", 30, "--num-frames", 9, "--skip-completed", "--cluster-type", "local",
           "--output-dir", "out", "--output-file", "o.h5", "--max-depth", 2, "--ncpus", 2)

    assert seen["output_dir"] == "out"
    assert seen["explicit"] == {"bg_roi_depth_range": (600.0, 700.0), "fps": 25, "noise_tolerance": 30}
    assert (seen["opts"].num_frames, seen["opts"].skip_completed, seen["opts"].max_depth) == (9, True, 2)


def test_batch_extract_without_typed_options_hands_the_step_none(project, monkeypatch, steps):
    root, _, config = project
    seen = {}

    def batch_extract(input_dir, opts, **kwargs):
        seen.update(kwargs)
        return steps.BatchExtractResult(found=[], extracted=[], failures=[], script=None)

    monkeypatch.setattr(steps, "batch_extract", batch_extract)
    invoke("batch-extract", root, "--config-file", config)

    assert (seen["output_dir"], seen["explicit"]) == (None, {})


def test_a_typed_depth_range_is_used_as_given(project, called):
    """A --bg-roi-depth-range typed on find-roi or extract is used as given (``RoiOptions.resolve``)."""
    root, recording, config = project
    invoke("find-roi", recording, "--config-file", config, "--bg-roi-depth-range", 600, 700)
    invoke("extract", recording, "--config-file", config, "--bg-roi-depth-range", 600, 700)
    invoke("extract", recording, "--config-file", config)
    assert [o.manual_set_depth_range for o in called["find_roi"] + called["extract"]] == [True, True, False]


def test_recording_settings_is_the_dict_extraction_hands_its_wrapper(steps):
    """One function gives the arena tool what extraction is handed: the options resolved from
    config.yaml's settings and the entry's arena settings (and any setting config.yaml lacks),
    tuples as the options hold them."""
    from moseq2_extract.session_config import recording_options, recording_settings

    config_data = {"bg_roi_depth_range": [600, 700], "noise_tolerance": 20, "fps": 30, "extra": 1}
    entry = {"noise_tolerance": 15, "fps": 25, "only_in_entry": 2, "crop_size": [70, 70]}

    settings = recording_settings(config_data, entry)

    assert settings == steps._settings(recording_options(config_data, entry))
    assert (settings["noise_tolerance"], settings["fps"]) == (15, 30)
    assert "only_in_entry" not in settings and "extra" not in settings
    assert settings["bg_roi_depth_range"] == (600, 700)
    assert settings["crop_size"] == (70, 70)
    assert settings["max_height"] == 120
