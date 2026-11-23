"""Each extraction step is one function, called by both front ends.

``moseq2-extract find-roi`` and ``moseq2 standalone extract`` resolve their
settings into one options object (``moseq2_extract.options``) and hand it to
the file layer, ``moseq2_extract.steps``; ``generate-config`` and
``download-flip-file``, on the command line and as the notebook's functions,
call ``moseq2_extract.project``. ``moseq2 extract`` and the notebook's Extract
step both run ``moseq2_extract.project.extract`` in the project, handing it the
settings typed (or changed on the page) for it to save. A ``--config-file``
that does not exist stops the command, naming the file, and ``moseq2 extract``
in a folder with no ``config.yaml`` stops, naming the folder.

``extract_chunk`` takes every setting the options declare from its caller, with
no default of its own, and the arena tool saves exactly the settings
``ExtractOptions`` marks as the tool's.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import os

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
from moseq2_app.cli import cli
from moseq2_common.yaml import write_yaml


@pytest.fixture
def steps():
    import moseq2_extract.steps

    return moseq2_extract.steps


@pytest.fixture
def project_steps():
    import moseq2_extract.project

    return moseq2_extract.project


@pytest.fixture
def options():
    import moseq2_extract.options

    return moseq2_extract.options


@pytest.fixture
def called(monkeypatch, steps, project_steps):
    """Each step replaced by a recorder of its options (or its arguments, for the project's extraction and the
    setup steps)."""
    calls = {name: [] for name in ("find_roi", "extract", "project_extract", "generate_config",
                                   "download_flip_file")}

    def find_roi(input_file, output_dir, opts, **kwargs):
        calls["find_roi"].append(opts)
        return steps.FindRoiResult(roi_files=[])

    def extract(input_file, output_dir, opts, **kwargs):
        calls["extract"].append(opts)
        return steps.ExtractResult(h5_file=None)

    def project_extract(project, *, recordings=None, settings=None, reporter=None):
        calls["project_extract"].append({"folder": os.path.realpath(project.folder), "recordings": recordings,
                                         "settings": settings})
        return project_steps.ExtractRun(extracted=[], failed={}, up_to_date=[], jobs={})

    def generate_config(output_file, camera_type, *, overwrite):
        calls["generate_config"].append((output_file, camera_type))
        return output_file

    def download_flip_file(config_file, output_dir, selection):
        calls["download_flip_file"].append(selection)
        return None

    for name, function in (("find_roi", find_roi), ("extract", extract)):
        monkeypatch.setattr(steps, name, function)
    for name, function in (("extract", project_extract), ("generate_config", generate_config),
                           ("download_flip_file", download_flip_file)):
        monkeypatch.setattr(project_steps, name, function)
    return calls


@pytest.fixture
def project(tmp_path):
    """A recording file (empty: the steps are stood in for) and a config.yaml."""
    recording = tmp_path / "session" / "depth.dat"
    recording.parent.mkdir()
    recording.write_bytes(b"")
    config = tmp_path / "config.yaml"
    write_yaml({"extract": {"noise_tolerance": 20}, "computing": {"extract": {"cluster_type": "local"}}},
               str(config))
    return tmp_path, recording, config


@contextlib.contextmanager
def working_in(folder):
    """Run with ``folder`` as the working folder, as a person running a command from the project does."""
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def invoke(*args, group=extract_cli.cli):
    """``ARGS`` on ``group``'s command line: ``moseq2-extract`` unless ``group`` is another."""
    result = CliRunner().invoke(group, [str(a) for a in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def test_every_extract_command_calls_its_step(project, called, options):
    root, recording, config = project

    invoke("find-roi", recording, "--config-file", config)
    invoke("standalone", "extract", recording, "--config-file", config, group=cli)
    with working_in(root):
        invoke("extract", group=cli)
    invoke("generate-config", "-o", root / "new_config.yaml")
    invoke("download-flip-file", config, "--output-dir", root)

    expected = {"find_roi": options.RoiOptions, "extract": options.ExtractOptions}
    for name, cls in expected.items():
        assert len(called[name]) == 1, name
        assert type(called[name][0]) is cls, name
    assert called["extract"][0].noise_tolerance == 20
    assert called["project_extract"] == [{"folder": os.path.realpath(root), "recordings": None, "settings": {}}]
    assert called["generate_config"] == [(str(root / "new_config.yaml"), "auto")]
    assert called["download_flip_file"] == [None]


def test_the_notebook_functions_call_the_steps(project, called, options):
    from moseq2_app.notebook import extract
    from moseq2_extract.gui import download_flip_command, generate_config_command

    root, recording, config = project

    with working_in(root):
        widget = extract()
        widget.start()
        run = widget.wait()
    generate_config_command(str(root / "new_config.yaml"), camera_type="azure")
    download_flip_command(str(root), str(config), selection=1)

    assert run["state"] == "finished", widget.problem
    assert called["project_extract"] == [{"folder": os.path.realpath(root), "recordings": None, "settings": {}}]
    assert called["generate_config"] == [(str(root / "new_config.yaml"), "azure")]
    assert called["download_flip_file"] == [1]


@pytest.mark.parametrize("command", ["standalone-extract", "find-roi", "extract"])
def test_extract_commands_stop_on_a_missing_config_file(project, command):
    """``--config-file`` naming no file stops ``moseq2 standalone extract`` and ``moseq2-extract find-roi``,
    naming it; ``moseq2 extract``, whose config file is the working folder's ``config.yaml``, stops in a folder
    without one, naming the folder."""
    root, recording, _ = project
    if command == "extract":
        elsewhere = root / "not_a_project"
        elsewhere.mkdir()
        missing = elsewhere / "config.yaml"
        with working_in(elsewhere):
            result = CliRunner().invoke(cli, ["extract"])
        assert result.exit_code != 0
        assert f"{elsewhere} has no config.yaml" in result.output
        assert not missing.exists()
        return
    missing = root / "missing.yaml"
    group, words = (cli, ["standalone", "extract"]) if command == "standalone-extract" else (extract_cli.cli,
                                                                                            [command])
    result = CliRunner().invoke(group, [*words, str(recording), "--config-file", str(missing)])
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


def test_batch_extract_hands_the_step_the_typed_extraction_options(project, called):
    """Every setting typed on ``moseq2 extract``'s command line, and only those, reaches the project's
    extraction as the settings it saves before the run: extraction's own (num_frames among them), the
    recordings search's and the computing block's; a recording named is handed on as named."""
    root, _, _ = project
    with working_in(root):
        invoke("extract", "--bg-roi-depth-range", 600, 700, "--noise-tolerance", 30, "--num-frames", 9,
               "--cluster-type", "local", "--cores", 2, "--max-depth", 2, "session/depth.dat", group=cli)

    (seen,) = called["project_extract"]
    assert seen["recordings"] == ["session/depth.dat"]
    assert seen["settings"] == {"bg_roi_depth_range": (600.0, 700.0), "noise_tolerance": 30, "num_frames": 9,
                                "cluster_type": "local", "cores": 2, "max_depth": 2}


def test_batch_extract_without_typed_options_hands_the_step_none(project, called):
    root, _, _ = project
    with working_in(root):
        invoke("extract", group=cli)

    (seen,) = called["project_extract"]
    assert (seen["recordings"], seen["settings"]) == (None, {})


def test_a_typed_depth_range_is_used_as_given(project, called):
    """A --bg-roi-depth-range typed on ``moseq2-extract find-roi`` or ``moseq2 standalone extract`` is used as
    given (``RoiOptions.resolve``)."""
    root, recording, config = project
    invoke("find-roi", recording, "--config-file", config, "--bg-roi-depth-range", 600, 700)
    invoke("standalone", "extract", recording, "--config-file", config, "--bg-roi-depth-range", 600, 700, group=cli)
    invoke("standalone", "extract", recording, "--config-file", config, group=cli)
    assert [o.manual_set_depth_range for o in called["find_roi"] + called["extract"]] == [True, True, False]


def test_recording_settings_is_the_dict_extraction_hands_its_wrapper(steps):
    """One function gives the arena tool what extraction is handed: the options resolved from
    config.yaml's settings and the entry's arena settings, tuples as the options hold them.
    ``compress_threads`` is a one-run switch that neither sets."""
    from moseq2_extract.options import ExtractOptions
    from moseq2_extract.session_config import recording_options, recording_settings

    config_data = {"bg_roi_depth_range": [600, 700], "noise_tolerance": 20, "compress_threads": 30, "extra": 1}
    entry = {"noise_tolerance": 15, "compress_threads": 25, "only_in_entry": 2, "crop_size": [70, 70]}

    settings = recording_settings(config_data, entry)

    assert settings == steps._settings(recording_options(config_data, entry))
    assert (settings["noise_tolerance"], settings["compress_threads"]) == (15, ExtractOptions().compress_threads)
    assert "only_in_entry" not in settings and "extra" not in settings
    assert settings["bg_roi_depth_range"] == (600, 700)
    assert settings["crop_size"] == (70, 70)
    assert settings["max_height"] == 120
