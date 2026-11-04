"""Options typed on the command line outrank the config file.

``command_with_config`` merges a command's options with its ``--config-file``.
An option typed on the command line wins, including when the typed value
happens to equal the option's default (moseq2-extract#24): ``--crop-size 80 80``
over a config holding ``[120, 120]`` extracts at 80. An option not typed takes
the config file's value where it has one.

``batch-extract`` run locally extracts each recording with the settings the
arena tool sets from that recording's entry in the session config, and every
other setting from the config file. The options typed on its command line are
applied over both, and ``--num-frames`` reaches the extraction
(moseq2-extract#158).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_extract.steps as extract_steps
from commands import split_command
from moseq2_common.yaml import write_yaml


def run(*args):
    result = CliRunner().invoke(extract_cli.cli, list(args), catch_exceptions=False)
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
    recording.touch()
    config = tmp_path / "config.yaml"
    write_yaml({"crop_size": [120, 120]}, str(config))

    run("extract", str(recording), "--config-file", str(config), *typed)

    assert tuple(received[0]["crop_size"]) == expected


SESSION = "session_1"


@pytest.fixture
def project(tmp_path):
    """One recording, a config file and a session config that disagree with
    the defaults and with each other. ``crop_size`` and ``manual_set_depth_range``
    are settings the arena tool sets, so the entry's apply; ``frame_trim`` and
    ``fps`` are not, so the config file's apply."""
    session = tmp_path / SESSION
    session.mkdir()
    (session / "depth.avi").touch()
    (session / "metadata.json").write_text("{}")

    session_config = tmp_path / "session_config.yaml"
    write_yaml(
        {
            SESSION: {
                "crop_size": [120, 120],
                "frame_trim": [5, 5],
                "fps": 15,
                "bg_roi_depth_range": [550, 650],
                "manual_set_depth_range": False,
            }
        },
        str(session_config),
    )
    config = tmp_path / "config.yaml"
    write_yaml(
        {
            "cluster_type": "local",
            "crop_size": [90, 90],
            "frame_trim": [0, 0],
            "fps": 20,
            "session_config_path": str(session_config),
        },
        str(config),
    )
    return tmp_path, config


@pytest.fixture
def extracted(monkeypatch):
    """What each local extraction hands ``extract_wrapper``, instead of running it."""
    calls = []

    def record(input_file, output_dir, config_data, num_frames=None, skip=False):
        calls.append(
            {"config": config_data, "num_frames": num_frames, "output_dir": output_dir}
        )

    monkeypatch.setattr(extract_steps, "extract_wrapper", record)
    return calls


def batch(root, config, *typed):
    return run(
        "batch-extract",
        str(root),
        "--config-file",
        str(config),
        "--extensions",
        ".avi",
        *typed,
    )


def test_batch_extract_typed_options_reach_each_recording(project, extracted):
    root, config = project
    batch(
        root,
        config,
        "--num-frames",
        "50",
        "--frame-trim",
        "10",
        "10",
        "--crop-size",
        "80",
        "80",
    )

    assert len(extracted) == 1
    call = extracted[0]
    assert call["num_frames"] == 50
    assert tuple(call["config"]["frame_trim"]) == (10, 10)
    assert tuple(call["config"]["crop_size"]) == (80, 80)
    # not typed, and not a setting the arena tool sets: the config file's value
    assert call["config"]["fps"] == 20


def test_batch_extract_without_typed_options_uses_the_session_config_for_the_arena_tool_settings(
    project, extracted
):
    root, config = project
    batch(root, config)

    call = extracted[0]
    assert call["num_frames"] is None
    assert tuple(call["config"]["crop_size"]) == (120, 120)
    assert call["config"]["manual_set_depth_range"] is False
    assert tuple(call["config"]["frame_trim"]) == (0, 0)
    assert call["config"]["fps"] == 20


def test_batch_extract_typed_depth_range_is_manual(project, extracted):
    root, config = project
    batch(root, config, "--bg-roi-depth-range", "600", "800")

    call = extracted[0]
    assert tuple(call["config"]["bg_roi_depth_range"]) == (600, 800)
    assert call["config"]["manual_set_depth_range"] is True


# batch-extract on SLURM writes one ``moseq2-extract extract`` command per
# recording into ``extract_out_script`` instead of extracting. The options typed
# on its command line reach each of those extractions with the same precedence
# as the local path (moseq2-extract#158). Each test runs the written command
# through ``extract`` itself, with ``extract_wrapper`` recorded instead of run,
# so what is checked is what the job would extract with.


@pytest.fixture
def slurm_extracted(monkeypatch, received):
    """Run ``batch-extract --cluster-type slurm`` without submitting anything,
    then run each written command through ``extract`` and return what it
    received: one dict per recording, the config and ``num_frames``."""
    monkeypatch.setattr(
        "moseq2_extract.helpers.extract.system",
        lambda *_: pytest.fail("batch-extract submitted a job"),
    )
    nums = []
    monkeypatch.setattr(
        extract_steps,
        "extract_wrapper",
        lambda f, out, config, num_frames=None, skip=False: (
            received.append(config) or nums.append(num_frames)
        ),
    )

    def go(root, config, *typed):
        batch(root, config, "--cluster-type", "slurm", *typed)
        script = (root / "extract_out.sh").read_text()
        lines = [ln for ln in script.splitlines() if ln.strip()]
        for line in lines:
            command = line.split("moseq2-extract extract", 1)[1].split(";", 1)[0]
            run("extract", *split_command(command))
        return [
            {"config": c, "num_frames": n} for c, n in zip(received, nums)
        ]

    return go


def test_slurm_batch_extract_typed_options_reach_each_recording(project, slurm_extracted):
    root, config = project
    calls = slurm_extracted(
        root,
        config,
        "--num-frames",
        "50",
        "--frame-trim",
        "10",
        "10",
        "--crop-size",
        "80",
        "80",
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["num_frames"] == 50
    assert tuple(call["config"]["frame_trim"]) == (10, 10)
    assert tuple(call["config"]["crop_size"]) == (80, 80)
    # not typed, and not a setting the arena tool sets: the config file's value
    assert call["config"]["fps"] == 20


def test_slurm_batch_extract_without_typed_options_uses_the_session_config_for_the_arena_tool_settings(
    project, slurm_extracted
):
    root, config = project
    call = slurm_extracted(root, config)[0]

    assert call["num_frames"] is None
    assert tuple(call["config"]["crop_size"]) == (120, 120)
    assert call["config"]["manual_set_depth_range"] is False
    assert tuple(call["config"]["frame_trim"]) == (0, 0)
    assert call["config"]["fps"] == 20


def test_slurm_batch_extract_typed_depth_range_is_manual(project, slurm_extracted):
    root, config = project
    call = slurm_extracted(root, config, "--bg-roi-depth-range", "600", "800")[0]

    assert tuple(call["config"]["bg_roi_depth_range"]) == (600, 800)
    assert call["config"]["manual_set_depth_range"] is True


def test_slurm_batch_extract_typed_options_without_a_session_config(
    project, slurm_extracted
):
    """With no session config, each recording is extracted from the config file,
    and the typed options still apply over it."""
    root, config = project
    write_yaml({"cluster_type": "local", "crop_size": [90, 90], "fps": 20}, str(config))

    call = slurm_extracted(
        root, config, "--num-frames", "50", "--crop-size", "80", "80"
    )[0]

    assert call["num_frames"] == 50
    assert tuple(call["config"]["crop_size"]) == (80, 80)
    # not typed: the config file's value
    assert call["config"]["fps"] == 20


def test_slurm_batch_extract_without_typed_options_or_session_config(
    project, slurm_extracted
):
    root, config = project
    write_yaml({"cluster_type": "local", "crop_size": [90, 90], "fps": 20}, str(config))

    call = slurm_extracted(root, config)[0]

    assert call["num_frames"] is None
    assert tuple(call["config"]["crop_size"]) == (90, 90)
    assert call["config"]["fps"] == 20


@pytest.mark.parametrize("command", ["extract", "batch-extract"])
def test_threads_help_says_it_is_for_reading_the_recording(command):
    """On the extraction commands ``--threads`` decodes; ``--compress-threads`` encodes."""
    (option,) = [p for p in extract_cli.cli.commands[command].params if p.name == "threads"]
    assert "decod" in option.help
    assert "encoding" not in option.help
