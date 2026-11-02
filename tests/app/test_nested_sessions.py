"""Projects whose sessions sit below group folders, and session folders that share a name.

``find_recordings`` (``moseq2_extract.recordings``) finds a project's
recordings for the progress cell, the arena tool and extraction. The tests
here follow what those cells do with what it finds:

- the extraction cell, with the notebook's own text, takes every recording
  type the pipeline reads, ``.mkv`` included;
- the validation and preview cells find each recording's extraction in the
  ``proc`` folder beside the recording, however deep it is;
- a session is named, in ``session_config.yaml`` and in the notebook's lists,
  by its folder's name, or by its folder's path below the project when two
  session folders share a name (``GroupA/mouse1`` and ``GroupB/mouse1``), so
  each keeps its own parameters through the arena tool, a local extraction
  and the SLURM jobs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import moseq2_extract.cli as extract_cli
from cells import code_cell
from commands import split_command
from moseq2_app.gui.progress import get_session_paths
from moseq2_app.roi.controller import InteractiveExtractionViewer
from moseq2_app.util import update_config
from moseq2_common.yaml import write_yaml
from moseq2_extract.gui import extract_found_sessions, generate_config_command
from moseq2_extract.util import read_yaml

EXTRACT_NOTEBOOK = "MoSeq2-Extract-Modeling-Notebook.ipynb"

#: The extraction cell, numbered as ``code_cell`` counts.
EXTRACTION_CELL = 6

#: A grouped project made from the fixture recordings: where each session
#: folder sits below the project, and the recording copied into it. The two
#: ``mouse1`` folders share a name; ``saline_example_0`` is the only folder of
#: its name.
GROUPED = {
    "GroupA/mouse1": "amphetamine_example_0",
    "GroupB/mouse1": "amphetamine_example_1",
    "saline_example_0": "saline_example_0",
}


def empty_session(root, where, recording="depth.dat", extracted=False):
    """A session folder at ``root/where`` holding an empty recording, and an
    empty extraction movie and flipped movie in its ``proc`` when ``extracted``."""
    folder = Path(root, where)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / recording).touch()
    (folder / "metadata.json").write_text("{}")
    if extracted:
        (folder / "proc").mkdir()
        (folder / "proc" / "results_00.mp4").touch()
        (folder / "proc" / "results_00_flipped.mp4").touch()
    return folder / recording


@pytest.fixture
def chosen(monkeypatch):
    """The recordings handed to the local extraction loop; nothing is extracted."""
    recorded = []
    monkeypatch.setattr(
        "moseq2_extract.gui.run_local_extract",
        lambda to_extract, config_file, skip_extracted=False: recorded.extend(to_extract),
    )
    return recorded


@pytest.fixture
def grouped(tmp_path, raw_sessions):
    """The ``GROUPED`` project, with a config from ``generate-config``."""
    root = tmp_path / "project"
    for where, source in GROUPED.items():
        shutil.copytree(Path(raw_sessions) / source, root / where)
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    return root


def open_arena_tool(root):
    from moseq2_app.roi.widget import ArenaMaskWidget

    return ArenaMaskWidget(str(root), str(root / "config.yaml"), str(root / "session_config.yaml"))


def give_each_session_its_own_noise_tolerance(root):
    """Write a ``session_config.yaml`` keyed as the arena tool keys ``GROUPED``
    (see ``test_the_arena_tool_keeps_parameters_for_each_same_named_folder``),
    with a different ``noise_tolerance`` (a setting the arena tool saves) for
    each session, the way a user tuning each recording leaves them different;
    return the noise tolerance by session."""
    base = read_yaml(str(root / "config.yaml"))
    tolerance = {key: 11 + number for number, key in enumerate(sorted(GROUPED))}
    write_yaml({key: {**base, "noise_tolerance": value} for key, value in tolerance.items()}, str(root / "session_config.yaml"))
    with update_config(str(root / "config.yaml")) as config:
        config["session_config_path"] = str(root / "session_config.yaml")
    return tolerance


# ------------------------------------------------------------------ the extraction cell


def test_extraction_without_extensions_takes_every_recording_type(tmp_path, chosen):
    """``extract_found_sessions`` called without extensions takes every type
    ``find_recordings`` knows: ``.dat``, ``.mkv`` (any name), ``.avi`` and ``.tar.gz``."""
    root = tmp_path / "project"
    expected = sorted(str(p) for p in (
        empty_session(root, "kinect"),
        empty_session(root, "azure", recording="2024-05-01-10-31-07.mkv"),
        empty_session(root, "converted", recording="depth.avi"),
        empty_session(root, "archived", recording="session_20190320.tar.gz"),
    ))
    config = tmp_path / "config.yaml"
    config.write_text("cluster_type: local\n")

    extract_found_sessions(str(root), str(config))

    assert sorted(chosen) == expected


def test_extensions_passed_by_an_older_notebook_are_still_obeyed(tmp_path, chosen, capsys):
    """An older notebook passes its list; it gets what it asks for, and the
    message names the ``.mkv`` passed over."""
    root = tmp_path / "project"
    dat = empty_session(root, "kinect")
    mkv = empty_session(root, "azure", recording="recording.mkv")
    config = tmp_path / "config.yaml"
    config.write_text("cluster_type: local\n")
    capsys.readouterr()

    extract_found_sessions(str(root), str(config), [".avi", ".dat", ".tar.gz"])

    assert chosen == [str(dat)]
    assert f"{mkv.parent}: holds recording.mkv" in capsys.readouterr().out


def test_the_notebooks_extraction_cell_extracts_an_mkv(tmp_path, chosen):
    """The cell as the notebook ships it, on a project of Azure or Femto ``.mkv``
    recordings and a Kinect v2 ``.dat``, with no edit by the user."""
    root = tmp_path / "project"
    expected = sorted(str(p) for p in (
        empty_session(root, "GroupA/femto_0", recording="2024-05-01-10-31-07.mkv"),
        empty_session(root, "GroupB/kinect_0"),
    ))
    config = root / "config.yaml"
    generate_config_command(str(config), camera_type="azure")

    namespace = {
        "progress_paths": {"base_dir": str(root), "config_file": str(config)},
        "skip_extracted": True,
    }
    exec(compile(code_cell(EXTRACT_NOTEBOOK, EXTRACTION_CELL), "<extraction cell>", "exec"), namespace)

    assert sorted(chosen) == expected


# ------------------------------------------------------------- extractions two levels down


def test_extractions_two_levels_down_are_listed(tmp_path):
    """Validation and preview list the movie in the ``proc`` folder beside each
    recording, at any depth the recordings are found at."""
    root = tmp_path / "project"
    for where in ("GroupA/mouse1", "GroupB/mouse2", "solo"):
        empty_session(root, where, extracted=True)
    empty_session(root, "GroupB/not_yet_extracted")

    extracted = get_session_paths(str(root), extracted=True)
    flipped = get_session_paths(str(root), extracted=True, flipped=True)

    assert extracted == {
        name: str(root / where / "proc" / "results_00.mp4")
        for name, where in (("mouse1", "GroupA/mouse1"), ("mouse2", "GroupB/mouse2"), ("solo", "solo"))
    }
    assert flipped == {
        name: str(root / where / "proc" / "results_00_flipped.mp4")
        for name, where in (("mouse1", "GroupA/mouse1"), ("mouse2", "GroupB/mouse2"), ("solo", "solo"))
    }


def test_an_extraction_whose_recording_was_removed_is_still_listed(tmp_path):
    """A session folder one level down whose recording was moved away after
    extraction keeps its extraction listed, as it always was."""
    root = tmp_path / "project"
    kept = empty_session(root, "s0", extracted=True)
    removed = empty_session(root, "s1", extracted=True)
    removed.unlink()

    assert get_session_paths(str(root), extracted=True) == {
        "s0": str(kept.parent / "proc" / "results_00.mp4"),
        "s1": str(removed.parent / "proc" / "results_00.mp4"),
    }


def test_the_preview_offers_extractions_two_levels_down(tmp_path):
    root = tmp_path / "project"
    for where in ("GroupA/mouse1", "GroupB/mouse1", "solo"):
        empty_session(root, where, extracted=True)

    viewer = InteractiveExtractionViewer(str(root))

    assert dict(viewer.sess_select.options) == {
        "GroupA/mouse1": str(root / "GroupA/mouse1/proc/results_00.mp4"),
        "GroupB/mouse1": str(root / "GroupB/mouse1/proc/results_00.mp4"),
        "solo": str(root / "solo/proc/results_00.mp4"),
    }


def test_validation_reports_every_extraction_in_a_grouped_project(tmp_path, extracted):
    """The chain's four extractions moved into group folders, two of them into
    folders that share a name, are all validated."""
    import re
    from contextlib import redirect_stdout
    from io import StringIO

    from moseq2_app.main import validate_extractions

    root = tmp_path / "project"
    layout = {
        "GroupA/mouse1": "amphetamine_example_0",
        "GroupB/mouse1": "amphetamine_example_1",
        "GroupB/saline_example_0": "saline_example_0",
        "saline_example_1": "saline_example_1",
    }
    for where, source in layout.items():
        shutil.copytree(Path(extracted) / source, root / where)

    printed = StringIO()
    with redirect_stdout(printed):
        validate_extractions(str(root))
    flagged = re.search(r"(\d+)/(\d+) were flagged with warning\(s\)\.", printed.getvalue())

    assert flagged, printed.getvalue()
    assert int(flagged.group(2)) == len(layout)


# ------------------------------------------------------------ session folders sharing a name


def test_the_progress_cell_names_shared_folder_names_by_their_path(tmp_path):
    """A folder whose name no other session folder has is named by it; two
    that share a name are named by their path below the project."""
    root = tmp_path / "project"
    a = empty_session(root, "GroupA/mouse1")
    b = empty_session(root, "GroupB/mouse1")
    solo = empty_session(root, "GroupB/mouse2")

    assert get_session_paths(str(root)) == {
        "GroupA/mouse1": str(a),
        "GroupB/mouse1": str(b),
        "mouse2": str(solo),
    }


def test_the_arena_tool_keeps_parameters_for_each_same_named_folder(grouped):
    """The arena tool saves each ``mouse1`` under its path and the folder whose
    name is its own under that name, and saving one leaves the other alone."""
    from widgets import buttons_of, click

    widget = open_arena_tool(grouped)
    saved = read_yaml(str(grouped / "session_config.yaml"))

    assert sorted(saved) == ["GroupA/mouse1", "GroupB/mouse1", "saline_example_0"]
    assert widget.sessions == {key: str(grouped / key / "depth.avi") for key in saved}

    data = widget.session_data
    data.path = "GroupA/mouse1"
    data.depth_range = (600, 700)
    click(buttons_of(widget.view)["Save session parameters"])

    saved = read_yaml(str(grouped / "session_config.yaml"))
    assert list(saved["GroupA/mouse1"]["bg_roi_depth_range"]) == [600, 700]
    assert list(saved["GroupB/mouse1"]["bg_roi_depth_range"]) != [600, 700]


def test_an_older_file_keyed_by_a_shared_folder_name_is_explained(grouped, capsys):
    """A ``session_config.yaml`` written before shared names were told apart
    holds one ``mouse1`` entry, saved for one of the two folders. The arena tool
    cannot tell which, so it detects parameters for each under its path and
    says so; the unshared name keeps its entry."""
    old = {**read_yaml(str(grouped / "config.yaml")), "fps": 7}
    write_yaml({"mouse1": old, "saline_example_0": old}, str(grouped / "session_config.yaml"))
    capsys.readouterr()

    open_arena_tool(grouped)

    saved = read_yaml(str(grouped / "session_config.yaml"))
    assert sorted(saved) == ["GroupA/mouse1", "GroupB/mouse1", "mouse1", "saline_example_0"]
    assert saved["saline_example_0"]["fps"] == 7
    assert saved["GroupA/mouse1"]["fps"] != 7
    out = capsys.readouterr().out
    assert (
        f'{grouped / "session_config.yaml"} holds parameters under "mouse1", a folder name '
        f'that {grouped / "GroupA/mouse1"} and {grouped / "GroupB/mouse1"} share' in out
    )


def test_a_local_extraction_reads_each_same_named_folders_parameters(grouped, monkeypatch):
    received = {}
    monkeypatch.setattr(
        "moseq2_extract.gui.extract_wrapper",
        lambda input_file, output_dir, config, num_frames=None, skip=False: received.update(
            {input_file: config["noise_tolerance"]}
        ),
    )
    tolerance = give_each_session_its_own_noise_tolerance(grouped)
    with update_config(str(grouped / "config.yaml")) as config:
        config["cluster_type"] = "local"

    extract_found_sessions(str(grouped), str(grouped / "config.yaml"), [".avi"])

    assert received == {str(grouped / key / "depth.avi"): tolerance[key] for key in GROUPED}


def test_slurm_jobs_read_each_same_named_folders_parameters(grouped, monkeypatch):
    """The SLURM path writes each session's parameters into its own
    ``proc/config.yaml``, and each job extracts with that file."""
    monkeypatch.setattr(
        "moseq2_extract.helpers.extract.system",
        lambda *_: pytest.fail("extract_found_sessions submitted a job"),
    )
    received = {}
    monkeypatch.setattr(
        extract_cli,
        "extract_wrapper",
        lambda input_file, output_dir, config, num_frames=None, skip=False: received.update(
            {input_file: config["noise_tolerance"]}
        ),
    )
    tolerance = give_each_session_its_own_noise_tolerance(grouped)
    with update_config(str(grouped / "config.yaml")) as config:
        config["cluster_type"] = "slurm"

    extract_found_sessions(str(grouped), str(grouped / "config.yaml"), [".avi"])

    for key in GROUPED:
        assert read_yaml(str(grouped / key / "proc" / "config.yaml"))["noise_tolerance"] == tolerance[key], key
    script = (grouped / "extract_out.sh").read_text()
    for line in filter(str.strip, script.splitlines()):
        command = line.split("moseq2-extract extract", 1)[1].split(";", 1)[0]
        from click.testing import CliRunner

        result = CliRunner().invoke(extract_cli.cli, ["extract", *split_command(command)], catch_exceptions=False)
        assert result.exit_code == 0, result.output
    assert received == {str(grouped / key / "depth.avi"): tolerance[key] for key in GROUPED}


def test_extraction_reads_a_file_keyed_by_folder_name(tmp_path, monkeypatch):
    """Where every session folder's name is its own, ``session_config.yaml``
    is keyed by folder name, as it always was, and extraction reads it."""
    received = {}
    monkeypatch.setattr(
        "moseq2_extract.gui.extract_wrapper",
        lambda input_file, output_dir, config, num_frames=None, skip=False: received.update(
            {input_file: config["noise_tolerance"]}
        ),
    )
    root = tmp_path / "project"
    a = empty_session(root, "GroupA/mouse1")
    b = empty_session(root, "GroupB/mouse2")
    config = root / "config.yaml"
    generate_config_command(str(config), camera_type="k2")
    base = read_yaml(str(config))
    write_yaml({"mouse1": {**base, "noise_tolerance": 11}, "mouse2": {**base, "noise_tolerance": 12}}, str(root / "session_config.yaml"))
    with update_config(str(config)) as data:
        data["cluster_type"] = "local"
        data["session_config_path"] = str(root / "session_config.yaml")

    extract_found_sessions(str(root), str(config), [".dat"])

    assert received == {str(a): 11, str(b): 12}
