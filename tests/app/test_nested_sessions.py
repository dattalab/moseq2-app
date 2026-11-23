"""Projects whose sessions sit below group folders, and session folders that share a name.

``find_recordings`` (``moseq2_extract.recordings``) finds a project's
recordings for the progress cell, the arena tool and extraction. The tests
here follow what those cells do with what it finds:

- the notebook's Extract step, as the notebook ships it, takes every
  recording type the pipeline reads, ``.mkv`` included;
- the validation and preview cells find each recording's extraction in its
  session folder in ``aggregate_results/``, however deep the recording is;
- a session is named in the notebook's lists by its folder's name, or by its
  folder's path below the project when two session folders share a name
  (``GroupA/mouse1`` and ``GroupB/mouse1``);
- ``session_config.yaml`` is keyed by the recording's path below the project
  (``GroupA/mouse1/depth.avi``), so each keeps its own parameters through the
  arena tool, a local extraction and the SLURM jobs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import moseq2_extract.steps as extract_steps
from cells import code_cell
from moseq2_common.project import Project
from commands import split_command
from moseq2_common.slurm import job_settings_file
from moseq2_app.gui.progress import get_session_paths
from moseq2_app.roi.controller import InteractiveExtractionViewer
from moseq2_common.yaml import read_yaml, write_config, write_yaml
from moseq2_extract.gui import generate_config_command
from moseq2_extract.project import extract
from slurm_fakes import fake_slurm, submitted_lines  # noqa: F401 -- the fixture

EXTRACT_NOTEBOOK = "MoSeq2-Pipeline.ipynb"

#: The Extract step's cell, numbered as ``code_cell`` counts.
EXTRACTION_CELL = 3

#: A grouped project made from the fixture recordings: where each session
#: folder sits below the project, and the recording copied into it. The two
#: ``mouse1`` folders share a name; ``saline_example_0`` is the only folder of
#: its name.
GROUPED = {
    "GroupA/mouse1": "amphetamine_example_0",
    "GroupB/mouse1": "amphetamine_example_1",
    "saline_example_0": "saline_example_0",
}


def session_folder(root, where):
    """The session folder in ``aggregate_results/`` of the one recording in ``root/where``."""
    return Path(root, "aggregate_results", where.replace("/", "_"))


def empty_session(root, where, recording="depth.dat", extracted=False):
    """A session folder at ``root/where`` holding an empty recording, and an empty extraction movie
    and flipped movie in its session folder in ``aggregate_results/`` when ``extracted``."""
    folder = Path(root, where)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / recording).touch()
    (folder / "metadata.json").write_text("{}")
    if extracted:
        outputs = session_folder(root, where)
        outputs.mkdir(parents=True)
        (outputs / "results.mp4").touch()
        (outputs / "results_flipped.mp4").touch()
    return folder / recording


@pytest.fixture
def chosen(monkeypatch):
    """The recordings extraction extracts, one at a time; nothing is extracted."""
    recorded = []
    monkeypatch.setattr("moseq2_extract.project.extract_recording",
                        lambda project, input_file, opts, **kw: recorded.append(input_file))
    return recorded


def set_config(root, extract=None, computing=None):
    """Set ``extract`` in ``root/config.yaml``'s extract section and ``computing`` in its
    ``computing.extract`` block."""
    data = read_yaml(str(root / "config.yaml"))
    data["extract"].update(extract or {})
    data["computing"]["extract"].update(computing or {})
    write_config(data, str(root / "config.yaml"))


def project_config(root, camera_type="k2", **extract):
    """``generate-config``'s ``root/config.yaml``, naming a head/tail classifier so nothing is downloaded."""
    generate_config_command(str(root / "config.yaml"), camera_type=camera_type)
    set_config(root, extract={"flip_classifier": str(root / "flip_classifier.npz"), **extract})
    return root / "config.yaml"


@pytest.fixture
def grouped(tmp_path, raw_sessions):
    """The ``GROUPED`` project, with a config from ``generate-config`` naming the fixture's head/tail
    classifier."""
    root = tmp_path / "project"
    for where, source in GROUPED.items():
        shutil.copytree(Path(raw_sessions) / source, root / where)
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    set_config(root, extract={"flip_classifier": str(Path(raw_sessions) / "flip_classifier_k2_c57_10to13weeks.npz")})
    return root


def open_arena_tool(root):
    from moseq2_app.roi.widget import ArenaMaskWidget

    return ArenaMaskWidget(str(root))


def give_each_session_its_own_noise_tolerance(root):
    """Write a ``session_config.yaml`` keyed as the arena tool keys ``GROUPED``
    (see ``test_the_arena_tool_keeps_parameters_for_each_same_named_folder``),
    with a different ``noise_tolerance`` (a setting the arena tool saves) for
    each session, the way a user tuning each recording leaves them different;
    return the noise tolerance by session."""
    tolerance = {key: 11 + number for number, key in enumerate(sorted(GROUPED))}
    write_yaml({f"{key}/depth.avi": {"noise_tolerance": value} for key, value in tolerance.items()},
               str(root / "session_config.yaml"))
    return tolerance


# ------------------------------------------------------------------ the Extract step


def test_extraction_takes_every_recording_type_by_default(tmp_path, chosen):
    """With the ``extensions`` ``generate-config`` writes, extraction takes every type
    ``find_recordings`` knows: ``.dat``, ``.mkv`` (any name), ``.avi`` and ``.tar.gz``."""
    root = tmp_path / "project"
    expected = sorted(str(p) for p in (
        empty_session(root, "kinect"),
        empty_session(root, "azure", recording="2024-05-01-10-31-07.mkv"),
        empty_session(root, "converted", recording="depth.avi"),
        empty_session(root, "archived", recording="session_20190320.tar.gz"),
    ))
    project_config(root)

    extract(Project.at(root))

    assert sorted(chosen) == expected


def test_an_extensions_setting_is_obeyed(tmp_path, chosen, capsys):
    """An ``extensions`` setting that leaves out ``.mkv`` gets what it asks for, and the
    message names the ``.mkv`` passed over."""
    root = tmp_path / "project"
    dat = empty_session(root, "kinect")
    mkv = empty_session(root, "azure", recording="recording.mkv")
    project_config(root, extensions=[".avi", ".dat", ".tar.gz"])
    capsys.readouterr()

    extract(Project.at(root))

    assert chosen == [str(dat)]
    assert f"{mkv.parent}: holds recording.mkv" in capsys.readouterr().err


def test_the_notebooks_extraction_cell_extracts_an_mkv(tmp_path, chosen):
    """The Extract cell as the notebook ships it, on a project of Azure or Femto ``.mkv``
    recordings and a Kinect v2 ``.dat``, with no edit by the user: the cell opens the
    Extract GUI in the notebook's folder, and its button extracts both."""
    import ast
    import os

    root = tmp_path / "project"
    expected = sorted(str(p) for p in (
        empty_session(root, "GroupA/femto_0", recording="2024-05-01-10-31-07.mkv"),
        empty_session(root, "GroupB/kinect_0"),
    ))
    project_config(root, camera_type="azure")

    tree = ast.parse(code_cell(EXTRACT_NOTEBOOK, EXTRACTION_CELL))
    last = tree.body.pop()
    namespace = {}
    before = os.getcwd()
    os.chdir(root)
    try:
        exec(compile(tree, "<Extract cell>", "exec"), namespace)
        widget = eval(compile(ast.Expression(last.value), "<Extract cell>", "eval"), namespace)
        widget.start()
        run = widget.wait()
    finally:
        os.chdir(before)

    assert run["state"] == "finished", widget.problem
    assert sorted(chosen) == expected


# ------------------------------------------------------------- extractions two levels down


def test_extractions_two_levels_down_are_listed(tmp_path):
    """Validation and preview list the movie in each recording's session folder, at any depth the
    recordings are found at."""
    root = tmp_path / "project"
    for where in ("GroupA/mouse1", "GroupB/mouse2", "solo"):
        empty_session(root, where, extracted=True)
    empty_session(root, "GroupB/not_yet_extracted")

    extracted = get_session_paths(str(root), extracted=True)
    flipped = get_session_paths(str(root), extracted=True, flipped=True)

    assert extracted == {
        name: str(session_folder(root, where) / "results.mp4")
        for name, where in (("mouse1", "GroupA/mouse1"), ("mouse2", "GroupB/mouse2"), ("solo", "solo"))
    }
    assert flipped == {
        name: str(session_folder(root, where) / "results_flipped.mp4")
        for name, where in (("mouse1", "GroupA/mouse1"), ("mouse2", "GroupB/mouse2"), ("solo", "solo"))
    }


def test_an_extraction_whose_recording_was_removed_is_still_listed(tmp_path):
    """A session moseq2-project.yaml lists keeps its extraction listed after its recording was moved
    away, named by its session key."""
    root = tmp_path / "project"
    empty_session(root, "s0", extracted=True)
    removed = empty_session(root, "s1", extracted=True)
    write_yaml({"sessions": {"s1/depth.dat": {"recording": "s1/depth.dat", "folder": "aggregate_results/s1",
                                              "uuid": "u1"}}}, str(root / "moseq2-project.yaml"))
    removed.unlink()

    assert get_session_paths(str(root), extracted=True) == {
        "s0": str(session_folder(root, "s0") / "results.mp4"),
        "s1/depth.dat": str(session_folder(root, "s1") / "results.mp4"),
    }


def test_the_preview_offers_extractions_two_levels_down(tmp_path):
    root = tmp_path / "project"
    for where in ("GroupA/mouse1", "GroupB/mouse1", "solo"):
        empty_session(root, where, extracted=True)

    viewer = InteractiveExtractionViewer(str(root))

    assert dict(viewer.sess_select.options) == {
        where: str(session_folder(root, where) / "results.mp4") for where in ("GroupA/mouse1", "GroupB/mouse1", "solo")
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
        shutil.copytree(Path(extracted) / "aggregate_results" / source, session_folder(root, where))

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
    """The arena tool keys each recording by its path below the project, ``mouse1``
    folders included, and saving one leaves the others unsaved."""
    from widgets import buttons_of, click

    widget = open_arena_tool(grouped)

    assert widget.sessions == {f"{key}/depth.avi": str(grouped / key / "depth.avi") for key in GROUPED}

    data = widget.session_data
    data.path = "GroupA/mouse1/depth.avi"
    data.set_config_data(data.path)
    data.depth_range = (600, 700)
    click(buttons_of(widget.view)["Save session parameters"])

    saved = read_yaml(str(grouped / "session_config.yaml"))
    assert list(saved) == ["GroupA/mouse1/depth.avi"]
    assert list(saved["GroupA/mouse1/depth.avi"]["bg_roi_depth_range"]) == [600, 700]


def test_a_local_extraction_reads_each_same_named_folders_parameters(grouped, monkeypatch):
    received = {}
    monkeypatch.setattr(
        "moseq2_extract.steps.extract_wrapper",
        lambda input_file, output_dir, config, num_frames=None, skip=False, uuid=None, record=None: received.update(
            {input_file: config["noise_tolerance"]}
        ),
    )
    tolerance = give_each_session_its_own_noise_tolerance(grouped)
    set_config(grouped, extract={"extensions": [".avi"]}, computing={"cluster_type": "local"})

    extract(Project.at(grouped))

    assert received == {str(grouped / key / "depth.avi"): tolerance[key] for key in GROUPED}


def test_slurm_jobs_read_each_same_named_folders_parameters(grouped, monkeypatch, fake_slurm):
    """The SLURM path writes each session's parameters into its own job settings
    file in its session folder, and each job (``moseq2 run-job``) extracts with that file."""
    fake_slurm.hold()
    received = {}
    monkeypatch.setattr(
        extract_steps,
        "extract_wrapper",
        lambda input_file, output_dir, config, num_frames=None, skip=False, uuid=None, record=None: received.update(
            {input_file: config["noise_tolerance"]}
        ),
    )
    tolerance = give_each_session_its_own_noise_tolerance(grouped)
    set_config(grouped, extract={"extensions": [".avi"]}, computing={"cluster_type": "slurm"})

    extract(Project.at(grouped))

    assert received == {}, "a job ran before its sbatch line was"
    for key in GROUPED:
        assert read_yaml(str(job_settings_file(session_folder(grouped, key), "extract")))["extract"]["noise_tolerance"] == tolerance[key], key
    for line in submitted_lines(Project.at(grouped)):
        words = split_command(line)
        job = split_command(words[words.index("--wrap") + 1].split(";", 1)[0])
        from click.testing import CliRunner

        from moseq2_app.cli import cli

        result = CliRunner().invoke(cli, job[job.index("run-job"):], prog_name="moseq2", catch_exceptions=False)
        assert result.exit_code == 0, result.output
    assert received == {str(grouped / key / "depth.avi"): tolerance[key] for key in GROUPED}
