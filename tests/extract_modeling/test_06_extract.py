"""The Extract step: extract every recording in the project.

    from moseq2_app.notebook import extract
    extract()

The heaviest step, and the one every later step is built on. The notebook's
cell opens the Extract GUI; pressing its button runs
``moseq2_extract.project.extract``, which ``moseq2 extract`` runs too. It walks
the project for depth recordings of the types ``config.yaml``'s ``extensions``
names, plans each one (new, to redo, or up to date), lists every planned
recording as a session in ``moseq2-project.yaml``, and extracts what the plan
says on this computer or, with ``computing.extract.cluster_type`` set to
``slurm``, writes one job per recording and the script of ``sbatch`` lines that
runs them. What it leaves behind is a folder for each recording in
``aggregate_results/``, named from the recording's path in the project, holding
the cropped frames, the scalars computed from them, a preview movie and a
record of the parameters used. Nothing is written beside the recordings.

The tests split along that seam. The ones that only need to know which
recordings were chosen run against a directory of empty files with a stand-in
for the extraction of one recording, which costs nothing. The ones that need
real output use the chain's own extraction, which runs once for the whole
suite.

A recording that fails is recorded in the run's ``failed`` and the rest are
extracted, so the step finishes without raising; a test reads the run and the
files on disk.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from video import index_is_at_the_front, probe

from moseq2_common.project import Project
from moseq2_common.records import read_record
from moseq2_common.yaml import read_yaml, write_config

from test_03_config import plain
from test_05_arena_mask import RECORDINGS, SESSIONS

#: An ``extensions`` setting that leaves out ``.mkv``.
EXTENSIONS = [".avi", ".dat", ".tar.gz"]

#: What extraction writes into every recording's session folder.
OUTPUTS = (
    "bground.tiff",
    "first_frame.tiff",
    "results.h5",
    "results.mp4",
    "roi_00.tiff",
)

#: The measurements that read zero rather than NaN when no mouse was found,
#: because they are counted in pixels and nothing is a legitimate zero.
ZERO_WHEN_EMPTY = ("area_px", "height_ave_mm")

#: The measurements taken between frames, so one frame without a mouse costs
#: the frame after it as well.
VELOCITIES = (
    "velocity_2d_mm",
    "velocity_2d_px",
    "velocity_3d_mm",
    "velocity_3d_px",
    "velocity_theta",
)

#: Every per-frame measurement the rest of the pipeline reads.
SCALARS = (
    "angle",
    "area_mm",
    "area_px",
    "centroid_x_mm",
    "centroid_x_px",
    "centroid_y_mm",
    "centroid_y_px",
    "height_ave_mm",
    "length_mm",
    "length_px",
    "velocity_2d_mm",
    "velocity_2d_px",
    "velocity_3d_mm",
    "velocity_3d_px",
    "velocity_theta",
    "width_mm",
    "width_px",
)

#: Parameters the h5 records differently from the session config, so comparing
#: them says nothing about whether the right config was used.
NOT_COMPARABLE = ("config_file", "num_frames", "output_file", "detected_true_depth")


def proc(project, recording):
    """The session folder extraction writes one recording's outputs into."""
    return Path(project) / "aggregate_results" / recording


def set_settings(root, section, **values):
    """Set ``values`` in ``root``'s config.yaml: in ``section``, or, for ``computing``, its ``extract`` block."""
    config_file = Path(root) / "config.yaml"
    data = read_yaml(str(config_file))
    block = data["computing"]["extract"] if section == "computing" else data[section]
    block.update(values)
    write_config(data, str(config_file))


def build_fake_project(root, recordings=RECORDINGS, extension=".avi"):
    """A project of empty depth files, for testing which recordings get chosen.

    The step finds recordings with ``find_recordings``, which decides from
    filenames alone and never opens anything. The config is the one
    ``generate-config`` writes, naming a head/tail classifier so nothing is
    downloaded; the stand-in extraction never reads it.
    """
    from moseq2_extract.project import generate_config

    root = Path(root)
    for name in recordings:
        session = root / name
        session.mkdir(parents=True)
        (session / f"depth{extension}").touch()
        (session / "metadata.json").write_text("{}")
    generate_config(str(root / "config.yaml"), "auto", overwrite=True)
    set_settings(root, "extract", flip_classifier=str(root / "flip_classifier.npz"))
    return root


def run_extraction(root, **kwargs):
    """The Extract step's run in the project at ``root``."""
    from moseq2_extract.project import extract

    return extract(Project.at(root), **kwargs)


@pytest.fixture
def chosen(monkeypatch):
    """Record the recordings the step extracts, and extract none.

    Returns the list, which fills in as the run extracts each recording.
    """
    recorded = []

    def record(project, input_file, opts, **kwargs):
        recorded.append(input_file)

    monkeypatch.setattr("moseq2_extract.project.extract_recording", record)
    return recorded


def test_every_recording_is_found(tmp_path, chosen):
    """1. The step's search: one depth file per recording, in sorted order."""
    root = build_fake_project(tmp_path / "project")
    run_extraction(root)

    assert chosen == sorted(str(root / name / "depth.avi") for name in RECORDINGS)


def test_only_the_named_extensions_are_searched(tmp_path, chosen):
    """2. A recording in a format ``extensions`` leaves out is left alone."""
    root = build_fake_project(tmp_path / "project", extension=".mkv")
    set_settings(root, "extract", extensions=EXTENSIONS)
    run_extraction(root)
    assert chosen == []

    set_settings(root, "extract", extensions=[".mkv"])
    run_extraction(root)
    assert len(chosen) == len(RECORDINGS)


def test_the_nidaq_stream_is_not_a_recording(tmp_path, chosen):
    """With its NI-DAQ stream on, the Kinect v2 app writes ``nidaq.dat`` beside
    ``depth.dat``; only the depth file is a recording."""
    root = build_fake_project(tmp_path / "project", extension=".dat")
    for name in RECORDINGS:
        (root / name / "nidaq.dat").touch()
    run_extraction(root)

    assert chosen == sorted(str(root / name / "depth.dat") for name in RECORDINGS)


def test_a_missing_config_is_reported_before_the_search(tmp_path, chosen):
    """3. The step's first check: ``moseq2 extract`` in a folder of recordings with no config.yaml stops,
    naming the folder, and extracts nothing."""
    from click.testing import CliRunner

    from moseq2_app.cli import cli

    root = build_fake_project(tmp_path / "project")
    (root / "config.yaml").unlink()
    before = os.getcwd()
    os.chdir(root)
    try:
        result = CliRunner().invoke(cli, ["extract"], prog_name="moseq2")
    finally:
        os.chdir(before)

    assert result.exit_code == 1, result.output
    assert f"{root} has no config.yaml" in result.stderr
    assert chosen == []


@pytest.mark.skipif(not os.path.isdir("/tmp"), reason="no /tmp on this platform")
def test_recordings_under_tmp_are_extracted(chosen):
    """4. Where a project lives is not the step's business.

    Recordings under ``/tmp``, or under any path with a ``tmp`` component
    such as ``/data/tmp/pilot``, are extracted like any others.
    """
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        root = build_fake_project(Path(tmp) / "project")
        run_extraction(root)

    assert len(chosen) == len(RECORDINGS)


@pytest.mark.parametrize(
    "named",
    [
        [RECORDINGS[0], RECORDINGS[2]],
        [RECORDINGS[0], RECORDINGS[3]],
        RECORDINGS[:2],
        RECORDINGS[2:],
    ],
)
def test_naming_recordings_extracts_only_those(tmp_path, chosen, named):
    """5. What a user does when they do not want all of them: name the recordings
    (``moseq2 extract RECORDING ...``), each by its path or its session key."""
    root = build_fake_project(tmp_path / "project")

    first, *rest = named
    run_extraction(root, recordings=[str(root / first / "depth.avi"), *(f"{name}/depth.avi" for name in rest)])

    assert chosen == [str(root / name / "depth.avi") for name in named]


def test_the_slurm_branch_writes_a_script_and_submits_nothing(tmp_path, chosen, monkeypatch):
    """6. The other half of the dispatch.

    With ``computing.extract.cluster_type`` set to ``slurm``, the step writes
    each recording's job settings file into its session folder and one
    ``sbatch`` line per recording, running ``moseq2 run-job`` on that file, into
    ``extract_out.sh``, and stops. Nothing reaches the scheduler.
    """
    import subprocess

    root = build_fake_project(tmp_path / "project", extension=".dat")
    set_settings(root, "computing", cluster_type="slurm")
    real_system, real_popen = os.system, subprocess.Popen

    def refuse_sbatch(run):
        def guarded(command, *args, **kwargs):
            words = command if isinstance(command, (list, tuple)) else [command]
            if any("sbatch" in str(word) for word in words):
                pytest.fail(f"the slurm branch submitted a job: {command}")
            return run(command, *args, **kwargs)
        return guarded

    monkeypatch.setattr(os, "system", refuse_sbatch(real_system))
    monkeypatch.setattr(subprocess, "Popen", refuse_sbatch(real_popen))

    run = run_extraction(root)

    script = (root / "extract_out.sh").read_text()
    assert script.count("sbatch") == len(RECORDINGS)
    assert script.count("run-job") == len(RECORDINGS)
    assert sorted(run.jobs) == [f"{name}/depth.dat" for name in RECORDINGS]
    for key, job_file in run.jobs.items():
        assert Path(job_file).parent == proc(root, key.split("/")[0])
        assert str(job_file) in script
        assert read_yaml(str(job_file))["job"]["recording"] == key
    assert chosen == [], "the local extractor ran on the slurm branch"


def test_every_recording_is_extracted(extracted):
    """7. The step's output: a full session folder for each recording, and nothing beside it."""
    for name in RECORDINGS:
        written = sorted(p.name for p in proc(extracted, name).iterdir())
        assert written == sorted(OUTPUTS), name
        assert not (Path(extracted) / name / "proc").exists(), name


def test_the_extracted_frames_are_a_cropped_movie_of_the_mouse(extracted):
    """8. The frames every later step reads.

    Cropped to the configured size, one frame per frame of the recording, and
    paired with the mask that produced them.
    """
    for name in RECORDINGS:
        with h5py.File(proc(extracted, name) / "results.h5") as f:
            assert f["frames"].shape == (600, 80, 80)
            assert f["frames"].dtype == np.uint8
            assert f["frames_mask"].shape == f["frames"].shape
            assert f["frames_mask"].dtype == bool
            assert f["timestamps"].shape == (600,)
            assert np.all(np.diff(f["timestamps"][:]) > 0), "timestamps go backwards"


def test_a_mouse_is_found_in_all_but_the_opening_frames(extracted):
    """9. The frames hold a mouse rather than correctly shaped emptiness.

    ``get_bground_im_file`` takes the median of the frames it samples at a
    stride of 1000, so a 600-frame recording gets a background built from
    frame 0 alone, carrying that frame's mouse with it. Wherever the mouse has
    not yet moved away from where it stood in frame 0, it cancels against the
    background and the extraction finds nothing. That costs the opening three
    to five frames of each recording, and three more in the middle of one of
    them where the mouse walked back through the same spot.

    It is a property of recordings this short rather than of the extraction:
    the walkthrough's 35964-frame recordings build the background from 36
    frames, where no single pose survives the median.
    """
    for name in RECORDINGS:
        with h5py.File(proc(extracted, name) / "results.h5") as f:
            found = (f["frames"][:] > 0).sum(axis=(1, 2))

        empty = np.flatnonzero(found == 0)
        assert len(empty) <= 10, f"{name}: {len(empty)} frames without a mouse"
        assert empty.max(initial=0) < 60, (
            f"{name}: a frame is empty at {empty.max()}, past the opening seconds, "
            "so this is no longer the background carrying frame 0's mouse"
        )
        assert found[found > 0].mean() > 100, (
            f"{name}: mouse averages {found[found > 0].mean():.0f} pixels"
        )


def test_every_scalar_is_measured(extracted):
    """10. The seventeen per-frame measurements the modeling and analysis read.

    Which frames carry a value and which do not is the next test's business.
    """
    for name in RECORDINGS:
        with h5py.File(proc(extracted, name) / "results.h5") as f:
            assert sorted(f["scalars"]) == sorted(SCALARS), name
            for scalar in SCALARS:
                values = f[f"scalars/{scalar}"]
                assert values.shape == (600,), f"{name}: {scalar}"
                assert values.dtype == np.float32, f"{name}: {scalar}"


def test_a_frame_without_a_mouse_has_no_measurements(extracted):
    """11. The scalars and the frames agree about which frames hold a mouse.

    Twelve of the measurements are NaN where nothing was found. The two
    counted in pixels read zero instead, since zero pixels is what was
    measured. The five velocities are differences between frames, so an empty
    frame takes the frame after it with it.
    """
    for name in RECORDINGS:
        with h5py.File(proc(extracted, name) / "results.h5") as f:
            empty = (f["frames"][:] > 0).sum(axis=(1, 2)) == 0
            after = np.concatenate([[False], empty[:-1]])
            for scalar in SCALARS:
                values = f[f"scalars/{scalar}"][:]
                missing = ~np.isfinite(values)
                if scalar in ZERO_WHEN_EMPTY:
                    assert not missing.any(), f"{name}: {scalar} went NaN"
                    assert (values[empty] == 0).all(), f"{name}: {scalar}"
                    continue
                expected = empty | after if scalar in VELOCITIES else empty
                np.testing.assert_array_equal(
                    missing, expected, err_msg=f"{name}: {scalar}"
                )


def test_each_recording_reports_itself_finished(extracted):
    """12. How the next run, and the project's progress report, know this is done: ``results.h5`` exists
    under its final name, with the extraction's record, and nothing is left under a temporary name."""
    for name in RECORDINGS:
        folder = proc(extracted, name)
        record = read_record(folder / "results.h5")
        assert record["step"] == "extract"
        assert record["ended"]
        assert not list(folder.glob(".*partial*"))


def test_extraction_ran_with_the_parameters_the_widget_saved(extracted, arena):
    """13. The contract between the arena tool and this step.

    Extraction reads the project's ``session_config.yaml`` and takes the
    settings the arena widget sets from the entry belonging to the recording,
    so what a user settled on in the widget is literally what the extraction
    ran with. The extraction's record holds that set, which is where the two
    are compared; a depth range the extraction detected (the entry not setting
    ``manual_set_depth_range``) is the detected one.
    """
    saved = plain(read_yaml(str(arena)))
    for name in RECORDINGS:
        used = plain(read_record(proc(extracted, name) / "results.h5")["options"])
        for key, value in saved[f"{name}/depth.avi"].items():
            if key in NOT_COMPARABLE or key not in used:
                continue
            if key == "bg_roi_depth_range" and not used["manual_set_depth_range"]:
                continue
            recorded = used[key]
            if isinstance(recorded, bytes):
                recorded = recorded.decode()
            assert recorded == value, f"{name}: {key} was {recorded!r}, saved {value!r}"


def test_the_preview_movie_is_h264(extracted):
    """14a. The movie a user opens to see whether extraction worked.

    Written as H.264 with 4:2:0 chroma, which is what a browser will decode.
    """
    for name in RECORDINGS:
        codec, pix_fmt = probe(proc(extracted, name) / "results.mp4")
        assert (codec, pix_fmt) == ("h264", "yuv420p"), name


def test_the_preview_movie_starts_without_reading_all_of_it(extracted):
    """14b. The index belongs at the front.

    With the moov atom at the end a browser has to fetch the whole file before
    it can show the first frame or seek, which is the difference between a
    preview that opens and one that appears to hang.
    """
    for name in RECORDINGS:
        assert index_is_at_the_front(proc(extracted, name) / "results.mp4"), name


def test_rerunning_the_step_extracts_nothing_again(extracted):
    """15. A second run is free: reopened, the step shows every recording up to date, and pressing its
    button extracts none of them."""
    from moseq2_app.notebook import extract

    before = {
        name: proc(extracted, name).joinpath("results.h5").stat().st_mtime_ns
        for name in RECORDINGS
    }

    cwd = os.getcwd()
    os.chdir(extracted)
    try:
        widget = extract()
        plans = {row["id"]: row["plan"] for row in widget.recordings}
        widget.start()
        run = widget.wait()
    finally:
        os.chdir(cwd)

    assert plans == {key: "done" for key in SESSIONS}
    assert run["state"] == "finished", widget.problem
    assert run["items"] == {}
    for name in RECORDINGS:
        assert (
            proc(extracted, name).joinpath("results.h5").stat().st_mtime_ns
            == before[name]
        ), f"{name} was extracted again"
