"""Cell 6: extract every recording in the project.

    with update_config(progress_paths['config_file']) as config_data:
        config_data['cluster_type'] = 'local'

    extensions = ['.avi', '.dat', '.tar.gz']
    extract_all = True

    extract_found_sessions(progress_paths['base_dir'],
                           progress_paths['config_file'],
                           extensions,
                           extract_all=extract_all,
                           skip_extracted=skip_extracted)

The heaviest cell, and the one every later cell is built on. It does four
things: walk the project for depth files with any of the given extensions,
narrow that list to what the user wants, dispatch on ``cluster_type``, and
extract. What it leaves behind is a ``proc`` directory beside each recording
holding the cropped frames, the scalars computed from them, a preview movie
and a record of the parameters used.

The tests split along that seam. The ones that only need to know which
recordings were chosen run against a directory of empty files with a stand-in
for the extraction, which costs nothing. The ones that need real output use
the chain's own extraction, which runs once for the whole suite.

Two things make the cell quieter than it looks. ``run_local_extract`` catches
every exception a recording raises, prints it and moves to the next, so the
cell finishes and prints ``Extractions Complete.`` whether or not anything was
extracted. And the recordings it extracts have already been filtered twice, so
a project that yields nothing is indistinguishable from one that extracted
everything. Both mean a test has to look at the files on disk.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from video import index_is_at_the_front, probe

from moseq2_extract.gui import extract_found_sessions
from moseq2_extract.util import read_yaml

from test_03_config import plain
from test_05_arena_mask import RECORDINGS

#: The extensions the notebook searches for.
EXTENSIONS = [".avi", ".dat", ".tar.gz"]

#: What extraction writes beside every recording.
OUTPUTS = (
    "bground.tiff",
    "first_frame.tiff",
    "results_00.h5",
    "results_00.mp4",
    "results_00.yaml",
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
    """The directory extraction writes beside one recording."""
    return Path(project) / recording / "proc"


def build_fake_project(root, recordings=RECORDINGS, extension=".avi"):
    """A project of empty depth files, for testing which recordings get chosen.

    ``recursive_find_unextracted_dirs`` is called with ``skip_checks=True``, so
    it decides from filenames alone and never opens anything.
    """
    root = Path(root)
    for name in recordings:
        session = root / name
        session.mkdir(parents=True)
        (session / f"depth{extension}").touch()
        (session / "metadata.json").write_text("{}")
    (root / "config.yaml").write_text("cluster_type: local\n")
    return root


@pytest.fixture
def chosen(monkeypatch):
    """Record the recordings handed to the extraction loop, and extract none.

    Returns the list, which fills in when ``extract_found_sessions`` is called.
    """
    recorded = []

    def record(to_extract, config_file, skip_extracted=False):
        recorded.extend(to_extract)

    monkeypatch.setattr("moseq2_extract.gui.run_local_extract", record)
    return recorded


def test_every_recording_is_found(tmp_path, chosen):
    """1. The cell's search: one depth file per recording, in sorted order."""
    root = build_fake_project(tmp_path / "project")
    extract_found_sessions(str(root), str(root / "config.yaml"), EXTENSIONS)

    assert chosen == sorted(str(root / name / "depth.avi") for name in RECORDINGS)


def test_only_the_named_extensions_are_searched(tmp_path, chosen):
    """2. A recording in a format the user did not ask for is left alone."""
    root = build_fake_project(tmp_path / "project", extension=".mkv")
    extract_found_sessions(str(root), str(root / "config.yaml"), EXTENSIONS)
    assert chosen == []

    extract_found_sessions(str(root), str(root / "config.yaml"), [".mkv"])
    assert len(chosen) == len(RECORDINGS)


def test_a_missing_config_is_reported_before_the_search(tmp_path, chosen):
    """3. The cell's one explicit error."""
    root = build_fake_project(tmp_path / "project")
    with pytest.raises(IOError, match="does not exist"):
        extract_found_sessions(str(root), str(root / "nope.yaml"), EXTENSIONS)
    assert chosen == []


@pytest.mark.skipif(not os.path.isdir("/tmp"), reason="no /tmp on this platform")
def test_recordings_under_tmp_are_extracted(chosen):
    """4. Where a project lives is not the cell's business.

    Recordings under ``/tmp``, or under any path with a ``tmp`` component
    such as ``/data/tmp/pilot``, are extracted like any others.
    """
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        root = build_fake_project(Path(tmp) / "project")
        extract_found_sessions(str(root), str(root / "config.yaml"), EXTENSIONS)

    assert len(chosen) == len(RECORDINGS)


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("1,3", [RECORDINGS[0], RECORDINGS[2]]),
        ("1-2", RECORDINGS[:2]),
        ("e1-2", RECORDINGS[2:]),
    ],
)
def test_selecting_a_subset_of_recordings(tmp_path, chosen, monkeypatch, typed, expected):
    """5. What a user can type when they do not want all of them.

    With ``extract_all=False`` the cell lists the recordings and asks. Commas
    pick individual ones, a hyphen gives an inclusive range, and an ``e``
    prefix excludes a range instead. The indices shown are one-based.
    """
    root = build_fake_project(tmp_path / "project")
    monkeypatch.setattr("builtins.input", lambda *_: typed)

    extract_found_sessions(
        str(root), str(root / "config.yaml"), EXTENSIONS, extract_all=False
    )

    assert chosen == [str(root / name / "depth.avi") for name in expected]


def test_an_empty_answer_asks_again(tmp_path, chosen, monkeypatch):
    """The prompt keeps asking until it is given something it understands.

    An empty answer prints `Invalid input. Try again or press q to quit.` and
    loops, so a user who hits return by accident is not taken to mean
    everything, or nothing.
    """
    root = build_fake_project(tmp_path / "project")
    answers = iter(["", "1"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    extract_found_sessions(
        str(root), str(root / "config.yaml"), EXTENSIONS, extract_all=False
    )
    assert chosen == [str(root / RECORDINGS[0] / "depth.avi")]


def test_quitting_the_prompt_extracts_nothing(tmp_path, chosen, monkeypatch):
    """6. The way out of the prompt."""
    root = build_fake_project(tmp_path / "project")
    monkeypatch.setattr("builtins.input", lambda *_: "q")

    extract_found_sessions(
        str(root), str(root / "config.yaml"), EXTENSIONS, extract_all=False
    )
    assert chosen == []


@pytest.mark.parametrize("typo", ["first", "1,x", "1 2"])
def test_an_unreadable_selection_asks_again(tmp_path, chosen, monkeypatch, capsys, typo):
    """7. A typo at the prompt costs a user one more line, not the cell.

    An answer that is not a number, a list or a range prints ``Invalid input.
    Try again or press q to quit.`` and asks again, and nothing from it carries
    over into the next answer.
    """
    root = build_fake_project(tmp_path / "project")
    answers = iter([typo, "2"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))

    extract_found_sessions(
        str(root), str(root / "config.yaml"), EXTENSIONS, extract_all=False
    )
    assert chosen == [str(root / RECORDINGS[1] / "depth.avi")]
    assert "Invalid input. Try again or press q to quit." in capsys.readouterr().out


def test_the_slurm_branch_writes_a_script_and_submits_nothing(
    tmp_path, chosen, monkeypatch
):
    """8. The other half of the dispatch, which the notebook leaves commented out.

    Setting ``cluster_type`` to anything but ``local`` writes one ``sbatch``
    line per recording into ``extract_out_script`` and stops. Submitting is
    behind ``run_cmd``, which is off by default, so running the cell this way
    does not reach the scheduler.
    """
    root = build_fake_project(tmp_path / "project")
    (root / "config.yaml").write_text("cluster_type: slurm\n")
    monkeypatch.setattr(
        "moseq2_extract.helpers.extract.system",
        lambda *_: pytest.fail("the slurm branch submitted a job"),
    )

    extract_found_sessions(str(root), str(root / "config.yaml"), EXTENSIONS)

    script = (root / "extract_out.sh").read_text()
    assert script.count("sbatch") == len(RECORDINGS)
    assert script.count("moseq2-extract extract") == len(RECORDINGS)
    for name in RECORDINGS:
        assert str(root / name / "depth.avi") in script
    assert chosen == [], "the local extractor ran on the slurm branch"


def test_every_recording_is_extracted(extracted):
    """9. The cell's output: a full proc directory beside each recording."""
    for name in RECORDINGS:
        written = sorted(p.name for p in proc(extracted, name).iterdir())
        assert written == sorted(OUTPUTS), name


def test_the_extracted_frames_are_a_cropped_movie_of_the_mouse(extracted):
    """10. The frames every later cell reads.

    Cropped to the configured size, one frame per frame of the recording, and
    paired with the mask that produced them.
    """
    for name in RECORDINGS:
        with h5py.File(proc(extracted, name) / "results_00.h5") as f:
            assert f["frames"].shape == (600, 80, 80)
            assert f["frames"].dtype == np.uint8
            assert f["frames_mask"].shape == f["frames"].shape
            assert f["frames_mask"].dtype == bool
            assert f["timestamps"].shape == (600,)
            assert np.all(np.diff(f["timestamps"][:]) > 0), "timestamps go backwards"


def test_a_mouse_is_found_in_all_but_the_opening_frames(extracted):
    """11. The frames hold a mouse rather than correctly shaped emptiness.

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
        with h5py.File(proc(extracted, name) / "results_00.h5") as f:
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
    """12. The seventeen per-frame measurements the modeling and analysis read.

    Which frames carry a value and which do not is the next test's business.
    """
    for name in RECORDINGS:
        with h5py.File(proc(extracted, name) / "results_00.h5") as f:
            assert sorted(f["scalars"]) == sorted(SCALARS), name
            for scalar in SCALARS:
                values = f[f"scalars/{scalar}"]
                assert values.shape == (600,), f"{name}: {scalar}"
                assert values.dtype == np.float32, f"{name}: {scalar}"


def test_a_frame_without_a_mouse_has_no_measurements(extracted):
    """13. The scalars and the frames agree about which frames hold a mouse.

    Twelve of the measurements are NaN where nothing was found. The two
    counted in pixels read zero instead, since zero pixels is what was
    measured. The five velocities are differences between frames, so an empty
    frame takes the frame after it with it.
    """
    for name in RECORDINGS:
        with h5py.File(proc(extracted, name) / "results_00.h5") as f:
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
    """14. How the next run, and cell 2's progress report, know this is done."""
    for name in RECORDINGS:
        results = plain(read_yaml(str(proc(extracted, name) / "results_00.yaml")))
        assert results["complete"] is True
        assert results["skip"] is False
        with h5py.File(proc(extracted, name) / "results_00.h5") as f:
            assert results["uuid"] == f["metadata/uuid"][()].decode()


def test_extraction_ran_with_the_parameters_the_widget_saved(extracted, arena):
    """15. The contract between cell 5 and this one.

    ``extract_command`` follows ``session_config_path`` out of the project
    config and replaces the whole parameter set with the entry belonging to
    the recording, so what a user settled on in the arena widget is literally
    what the extraction ran with. The h5 records that set, which is where the
    two are compared.
    """
    saved = plain(read_yaml(str(arena)))
    for name in RECORDINGS:
        with h5py.File(proc(extracted, name) / "results_00.h5") as f:
            used = {
                key: plain(value[()].tolist() if value.shape else value[()])
                for key, value in f["metadata/extraction/parameters"].items()
            }
        for key, value in saved[name].items():
            if key in NOT_COMPARABLE or key not in used:
                continue
            recorded = used[key]
            if isinstance(recorded, bytes):
                recorded = recorded.decode()
            assert recorded == value, f"{name}: {key} was {recorded!r}, saved {value!r}"


def test_the_preview_movie_is_h264(extracted):
    """16a. The movie a user opens to see whether extraction worked.

    Written as H.264 with 4:2:0 chroma, which is what a browser will decode.
    """
    for name in RECORDINGS:
        codec, pix_fmt = probe(proc(extracted, name) / "results_00.mp4")
        assert (codec, pix_fmt) == ("h264", "yuv420p"), name


@pytest.mark.xfail(
    reason="the movie is written without -movflags +faststart, so its moov atom "
    "is appended after the frame data"
)
def test_the_preview_movie_starts_without_reading_all_of_it(extracted):
    """16b. The index belongs at the front.

    With the moov atom at the end a browser has to fetch the whole file before
    it can show the first frame or seek, which is the difference between a
    preview that opens and one that appears to hang.
    """
    for name in RECORDINGS:
        assert index_is_at_the_front(proc(extracted, name) / "results_00.mp4"), name


def test_rerunning_the_cell_extracts_nothing_again(extracted, configured):
    """17. ``skip_extracted`` carries over from cell 5, so a second run is free."""
    before = {
        name: proc(extracted, name).joinpath("results_00.h5").stat().st_mtime_ns
        for name in RECORDINGS
    }

    extract_found_sessions(
        str(extracted),
        str(configured),
        EXTENSIONS,
        extract_all=True,
        skip_extracted=True,
    )

    for name in RECORDINGS:
        assert (
            proc(extracted, name).joinpath("results_00.h5").stat().st_mtime_ns
            == before[name]
        ), f"{name} was extracted again"
