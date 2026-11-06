"""Cell 5: find each recording's arena floor and save its extraction parameters.

    ArenaMaskWidget(str(project.folder), skip_extracted=skip_extracted)

The first interactive cell. Everything up to here produced one configuration
for the whole project; this cell produces one per recording, because the arena
sits at a different depth in every rig and often moves between recordings.

Opening the widget writes nothing: it shows each recording's saved parameters
over the configuration, or, for a recording with none saved, the configuration
with ``detect_and_set_camera_parameters`` run over it. The widget lets a user
look at the floor it found, adjust the parameters, and save; saving writes the
arena tool's settings for that recording, and only that recording, into
``session_config.yaml`` in the project folder, keyed by the recording's path
below the project. Running the cell again shows what was saved.

The widget is a Panel object, so the tests drive it the way the browser does:
they set a parameter, or assign a button's ``value``, and the callbacks run in
process. What that does not cover is the rendering. The layout's plotting half
is a holoviews ``DynamicMap``, and whether bokeh draws it in a notebook is not
something a test without a browser can see.

Two properties of the fixture recordings are worth knowing when reading the
numbers below. The background a mask is cut from is the median of the frames
``get_bground_im_file`` samples at a stride of 1000, so a 600-frame recording
yields a background built from a single frame, where the walkthrough's
35964-frame recordings use 36. It is enough to find the floor, which is static.
And the parameters the widget detects for these recordings are the generated
config's, bar one: the walkthrough's own session config holds the same values,
because there was nothing in it that needed tuning.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from moseq2_app.roi.widget import ArenaMaskWidget
from moseq2_app.util import update_config
from moseq2_common.yaml import read_yaml

from test_03_config import plain
from widgets import buttons_of, click

#: The four recordings' folders.
RECORDINGS = [
    "amphetamine_example_0",
    "amphetamine_example_1",
    "saline_example_0",
    "saline_example_1",
]

#: The four sessions, keyed by recording, in the order the selector offers them.
SESSIONS = [f"{name}/depth.avi" for name in RECORDINGS]

#: What the widget offers to do, by the label on the button.
ACTIONS = [
    "Compute arena mask",
    "Compute extraction",
    "Save session parameters",
    "Save session parameters and move to next",
]

#: Parameters holding a filesystem path, which differs every run.
PATH_KEYS = ("config_file", "flip_classifier", "output_file", "session_config_path")

#: Every parameter the save button writes, and the widget parameter it comes
#: from. A value that is a pair is written as ``(low, high)`` of one range.
SAVED_BY = {
    "bg_roi_depth_range": "depth_range",
    "bg_roi_index": "mask_index",
    "bg_roi_shape": "mask_shape",
    "cable_filter_iters": "cable_filters",
    "cable_filter_shape": "cable_filter_shape",
    "chunk_overlap": "chunk_overlap",
    "compress": "compress",
    "dilate_iterations": "mask_dilations",
    "flip_classifier_smoothing": "flip_classifier_smoothing",
    "frame_dtype": "frame_dtype",
    "movie_dtype": "movie_dtype",
    "noise_tolerance": "noise_tolerance",
    "pixel_format": "pixel_format",
    "tail_filter_iters": "tail_filters",
    "tail_filter_shape": "tail_filter_shape",
    "use_tracking_model": "tracking_model_flag",
}


def parameters(session_config, recording="saline_example_0/depth.avi"):
    """One recording's saved parameters, as plain yaml types."""
    return plain(read_yaml(str(session_config)))[recording]


def open_widget(root):
    """Open the widget on a project, as the cell does, and index its buttons.

    Extracted recordings are listed too: a copy of the chain's project may be taken after
    another test file has extracted it, and the cell's own ``skip_extracted`` would then
    show none.
    """
    widget = ArenaMaskWidget(str(root), skip_extracted=False)
    return widget, widget.session_data, buttons_of(widget.view)


@pytest.fixture
def scratch(tmp_path, project, arena):
    """A copy of the project, for every test that opens the widget.

    The chain's own copy holds what the walkthrough saved, and cell 6 extracts
    from it. The tests below save tuned parameters, which would change what
    the chain extracts with, before or after the chain's extraction depending
    on the order the tests run in. So every test that opens the widget works
    on its own copy. Copying costs a fraction of a second and the backgrounds
    come with it already computed.
    """
    root = tmp_path / "project"
    shutil.copytree(project, root)
    return root


def test_every_recording_gets_a_parameter_file(arena, project):
    """1. The cell's output: one entry per recording saved, keyed by session.

    A recording is a folder holding a depth video. Later cells add folders of
    their own to the project (``aggregate_results/``, ``_pca/``, ``models/``),
    and the shared chain may have reached them before this test runs.
    """
    saved = read_yaml(str(arena))
    assert sorted(saved) == SESSIONS
    assert sorted(p.relative_to(project).as_posix() for p in project.glob("*/depth.avi")) == SESSIONS


def test_detected_parameters_are_unchanged(arena, data_regression):
    """2. Every arena-tool setting a recording's entry holds, pinned.

    These are the values extraction takes from the entry over config.yaml, and
    they come from a dependency's detection routine rather than from anything
    written down, so a change in moseq2-extract shows up here as a change to
    every user's extraction.
    """
    saved = parameters(arena)
    data_regression.check({k: v for k, v in sorted(saved.items()) if k not in PATH_KEYS})


def test_every_recording_gets_the_same_parameters(arena):
    """3. Detection is per recording, and these four recordings agree."""
    saved = plain(read_yaml(str(arena)))
    reference = {k: v for k, v in saved[SESSIONS[0]].items() if k not in PATH_KEYS}
    for name in SESSIONS[1:]:
        assert {
            k: v for k, v in saved[name].items() if k not in PATH_KEYS
        } == reference, f"{name} was detected differently"


def test_saved_parameters_differ_from_the_configuration_in_one_place(arena, configured):
    """4. Everything that separates a recording's parameters from the project's.

    ``detect_and_set_camera_parameters`` reads each recording and may override
    the depth and filter parameters for the camera it finds. For these kinect2
    recordings the generated config already matches, including the unsigned
    16-bit samples of the raw depth video. The one difference comes from
    saving, which marks the depth range as manually set.

    Nothing else moves, which is why the walkthrough's own session config
    holds the generated defaults: there was nothing in it that needed tuning.
    """
    whole = plain(read_yaml(str(configured)))
    config = {**whole["extract"], "camera_type": whole["camera_type"]}
    saved = parameters(arena)

    differing = {
        key for key in saved if key not in PATH_KEYS and config.get(key) != saved[key]
    }
    assert differing == {"manual_set_depth_range"}


def test_extraction_finds_the_parameter_file_by_its_name(arena, configured, project):
    """5. How extraction finds the per-recording parameters in cell 6: at a fixed
    name in the project folder, with no pointer to it in config.yaml."""
    assert arena == project / "session_config.yaml"
    assert "session_config_path" not in plain(read_yaml(str(configured)))


def test_the_widget_offers_four_actions(scratch):
    """6. The buttons a user has, by the label each shows."""
    _, _, buttons = open_widget(scratch)
    assert sorted(buttons) == sorted(ACTIONS)


def test_the_selector_lists_every_recording(scratch):
    """7. The widget opens on the first recording and can reach them all."""
    import panel as pn

    widget, data, _ = open_widget(scratch)
    selector = widget.view.__panel__().select(pn.widgets.Select)[0]

    assert list(selector.options) == SESSIONS
    assert data.path == SESSIONS[0]
    assert data.param.path.objects == SESSIONS


def test_nothing_is_computed_until_a_button_is_pressed(scratch):
    """8. Opening the widget reads no frames; the four panels start empty."""
    _, data, _ = open_widget(scratch)
    assert set(data.images) == {
        "Background",
        "Arena mask",
        "Extracted mouse",
        "Frame (background subtracted)",
    }
    assert all(image is None for image in data.images.values())


def test_computing_the_arena_mask_finds_the_floor(scratch):
    """9. The cell's purpose: a mask covering the arena and nothing else.

    ``get_roi`` returns candidate regions ranked by how arena-like they are.
    The first is the floor; the rest are fragments a user scrolls past with
    the mask index. Pinning the floor's share of the frame rather than the
    mask itself keeps the check meaningful across OpenCV releases, which
    change region boundaries by a pixel or two.
    """
    _, data, buttons = open_widget(scratch)
    click(buttons["Compute arena mask"])

    background = data.images["Background"]
    assert background.shape == (424, 512)
    assert 0 <= background.min() and background.max() <= 1000

    masks = data.images["Arena mask"]
    assert len(masks) > 0
    assert all(mask.shape == background.shape and mask.dtype == bool for mask in masks)
    assert masks[0].mean() == max(mask.mean() for mask in masks)
    assert 0.20 < masks[0].mean() < 0.28, "the arena is not most of a quarter of the frame"
    assert data.param.mask_index.bounds == (0, len(masks) - 1)


def test_saving_records_the_parameters_as_set(scratch):
    """10. What the save button writes, parameter by parameter.

    This is the cell's contract with cell 6: whatever a user settles on in the
    widget is what extraction runs with.
    """
    _, data, buttons = open_widget(scratch)
    data.depth_range = (600, 700)
    data.mask_dilations = 5
    data.mouse_height = (15, 100)
    data.dilation_kernel = 12
    data.mask_shape = "rectangle"
    data.crop_size = 100
    data.mask_index = 1
    data.mask_weight1, data.mask_weight2, data.mask_weight3 = 2, 0.5, 3
    click(buttons["Save session parameters"])

    saved = parameters(scratch / "session_config.yaml", data.path)
    for key, parameter in SAVED_BY.items():
        assert saved[key] == plain(getattr(data, parameter)), f"{key} came from {parameter}"
    assert (saved["min_height"], saved["max_height"]) == data.mouse_height
    assert saved["crop_size"] == [100, 100]
    assert saved["bg_roi_dilate"] == [12, 12]
    assert saved["bg_roi_weights"] == [2, 0.5, 3]


def assert_extraction_cuts_the_arena_shown(widget, data, buttons, root):
    """Compute and save in the tool, then find the arena the way extraction does,
    with the settings extraction resolves (the saved entry's arena tool settings,
    config.yaml's others), and compare it with the one the tool showed."""
    from moseq2_extract.helpers.wrappers import get_roi_wrapper
    from moseq2_extract.session_config import recordings_options

    click(buttons["Compute arena mask"])
    shown = data.images["Arena mask"][data.mask_index]
    click(buttons["Save session parameters"])

    recording = widget.sessions[data.path]
    resolved = recordings_options(str(root / "config.yaml"), [recording])[recording].as_dict()
    # the recording's session folder, where extraction writes and the tool keeps the background
    roi, _, _ = get_roi_wrapper(recording, resolved, output_dir=widget._output_dir(data.path))
    np.testing.assert_array_equal(roi.astype(bool), shown)


def test_tuned_weights_give_extraction_the_arena_the_tool_showed(scratch):
    """moseq2-extract#174: a user tunes a recording until the tool shows a good
    arena, and extraction has to cut that arena. Weights of (0, 1, 0) rank a
    different region of this recording first than the defaults do."""
    widget, data, buttons = open_widget(scratch)
    data.mask_weight1, data.mask_weight2, data.mask_weight3 = 0, 1, 0
    assert_extraction_cuts_the_arena_shown(widget, data, buttons, scratch)


def test_sorting_by_position_gives_extraction_the_arena_the_tool_showed(scratch):
    """``bg_sort_roi_by_position`` has no box in the tool; it comes from the config.
    Sorting this recording's five best regions top to bottom puts a fragment at
    the top edge ahead of the floor, and the tool has to show the same order."""
    with update_config(str(scratch / "config.yaml"), section="extract") as config_data:
        config_data["bg_sort_roi_by_position"] = True
        config_data["bg_sort_roi_by_position_max_rois"] = 5
    widget, data, buttons = open_widget(scratch)
    assert_extraction_cuts_the_arena_shown(widget, data, buttons, scratch)


def test_saving_leaves_the_other_recordings_alone(scratch):
    """11. A user tunes one recording at a time."""
    session_config = scratch / "session_config.yaml"
    _, data, buttons = open_widget(scratch)
    before = plain(read_yaml(str(session_config)))

    data.depth_range = (600, 700)
    click(buttons["Save session parameters"])

    after = plain(read_yaml(str(session_config)))
    assert after[data.path]["bg_roi_depth_range"] == [600, 700]
    for name in SESSIONS:
        if name != data.path:
            assert after[name] == before[name]


def test_saving_always_marks_the_depth_range_as_manual(scratch):
    """12. The one parameter a user does not choose.

    Saving sets ``manual_set_depth_range``, which tells extraction to take
    ``bg_roi_depth_range`` as given instead of detecting it. It is set whether
    or not the depth range was touched, so it is the only thing separating a
    saved recording from an untouched one. Captured because it is the reason a
    walkthrough's session config differs from a freshly opened widget's.
    """
    session_config = scratch / "session_config.yaml"
    fresh = plain(read_yaml(str(session_config)))
    for name in SESSIONS:
        fresh[name]["manual_set_depth_range"] = False
    import yaml

    session_config.write_text(yaml.safe_dump(fresh))

    _, data, buttons = open_widget(scratch)
    click(buttons["Save session parameters"])

    assert parameters(session_config, data.path)["manual_set_depth_range"] is True


def test_advancing_moves_to_the_next_recording_and_recomputes(scratch):
    """13. What the second save button does beyond saving.

    It advances the selector, reloads the parameters belonging to the new
    recording, and computes that recording's arena mask and a short test
    extraction, so the next recording is on screen ready to check.
    """
    _, data, buttons = open_widget(scratch)
    data.depth_range = (600, 700)
    click(buttons["Save session parameters and move to next"])

    assert data.path == SESSIONS[1]
    assert data.depth_range == (650, 750), "the new recording's own parameters"
    assert data.images["Arena mask"] is not None
    assert data.images["Extracted mouse"].shape == (100, 80, 80)
    assert data.images["Frame (background subtracted)"].shape[0] == 100


def test_advancing_past_the_last_recording_wraps(scratch):
    """14. There is no end of the list; it returns to the first recording."""
    _, data, buttons = open_widget(scratch)
    visited = []
    for _ in range(len(SESSIONS) + 1):
        visited.append(data.path)
        click(buttons["Save session parameters and move to next"])

    assert visited == SESSIONS + [SESSIONS[0]]


@pytest.mark.parametrize(
    ("typed", "expected"),
    [("3", [3]), ("[3, 3]", [3, 3]), ("(5, 5)", [5, 5]), ("3.7", [3]), ("None", [3])],
)
def test_filter_sizes_are_read_as_python_literals(scratch, typed, expected):
    """15. The filter sizes are text boxes, and the text is evaluated.

    A user types into a box and the widget runs the text through ``eval``, so
    a tuple is accepted as readily as a list, a float is truncated, and
    anything that is neither a number nor a sequence falls back to ``[3]``
    with a printed warning.
    """
    _, data, buttons = open_widget(scratch)
    data.spatial_filter = typed
    click(buttons["Save session parameters"])

    assert parameters(scratch / "session_config.yaml", data.path)["spatial_filter_size"] == expected


@pytest.mark.parametrize("typed", ["3 by 3", "", "three"])
@pytest.mark.parametrize(
    ("box", "key", "fallback"),
    [("spatial_filter", "spatial_filter_size", [3]), ("temporal_filter", "temporal_filter_size", [0])],
)
def test_a_filter_size_that_is_not_python_falls_back(scratch, typed, box, key, fallback):
    """Text that makes no sense as a filter size falls back, as a value of the
    wrong type does, and the save goes ahead.

    Clearing the box is the likeliest way to reach it.
    """
    _, data, buttons = open_widget(scratch)
    setattr(data, box, typed)
    click(buttons["Save session parameters"])

    assert parameters(scratch / "session_config.yaml", data.path)[key] == fallback


def test_the_weight_boxes_accept_a_decimal(scratch):
    """Every one of these parameters is declared ``param.Number``, so a user
    can type ``0.5`` into any of them, however the shipped config happens to
    spell its value (``1`` and ``30`` as well as ``0.1``).
    """
    import panel as pn

    widget, _, _ = open_widget(scratch)
    layout = widget.view.__panel__()
    boxes = {
        b.name: b
        for b in layout.select(pn.widgets.IntInput) + layout.select(pn.widgets.FloatInput)
    }

    for label in (
        "Mask area",
        "Mask extent",
        "Mask distance from center",
        "RANSAC noise tolerance",
    ):
        boxes[label].value = 0.5
        assert boxes[label].value == 0.5, label


# ------------------------------------------------- running the cell again


@pytest.fixture
def unextracted(tmp_path, project, arena):
    """A copy of the project as the arena step leaves it: recordings, config and
    session config, with nothing extracted.

    The cell skips extracted recordings (``skip_extracted = True``), so a copy
    made after the chain's extraction would show it none.
    """
    root = tmp_path / "project"
    shutil.copytree(
        project,
        root,
        ignore=shutil.ignore_patterns(
            "proc", "aggregate_results", "_pca", "models", "moseq2-project.yaml", "*.tar.gz"
        ),
    )
    return root


def run_arena_cell(root):
    """Run the notebook's own arena cell on a project; return the widget it shows.

    The cell ends on the widget, which a notebook displays. Evaluated here the
    way a kernel does, so that expression is what comes back.
    """
    import ast

    from cells import code_cell

    from moseq2_common.project import Project

    namespace = {"project": Project.at(root)}
    tree = ast.parse(code_cell("MoSeq2-Extract-Modeling-Notebook.ipynb", 5))
    last = tree.body.pop()
    exec(compile(tree, "<arena cell>", "exec"), namespace)
    return eval(compile(ast.Expression(last.value), "<arena cell>", "eval"), namespace)


def tune_first_recording(widget):
    """Tune and save the first recording, as a user would; return what was set."""
    data = widget.session_data
    data.depth_range = (600, 700)
    data.mask_weight1, data.mask_weight2, data.mask_weight3 = 0, 1, 0
    click(buttons_of(widget.view)["Save session parameters"])
    return data.path


def test_rerunning_the_cell_keeps_saved_parameters(unextracted):
    """A user tunes a recording, saves, and runs the cell again (after a kernel
    restart, say). The saved parameters stay in the file and are what the
    widget shows."""
    session_config = unextracted / "session_config.yaml"
    tuned = tune_first_recording(run_arena_cell(unextracted))
    saved = plain(read_yaml(str(session_config)))

    widget = run_arena_cell(unextracted)

    assert plain(read_yaml(str(session_config))) == saved
    assert saved[tuned]["bg_roi_depth_range"] == [600, 700]
    assert saved[tuned]["bg_roi_weights"] == [0, 1, 0]
    assert widget.session_data.path == tuned
    assert widget.session_data.depth_range == (600, 700)
    assert (
        widget.session_data.mask_weight1,
        widget.session_data.mask_weight2,
        widget.session_data.mask_weight3,
    ) == (0, 1, 0)


def test_a_new_recording_is_detected_and_nothing_is_written(unextracted):
    """A recording added to the project after the first run is shown with
    parameters detected for it; opening the tool writes no entry for it, and
    every recording already in the file keeps what it has."""
    from moseq2_extract.util import detect_and_set_camera_parameters
    from moseq2_app.util import read_and_clean_config

    session_config = unextracted / "session_config.yaml"
    tune_first_recording(run_arena_cell(unextracted))
    before = session_config.read_bytes()

    shutil.copytree(unextracted / "saline_example_1", unextracted / "saline_example_2")
    widget = run_arena_cell(unextracted)

    assert session_config.read_bytes() == before
    detected = detect_and_set_camera_parameters(
        read_and_clean_config(str(unextracted / "config.yaml")),
        str(unextracted / "saline_example_2" / "depth.avi"),
    )
    assert plain(widget.session_config["saline_example_2/depth.avi"]) == plain(detected)


# ------------------------------------------------- a plane-fit background

#: The recording the plane-fit tests run on.
STILL_RECORDING = "amphetamine_example_0"


def h5_arena_and_frames(results):
    """What an extraction cut and subtracted: its arena, its frames and its background."""
    import h5py

    with h5py.File(results, "r") as f:
        return (
            f["metadata/extraction/roi"][()].astype(bool),
            f["frames"][()],
            f["metadata/extraction/background"][()],
        )


@pytest.fixture(scope="module")
def plane_run(tmp_path_factory, raw_sessions):
    """A user with a mouse that sits still sets ``use_plane_bground``, finds the
    arena in the tool (clicking "Compute arena mask" twice, then "Compute
    extraction"), saves, runs the extraction cell, and runs it again.

    ``use_plane_bground`` replaces the background extraction subtracts with a
    plane fit to the floor. The arena is found on the median background, where
    the floor and the walls differ in depth; on the plane the floor is
    everywhere, and the arena found there is the whole frame.

    One recording, so the two extractions take seconds. Returns what each step
    showed or wrote.
    """
    from moseq2_extract.gui import extract_found_sessions, generate_config_command
    from moseq2_extract.io.image import read_image
    from moseq2_extract.session_config import recordings_options

    root = tmp_path_factory.mktemp("plane") / "project"
    shutil.copytree(raw_sessions / STILL_RECORDING, root / STILL_RECORDING)
    config = root / "config.yaml"
    generate_config_command(str(config), camera_type="k2")
    with update_config(str(config), section="extract") as config_data:
        config_data["use_plane_bground"] = True
        config_data["cluster_type"] = "local"

    widget, data, buttons = open_widget(root)
    proc = root / "aggregate_results" / STILL_RECORDING
    run = {"root": root, "proc": proc, "widget": widget}

    click(buttons["Compute arena mask"])
    run["first click"] = data.images["Arena mask"][data.mask_index]
    run["background shown"] = data.images["Background"]
    click(buttons["Compute arena mask"])
    run["second click"] = data.images["Arena mask"][data.mask_index]

    # "Compute extraction" finds the arena again before extracting; keep what it found
    found = []
    compute_arena_mask = widget.compute_arena_mask

    def recording_compute_arena_mask():
        background, rois = compute_arena_mask()
        found.append((background, rois))
        return background, rois

    widget.compute_arena_mask = recording_compute_arena_mask
    click(buttons["Compute extraction"])
    widget.compute_arena_mask = compute_arena_mask
    run["compute extraction"] = found[-1][1][data.mask_index]

    click(buttons["Save session parameters"])
    # the settings extraction resolves from the saved entry and config.yaml
    recording = widget.sessions[data.path]
    run["saved"] = recordings_options(str(config), [recording])[recording].as_dict()

    results = proc / "results.h5"
    extract_found_sessions(str(root), str(config), extract_all=True, skip_extracted=False)
    run["first extraction"] = h5_arena_and_frames(results)
    run["bground.tiff after the first extraction"] = read_image(str(proc / "bground.tiff"), scale=True)
    extract_found_sessions(str(root), str(config), extract_all=True, skip_extracted=False)
    run["second extraction"] = h5_arena_and_frames(results)
    run["bground.tiff after the second extraction"] = read_image(str(proc / "bground.tiff"), scale=True)
    return run


@pytest.fixture(scope="module")
def median_background(tmp_path_factory, plane_run):
    """The recording's median background, computed in an empty folder the way
    extraction and the tool compute it, read back as ``bground.tiff`` holds it."""
    from moseq2_extract.extract.proc import get_bground_im_file
    from moseq2_extract.io.image import read_image

    clean = tmp_path_factory.mktemp("median")
    recording = plane_run["widget"].sessions[f"{STILL_RECORDING}/depth.avi"]
    get_bground_im_file(recording, output_dir=str(clean))
    return read_image(str(clean / "bground.tiff"), scale=True)


def test_a_plane_background_finds_the_floor_on_the_first_click(plane_run):
    """The arena is most of a quarter of the frame, not all of it."""
    shown = plane_run["first click"]
    assert 0.20 < shown.mean() < 0.28, f"the arena is {shown.mean():.0%} of the frame"


def test_a_plane_background_shows_the_same_arena_on_the_second_click(plane_run):
    np.testing.assert_array_equal(plane_run["second click"], plane_run["first click"])


def test_a_plane_background_gives_compute_extraction_the_arena_shown(plane_run):
    np.testing.assert_array_equal(plane_run["compute extraction"], plane_run["first click"])


def test_a_plane_background_gives_extraction_the_arena_the_tool_showed(plane_run):
    arena, _, _ = plane_run["first extraction"]
    np.testing.assert_array_equal(arena, plane_run["first click"])


def test_a_plane_background_gives_a_second_extraction_the_same_arena_and_frames(plane_run):
    first_arena, first_frames, first_background = plane_run["first extraction"]
    second_arena, second_frames, second_background = plane_run["second extraction"]
    np.testing.assert_array_equal(second_arena, plane_run["first click"])
    np.testing.assert_array_equal(second_arena, first_arena)
    np.testing.assert_array_equal(second_background, first_background)
    np.testing.assert_array_equal(second_frames, first_frames)


def test_a_plane_background_leaves_the_median_in_bground_tiff(plane_run, median_background):
    """The session folder's ``bground.tiff`` is the median background extraction computed,
    so the plane fit never replaces it."""
    np.testing.assert_array_equal(plane_run["bground.tiff after the first extraction"], median_background)
    np.testing.assert_array_equal(plane_run["bground.tiff after the second extraction"], median_background)


def test_a_plane_background_is_what_extraction_subtracts(plane_run, median_background):
    """Extraction subtracts, and records, the plane fit to the floor of the
    median background with the saved parameters; the tool shows and subtracts
    the same plane."""
    from moseq2_extract.extract.proc import get_roi
    from moseq2_common.frames import select_strel

    params = dict(plane_run["saved"])
    _, plane = get_roi(
        median_background,
        **params,
        strel_dilate=select_strel(params["bg_roi_shape"], tuple(params["bg_roi_dilate"])),
        strel_erode=select_strel(params["bg_roi_shape"], tuple(params["bg_roi_erode"])),
        get_all_data=False,
    )
    a, b, c, d = plane
    yy, xx = np.mgrid[: median_background.shape[0], : median_background.shape[1]]
    expected = (a * xx + b * yy + d) / -c

    _, _, recorded = plane_run["first extraction"]
    np.testing.assert_allclose(recorded, expected)
    np.testing.assert_allclose(plane_run["background shown"], expected)


# ------------------------------------------ the preview against extraction

#: How many frames the preview and the extraction it is compared with take.
PREVIEW_FRAMES = 30


def test_arena_preview_matches_extraction_for_graduated_walls(tmp_path, raw_sessions):
    """L13: "Compute extraction" shows what extraction will write.

    ``graduate_walls`` subtracts the floor's depth, measured from the arena,
    where the mouse sits on the bucket's edge, and with more than one dilation
    it replaces the background with a gradient built from that depth. The
    preview has to measure the floor and build the background as extraction
    does, or the mouse it shows is not the mouse extraction writes.

    The preview extracts frames 1 to ``PREVIEW_FRAMES``; extraction is given
    the same frames (one frame trimmed from the start), so both run one chunk
    over the same frames with the same saved settings.
    """
    import h5py
    from moseq2_extract.gui import generate_config_command
    from moseq2_extract.helpers.wrappers import extract_wrapper
    from moseq2_extract.session_config import recordings_options

    root = tmp_path / "project"
    shutil.copytree(raw_sessions / "saline_example_0", root / "saline_example_0")
    config = root / "config.yaml"
    generate_config_command(str(config), camera_type="k2")
    with update_config(str(config), section="extract") as config_data:
        config_data["graduate_walls"] = True

    widget, data, buttons = open_widget(root)
    data.frames_to_extract = PREVIEW_FRAMES
    data.mask_dilations = 3
    click(buttons["Compute extraction"])
    shown = np.asarray(data.images["Extracted mouse"])
    click(buttons["Save session parameters"])

    recording = widget.sessions[data.path]
    # the dict extraction hands its work function: the recording's resolved options
    settings = recordings_options(str(config), [recording])[recording].as_dict()
    assert settings["graduate_walls"] and settings["dilate_iterations"] > 1, (
        "the saved settings do not graduate the walls, so this compares nothing"
    )
    settings["frame_trim"] = (1, 0)
    output_dir = extract_wrapper(recording, widget._output_dir(data.path), settings, num_frames=PREVIEW_FRAMES + 1)
    with h5py.File(f"{output_dir}/results.h5", "r") as f:
        extracted = f["frames"][()]

    assert shown.shape == extracted.shape
    np.testing.assert_array_equal(shown, extracted)


# ------------------------------------------ filters turned off in config.yaml

@pytest.mark.parametrize(
    "key, off",
    [
        ("tail_filter_shape", "none"),
        ("cable_filter_shape", "none"),
        ("tail_filter_size", [0, 0]),
        ("cable_filter_size", [0, 0]),
        ("bg_roi_shape", "none"),
    ],
)
def test_the_tool_opens_and_previews_a_config_that_turns_a_filter_off(tmp_path, raw_sessions, key, off):
    """``none`` and ``0 0`` turn the tail, cable and arena-mask filters off.

    The command line takes these settings, so the arena tool, which shows
    ``config.yaml``'s values for a recording with no entry, has to take them too: open on a
    ``config.yaml`` holding one, preview an extraction, and save it unchanged.
    """
    from moseq2_extract.gui import generate_config_command

    root = tmp_path / "project"
    shutil.copytree(raw_sessions / "saline_example_0", root / "saline_example_0")
    config = root / "config.yaml"
    generate_config_command(str(config), camera_type="k2")
    with update_config(str(config), section="extract") as config_data:
        config_data[key] = off

    _, data, buttons = open_widget(root)
    data.frames_to_extract = 10
    click(buttons["Compute extraction"])
    assert data.images["Extracted mouse"] is not None
    click(buttons["Save session parameters"])
    assert parameters(root / "session_config.yaml", data.path)[key] == off
