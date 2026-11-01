"""Cell 5: find each recording's arena floor and save its extraction parameters.

    session_config_path = join(progress_paths['base_dir'], 'session_config.yaml')
    progress_paths = update_progress(progress_filepath, 'session_config', session_config_path)

    with update_config(progress_paths['config_file']) as config_data:
        config_data['output_dir'] = 'proc'

    ArenaMaskWidget(progress_paths['base_dir'], progress_paths['config_file'],
                    progress_paths['session_config'],
                    skip_extracted=skip_extracted)

The first interactive cell. Everything up to here produced one configuration
for the whole project; this cell produces one per recording, because the arena
sits at a different depth in every rig and often moves between recordings.

Opening the widget already writes the file: it runs
``detect_and_set_camera_parameters`` over each recording that has no saved
parameters and saves a copy of the configuration under the recording's folder
name, or under the folder's path below the project when session folders share
a name. A recording that has saved parameters keeps them, so running the cell
again shows what was saved. The widget then lets a user look at the floor it
found, adjust the parameters, and save.

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

import numpy as np
import pytest

from moseq2_app.gui.progress import restore_progress_vars
from moseq2_app.roi.widget import ArenaMaskWidget
from moseq2_app.util import update_config
from moseq2_extract.util import read_yaml

from test_03_config import plain
from widgets import buttons_of, click

#: The four recordings, in the order the selector offers them.
RECORDINGS = [
    "amphetamine_example_0",
    "amphetamine_example_1",
    "saline_example_0",
    "saline_example_1",
]

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
    "tracking_model_mask_threshold": "tracking_model_mask_thresh",
    "use_tracking_model": "tracking_model_flag",
}


def parameters(session_config, recording="saline_example_0"):
    """One recording's saved parameters, as plain yaml types."""
    return plain(read_yaml(str(session_config)))[recording]


def open_widget(root):
    """Open the widget on a project, as the cell does, and index its buttons."""
    widget = ArenaMaskWidget(
        str(root), str(root / "config.yaml"), str(root / "session_config.yaml")
    )
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
    """1. The cell's output: one entry per recording, keyed by folder name.

    A recording is a folder holding a depth video. Later cells add folders of
    their own to the project (``aggregate_results/``, ``_pca/``, ``models/``),
    and the shared chain may have reached them before this test runs.
    """
    saved = read_yaml(str(arena))
    assert sorted(saved) == RECORDINGS
    assert sorted(p.parent.name for p in project.glob("*/depth.avi")) == RECORDINGS


def test_detected_parameters_are_unchanged(arena, data_regression):
    """2. Every parameter a recording is extracted with, pinned.

    These are the values the rest of the pipeline runs on, and they come from
    a dependency's detection routine rather than from anything written down,
    so a change in moseq2-extract shows up here as a change to every user's
    extraction.
    """
    saved = parameters(arena)
    data_regression.check({k: v for k, v in sorted(saved.items()) if k not in PATH_KEYS})


def test_every_recording_gets_the_same_parameters(arena):
    """3. Detection is per recording, and these four recordings agree."""
    saved = plain(read_yaml(str(arena)))
    reference = {k: v for k, v in saved[RECORDINGS[0]].items() if k not in PATH_KEYS}
    for name in RECORDINGS[1:]:
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
    config = plain(read_yaml(str(configured)))
    saved = parameters(arena)

    differing = {
        key for key in saved if key not in PATH_KEYS and config.get(key) != saved[key]
    }
    assert differing == {"manual_set_depth_range"}


def test_the_configuration_points_at_the_parameter_file(arena, configured, progress_file):
    """5. How extraction finds the per-recording parameters in cell 6."""
    assert plain(read_yaml(str(configured)))["session_config_path"] == str(arena)
    recorded = restore_progress_vars(str(progress_file), init=False, overwrite=False)
    assert recorded["session_config"] == str(arena)


def test_the_widget_offers_four_actions(scratch):
    """6. The buttons a user has, by the label each shows."""
    _, _, buttons = open_widget(scratch)
    assert sorted(buttons) == sorted(ACTIONS)


def test_the_selector_lists_every_recording(scratch):
    """7. The widget opens on the first recording and can reach them all."""
    import panel as pn

    widget, data, _ = open_widget(scratch)
    selector = widget.view.__panel__().select(pn.widgets.Select)[0]

    assert list(selector.options) == RECORDINGS
    assert data.path == RECORDINGS[0]
    assert data.param.path.objects == RECORDINGS


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
    from the saved parameters, and compare it with the one the tool showed."""
    from moseq2_extract.helpers.wrappers import get_roi_wrapper

    click(buttons["Compute arena mask"])
    shown = data.images["Arena mask"][data.mask_index]
    click(buttons["Save session parameters"])

    saved = read_yaml(str(root / "session_config.yaml"))[data.path]
    recording = widget.sessions[data.path]
    roi, _, _ = get_roi_wrapper(recording, saved, output_dir=str(root / data.path / "proc"))
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
    with update_config(str(scratch / "config.yaml")) as config_data:
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
    for name in RECORDINGS:
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
    for name in RECORDINGS:
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

    assert data.path == RECORDINGS[1]
    assert data.depth_range == (650, 750), "the new recording's own parameters"
    assert data.images["Arena mask"] is not None
    assert data.images["Extracted mouse"].shape == (100, 80, 80)
    assert data.images["Frame (background subtracted)"].shape[0] == 100


def test_advancing_past_the_last_recording_wraps(scratch):
    """14. There is no end of the list; it returns to the first recording."""
    _, data, buttons = open_widget(scratch)
    visited = []
    for _ in range(len(RECORDINGS) + 1):
        visited.append(data.path)
        click(buttons["Save session parameters and move to next"])

    assert visited == RECORDINGS + [RECORDINGS[0]]


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


def test_the_tracking_likelihood_threshold_accepts_a_decimal(scratch):
    """The tracking model's mouse likelihood threshold is a ``param.Number``
    too. The shipped config spells it ``-16``, and a user who enables the
    tracking model can type ``-16.5`` into it.
    """
    import panel as pn

    widget, _, _ = open_widget(scratch)
    layout = widget.view.__panel__()
    boxes = {
        b.name: b
        for b in layout.select(pn.widgets.IntInput) + layout.select(pn.widgets.FloatInput)
    }

    box = boxes["Tracking model mouse likelihood threshold"]
    box.value = -16.5
    assert box.value == -16.5


# ------------------------------------------------- running the cell again


@pytest.fixture
def unextracted(tmp_path, project, arena):
    """A copy of the project as the arena step leaves it: recordings, config and
    session config, with nothing extracted.

    The cell skips extracted recordings (``skip_extracted = True``), so a copy
    made after the chain's extraction would show it none.
    """
    from moseq2_app.gui.progress import update_progress

    root = tmp_path / "project"
    shutil.copytree(
        project,
        root,
        ignore=shutil.ignore_patterns(
            "proc", "aggregate_results", "_pca", "models", "progress.yaml", "*.tar.gz"
        ),
    )
    progress = root / "progress.yaml"
    progress.write_text("{}\n")
    update_progress(str(progress), "base_dir", str(root))
    update_progress(str(progress), "config_file", str(root / "config.yaml"))
    return root


def run_arena_cell(root):
    """Run the notebook's own arena cell on a project; return the widget it shows.

    The cell ends on the widget, which a notebook displays. Evaluated here the
    way a kernel does, so that expression is what comes back.
    """
    import ast

    from cells import code_cell

    progress = root / "progress.yaml"
    namespace = {
        "progress_filepath": str(progress),
        "progress_paths": read_yaml(str(progress)),
    }
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


def test_a_new_recording_is_detected_and_the_others_keep_theirs(unextracted):
    """A recording added to the project after the first run gets parameters
    detected for it; every recording already in the file keeps what it has."""
    from moseq2_extract.util import detect_and_set_camera_parameters
    from moseq2_app.util import read_and_clean_config

    session_config = unextracted / "session_config.yaml"
    tune_first_recording(run_arena_cell(unextracted))
    before = plain(read_yaml(str(session_config)))

    shutil.copytree(unextracted / "saline_example_1", unextracted / "saline_example_2")
    widget = run_arena_cell(unextracted)

    after = plain(read_yaml(str(session_config)))
    assert sorted(after) == sorted(RECORDINGS + ["saline_example_2"])
    for name in RECORDINGS:
        assert after[name] == before[name], f"{name} was rewritten"
    detected = detect_and_set_camera_parameters(
        read_and_clean_config(str(unextracted / "config.yaml")),
        str(unextracted / "saline_example_2" / "depth.avi"),
    )
    assert after["saline_example_2"] == plain(detected)
    assert "saline_example_2" in widget.sessions


@pytest.mark.parametrize("value", [True, False])
def test_the_old_argument_is_accepted_ignored_and_explained(unextracted, capsys, value):
    """A notebook from an earlier release still passes overwrite_session_configs.
    It opens, keeps the saved parameters whatever the value, and says once that
    the argument can go."""
    session_config = unextracted / "session_config.yaml"
    tune_first_recording(run_arena_cell(unextracted))
    saved = plain(read_yaml(str(session_config)))
    capsys.readouterr()

    widget = ArenaMaskWidget(
        str(unextracted),
        str(unextracted / "config.yaml"),
        str(session_config),
        skip_extracted=True,
        overwrite_session_configs=value,
    )

    assert widget.view is not None
    assert plain(read_yaml(str(session_config))) == saved
    printed = [line for line in capsys.readouterr().out.splitlines() if "overwrite_session_configs" in line]
    assert printed == [
        "overwrite_session_configs is no longer used and is ignored: the arena tool "
        f"always keeps the parameters saved in {session_config} and detects parameters "
        "only for recordings that have none there yet. Remove overwrite_session_configs "
        "from the cell that opens ArenaMaskWidget."
    ]


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

    root = tmp_path_factory.mktemp("plane") / "project"
    shutil.copytree(raw_sessions / STILL_RECORDING, root / STILL_RECORDING)
    config = root / "config.yaml"
    generate_config_command(str(config), camera_type="k2")
    with update_config(str(config)) as config_data:
        config_data["output_dir"] = "proc"
        config_data["use_plane_bground"] = True
        config_data["cluster_type"] = "local"

    widget, data, buttons = open_widget(root)
    proc = root / data.path / "proc"
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
    run["saved"] = read_yaml(str(root / "session_config.yaml"))[data.path]

    results = proc / "results_00.h5"
    extract_found_sessions(str(root), str(config), extract_all=True, skip_extracted=False)
    run["first extraction"] = h5_arena_and_frames(results)
    run["bground.tiff after the first extraction"] = read_image(str(proc / "bground.tiff"), scale=True)
    extract_found_sessions(str(root), str(config), extract_all=True, skip_extracted=False)
    run["second extraction"] = h5_arena_and_frames(results)
    run["bground.tiff after the second extraction"] = read_image(str(proc / "bground.tiff"), scale=True)
    return run


@pytest.fixture(scope="module")
def median_background(tmp_path_factory, plane_run):
    """The recording's median background, computed from scratch in an empty
    folder the way the tool computes it, read back as the cache file holds it."""
    from moseq2_extract.extract.proc import get_bground_im_file
    from moseq2_extract.io.image import read_image

    clean = tmp_path_factory.mktemp("median")
    recording = plane_run["widget"].sessions[STILL_RECORDING]
    get_bground_im_file(recording, frame_stride=1000, output_dir=str(clean))
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
    """``proc/bground.tiff`` is the cache every later reader takes as the
    median background, so the plane fit never replaces it."""
    np.testing.assert_array_equal(plane_run["bground.tiff after the first extraction"], median_background)
    np.testing.assert_array_equal(plane_run["bground.tiff after the second extraction"], median_background)


def test_a_plane_background_is_what_extraction_subtracts(plane_run, median_background):
    """Extraction subtracts, and records, the plane fit to the floor of the
    median background with the saved parameters; the tool shows and subtracts
    the same plane."""
    from moseq2_extract.extract.proc import get_roi
    from moseq2_extract.util import select_strel

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
