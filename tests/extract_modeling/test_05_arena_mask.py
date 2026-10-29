"""Cell 5: find each recording's arena floor and save its extraction parameters.

    session_config_path = join(progress_paths['base_dir'], 'session_config.yaml')
    progress_paths = update_progress(progress_filepath, 'session_config', session_config_path)

    with update_config(progress_paths['config_file']) as config_data:
        config_data['camera_type'] = 'auto'
        config_data['output_dir'] = 'proc'

    ArenaMaskWidget(progress_paths['base_dir'], progress_paths['config_file'],
                    progress_paths['session_config'],
                    skip_extracted=skip_extracted,
                    overwrite_session_configs=overwrite_session_configs)

The first interactive cell. Everything up to here produced one configuration
for the whole project; this cell produces one per recording, because the arena
sits at a different depth in every rig and often moves between recordings.

Opening the widget already writes the file: it runs
``detect_and_set_camera_parameters`` over each recording and saves a copy of
the configuration under the recording's folder name. The widget then lets a
user look at the floor it found, adjust the parameters, and save.

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

import pytest

from moseq2_app.gui.progress import restore_progress_vars
from moseq2_app.roi.widget import ArenaMaskWidget
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
    """A copy of the project, for tests that change parameters and save.

    The chain's own copy holds what the walkthrough saved, and cell 6 extracts
    from it, so a test that writes different parameters works on its own copy.
    Copying costs a fraction of a second and the backgrounds come with it
    already computed.
    """
    root = tmp_path / "project"
    shutil.copytree(project, root)
    return root


def test_every_recording_gets_a_parameter_file(arena, project):
    """1. The cell's output: one entry per recording, keyed by folder name."""
    saved = read_yaml(str(arena))
    assert sorted(saved) == RECORDINGS
    assert sorted(p.name for p in project.iterdir() if p.is_dir()) == RECORDINGS


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


def test_the_widget_offers_four_actions(arena, project):
    """6. The buttons a user has, by the label each shows."""
    _, _, buttons = open_widget(project)
    assert sorted(buttons) == sorted(ACTIONS)


def test_the_selector_lists_every_recording(arena, project):
    """7. The widget opens on the first recording and can reach them all."""
    import panel as pn

    widget, data, _ = open_widget(project)
    selector = widget.view.__panel__().select(pn.widgets.Select)[0]

    assert list(selector.options) == RECORDINGS
    assert data.path == RECORDINGS[0]
    assert data.param.path.objects == RECORDINGS


def test_nothing_is_computed_until_a_button_is_pressed(arena, project):
    """8. Opening the widget reads no frames; the four panels start empty."""
    _, data, _ = open_widget(project)
    assert set(data.images) == {
        "Background",
        "Arena mask",
        "Extracted mouse",
        "Frame (background subtracted)",
    }
    assert all(image is None for image in data.images.values())


def test_computing_the_arena_mask_finds_the_floor(arena, project):
    """9. The cell's purpose: a mask covering the arena and nothing else.

    ``get_roi`` returns candidate regions ranked by how arena-like they are.
    The first is the floor; the rest are fragments a user scrolls past with
    the mask index. Pinning the floor's share of the frame rather than the
    mask itself keeps the check meaningful across OpenCV releases, which
    change region boundaries by a pixel or two.
    """
    _, data, buttons = open_widget(project)
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


def test_saving_leaves_the_other_recordings_alone(scratch):
    """11. A user tunes one recording at a time."""
    session_config = scratch / "session_config.yaml"
    before = plain(read_yaml(str(session_config)))

    _, data, buttons = open_widget(scratch)
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
