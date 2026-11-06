"""The settings the arena tool saves for a recording are the ones extraction takes from its entry.

``ARENA_TOOL_SETTINGS`` in moseq2-extract lists them, and extraction takes only
those from a recording's ``session_config.yaml`` entry and every other setting
from ``config.yaml``. The list and the tool's ``set_session_config_vars`` must
agree: a key the tool writes but the list lacks would be tuned in the tool and
then ignored, and a key the list holds but the tool never writes would never
be saved.

The second test follows the notebook: the
arena cell runs, a recording is tuned and saved, then ``config.yaml`` gains a
flip classifier and a ``frame_trim`` and the arena cell runs again before the
extraction cell.
"""

from __future__ import annotations

import shutil

import pytest

import moseq2_extract.steps as extract_steps
from moseq2_app.util import update_config
from moseq2_extract.gui import extract_command, generate_config_command
from moseq2_common.yaml import read_yaml

from widgets import buttons_of, click


class _Tuned:
    """Stands in for the tool's ``ArenaMaskData``: every value the tool shows."""

    path = "session"
    mouse_height = (10, 100)
    spatial_filter = "[3]"
    temporal_filter = "[0]"

    def __getattr__(self, name):
        return 1


def test_the_tool_saves_exactly_the_shared_list():
    from moseq2_app.roi.widget import ArenaMaskWidget
    from moseq2_extract.session_config import ARENA_TOOL_SETTINGS

    widget = ArenaMaskWidget.__new__(ArenaMaskWidget)
    widget.session_config = {"session": {}}
    widget.session_data = _Tuned()

    widget.set_session_config_vars()

    assert sorted(widget.session_config["session"]) == sorted(ARENA_TOOL_SETTINGS)


def open_arena_tool(root):
    """The notebook's arena cell."""
    import panel as pn

    from moseq2_app.roi.widget import ArenaMaskWidget

    pn.extension()
    return ArenaMaskWidget(str(root))


@pytest.fixture
def project(tmp_path, raw_sessions):
    root = tmp_path / "project"
    for name in ("saline_example_0", "saline_example_1"):
        shutil.copytree(f"{raw_sessions}/{name}", root / name)
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    return root


def test_config_yaml_changes_after_the_tool_reach_extraction(project, monkeypatch, raw_sessions):
    widget = open_arena_tool(project)
    data = widget.session_data
    assert data.path == "saline_example_0/depth.avi"
    data.crop_size = 100
    data.depth_range = (600, 700)
    click(buttons_of(widget.view)["Save session parameters"])

    classifier = f"{raw_sessions}/flip_classifier_k2_c57_10to13weeks.npz"
    with update_config(str(project / "config.yaml"), section="extract") as config:
        config["flip_classifier"] = classifier
        config["frame_trim"] = [100, 0]
        config["crop_size"] = [120, 120]
        config["bg_roi_depth_range"] = [650, 750]
    open_arena_tool(project)

    received = {}
    monkeypatch.setattr(
        extract_steps,
        "extract_wrapper",
        lambda input_file, output_dir, config, num_frames=None, skip=False, uuid=None, record=None: received.update(config),
    )
    extract_command(str(project / "saline_example_0" / "depth.avi"), None, str(project / "config.yaml"))

    assert received["flip_classifier"] == classifier
    assert tuple(received["frame_trim"]) == (100, 0)
    # tuned in the tool
    assert tuple(received["crop_size"]) == (100, 100)
    assert tuple(received["bg_roi_depth_range"]) == (600, 700)
    assert received["manual_set_depth_range"] is True

    # the entry holds the tool's settings only, and extraction does not rewrite it
    saved = read_yaml(str(project / "session_config.yaml"))["saline_example_0/depth.avi"]
    assert "flip_classifier" not in saved
    assert saved["crop_size"] == [100, 100]


def _recorded(monkeypatch, name, calls):
    """Replace ``name`` in the widget's module with a recorder of each call's arguments,
    which then runs the function."""
    import moseq2_app.roi.widget as widget_module

    real = getattr(widget_module, name)

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(widget_module, name, recorder)


def _save_then_change_config(project, raw_sessions, settings):
    """Open the tool, save the first recording, write ``settings`` into config.yaml's extract
    section, and reopen it."""
    widget = open_arena_tool(project)
    click(buttons_of(widget.view)["Save session parameters"])
    saved = read_yaml(str(project / "session_config.yaml"))["saline_example_0/depth.avi"]
    with update_config(str(project / "config.yaml"), section="extract") as config:
        config.update(settings)
    return saved, open_arena_tool(project)


def test_config_yaml_changes_after_the_tool_reach_the_preview_extraction(project, monkeypatch, raw_sessions):
    """ "Compute extraction" resolves its settings as extraction does: the tool's values,
    including ones tuned and not yet saved, over config.yaml as it is on disk now."""
    classifier = f"{raw_sessions}/flip_classifier_k2_c57_10to13weeks.npz"
    saved, widget = _save_then_change_config(
        project, raw_sessions, {"flip_classifier": classifier, "frame_trim": [100, 0]}
    )
    assert "flip_classifier" not in saved
    data = widget.session_data
    assert data.path == "saline_example_0/depth.avi"
    low, high = data.depth_range
    data.depth_range = (low - 5, high + 5)
    data.tail_filters = 2
    data.frames_to_extract = 5

    calls = []
    _recorded(monkeypatch, "extract_chunk", calls)
    click(buttons_of(widget.view)["Compute extraction"])

    assert len(calls) == 1
    received = calls[0][1]
    assert received["flip_classifier"] == classifier
    assert tuple(received["frame_trim"]) == (100, 0)
    # tuned in the tool and not saved
    assert received["tail_filter_iters"] == 2
    assert tuple(received["bg_roi_depth_range"]) == (low - 5, high + 5)
    assert received["manual_set_depth_range"] is True


def test_the_preview_stops_on_a_flip_classifier_that_does_not_fit_the_crop_size(project, raw_sessions):
    """L13: "Compute extraction" runs extraction's flip-classifier check, so a crop size the
    classifier was not trained on stops the preview with the message extraction gives,
    naming both sizes, before any frame is extracted."""
    classifier = f"{raw_sessions}/flip_classifier_k2_c57_10to13weeks.npz"
    _, widget = _save_then_change_config(project, raw_sessions, {"flip_classifier": classifier})
    widget.session_data.crop_size = 100
    widget.session_data.frames_to_extract = 5

    with pytest.raises(ValueError, match=r"Your crop_size is \(100, 100\)") as stopped:
        click(buttons_of(widget.view)["Compute extraction"])
    assert "expects 6400 pixels (i.e. crop_size (80, 80))" in str(stopped.value)


def _classifier_trained_at(root, config_file):
    """A flip classifier trained on 80 x 80 frames, cleaned with ``config_file``'s extract settings."""
    import pickle

    import h5py
    import numpy as np

    from moseq2_app.steps import train_flip_classifier

    root.mkdir()
    rng = np.random.default_rng(0)
    with h5py.File(root / "session.h5", "w") as f:
        frames = np.zeros((30, 80, 80), dtype=np.uint8)
        frames[:, 35:45, 10:40] = 50
        frames[:, 30:50, 40:65] = 80
        f["frames"] = frames + rng.integers(0, 5, size=frames.shape, dtype=np.uint8)
    with (root / "flip-training-frame-ranges.p").open("wb") as f:
        pickle.dump(({"session": str(root / "session.h5")}, {"session": [(False, range(0, 30))]}), f)
    return str(train_flip_classifier(str(root / "flip-training-frame-ranges.p"), str(root / "flip.p"),
                                     str(config_file)).classifier)


def test_the_preview_warns_on_a_filter_the_flip_classifier_was_not_trained_with(project, tmp_path, raw_sessions):
    """L13: "Compute extraction" runs extraction's flip-classifier check. A filter tuned in the tool
    to a value the classifier was not trained with gets extraction's warning, naming the recording,
    the setting and both values, and the preview is still computed and shown."""
    import warnings

    classifier = _classifier_trained_at(tmp_path / "flip", project / "config.yaml")
    _, widget = _save_then_change_config(project, raw_sessions, {"flip_classifier": classifier})
    data = widget.session_data
    data.frames_to_extract = 5
    recording = str(project / "saline_example_0" / "depth.avi")

    def preview():
        data.images = {**data.images, "Extracted mouse": None}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            click(buttons_of(widget.view)["Compute extraction"])
        assert data.images["Extracted mouse"] is not None and len(data.images["Extracted mouse"]) == 5
        from moseq2_extract.helpers.wrappers import FlipClassifierCleaningWarning

        return [str(w.message) for w in caught if issubclass(w.category, FlipClassifierCleaningWarning)]

    assert preview() == []

    data.spatial_filter = "[5]"
    [message] = preview()
    assert classifier in message and recording in message
    assert "spatial_filter_size is [5] now, [3] in training" in message
    assert "session_config.yaml" in message


def test_config_yaml_changes_after_the_tool_reach_the_arena_shown(project, monkeypatch, raw_sessions):
    """ "Compute arena mask" finds the arena with config.yaml's settings outside the tool's,
    and writes the background into the folder extraction writes to, the recording's session folder."""
    saved, widget = _save_then_change_config(
        project, raw_sessions, {"bg_roi_erode": [3, 3], "bg_roi_fill_holes": False}
    )
    assert "bg_roi_fill_holes" not in saved
    data = widget.session_data
    data.mask_dilations = 2

    calls = []
    _recorded(monkeypatch, "find_arena_candidates", calls)
    click(buttons_of(widget.view)["Compute arena mask"])

    assert len(calls) == 1
    (_, received, output_dir), _ = calls[0]
    assert tuple(received["bg_roi_erode"]) == (3, 3)
    assert received["bg_roi_fill_holes"] is False
    assert received["dilate_iterations"] == 2
    assert output_dir == str(project / "aggregate_results" / "saline_example_0")
    assert (project / "aggregate_results" / "saline_example_0" / "bground.tiff").exists()


def test_background_is_the_same_whether_the_arena_tool_ran_first(project, tmp_path):
    """X12: extraction computes the background it subtracts itself, sampling the recording at
    the one stride the arena tool shares, and never reads one back from the session folder. A
    recording whose arena the tool found first is extracted against the background a recording
    the tool never opened gets."""
    import h5py
    import numpy as np

    untouched = tmp_path / "untouched"
    shutil.copytree(project, untouched)
    widget = open_arena_tool(project)
    click(buttons_of(widget.view)["Compute arena mask"])

    backgrounds = {}
    for root in (project, untouched):
        extract_command(str(root / "saline_example_0" / "depth.avi"), None, str(root / "config.yaml"),
                        num_frames=50)
        with h5py.File(root / "aggregate_results" / "saline_example_0" / "results.h5", "r") as f:
            backgrounds[root] = f["metadata/extraction/background"][()]

    np.testing.assert_array_equal(backgrounds[project], backgrounds[untouched])
