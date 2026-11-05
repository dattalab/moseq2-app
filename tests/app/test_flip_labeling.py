"""Labeling frames and training a flip classifier, as the notebook does it.

The Flip-Classifier-Training notebook opens a labeling widget on the
``aggregate_results/`` folder the extraction notebook wrote. A user picks a
session, moves to a frame, clicks Start Range, steps Forward, and clicks
Facing Left or Facing Right. The widget holds each finished range until the
user clicks Save Labels or runs the save cell, which writes them to
``flip-training-frame-ranges.p`` beside that folder. The training cell trains a
classifier on the frames the file names, each session's cleaned with the
settings its recording is extracted with, from the extraction config.

These tests run the notebook's own cells, with only the three paths a user
types replaced, against the extraction chain's output copied into the test's
own folder. The cells that would change the extractions in place are not run.
"""

from __future__ import annotations

import pickle
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from cells import code_cell

NOTEBOOK = "Flip-Classifier-Training-Notebook.ipynb"

#: The notebook's code cells, numbered from 1 as ``code_cell`` counts them.
SETUP, SHOW, SAVE, TRAIN = 1, 2, 3, 4


def press(button):
    """Click a Panel button the way the browser does.

    The labeling widget wires its buttons with ``on_click``, which listens for
    ``clicks`` rather than the ``value`` that ``widgets.click`` sets. A click
    in the browser reaches the button as a ``ButtonClick`` event, which
    ``_process_event`` turns into both.
    """
    button._process_event(None)


def fill_in(source, **paths):
    """``source`` with the value of each named assignment replaced by its path,
    which is how a user fills in the placeholder paths before running a cell."""
    for name, value in paths.items():
        # a function, not a string: re.subn reads a replacement string as a
        # template and turns each escaped backslash in the repr back into one
        assignment = f"{name} = {str(value)!r}"
        source, count = re.subn(
            rf"^{name} = .*$", lambda _: assignment, source, flags=re.MULTILINE
        )
        assert count == 1, f"the cell no longer assigns {name}"
    return source


def run_cell(number, namespace, **paths):
    """Run one of the notebook's code cells in ``namespace``, its ``paths`` filled in."""
    exec(fill_in(code_cell(NOTEBOOK, number), **paths), namespace)
    return namespace


def test_a_windows_path_is_filled_in_as_typed():
    """On Windows every path holds backslashes, which must reach the cell as written."""
    path = r"C:\Users\runneradmin\AppData\Local\Temp\project0\config.yaml"
    namespace = {}
    exec(fill_in("config_path = ''\n", config_path=path), namespace)
    assert namespace["config_path"] == path


def open_notebook(tmp_path, configured):
    """Run the notebook's first two cells in a fresh namespace, as after a restart."""
    namespace = run_cell(
        SETUP,
        {},
        config_path=configured,
        input_dir=tmp_path / "aggregate_results",
        model_path=tmp_path / "flip-classifier.pkl",
    )
    run_cell(SHOW, namespace)
    return namespace


@pytest.fixture
def notebook(tmp_path, aggregated, configured):
    """The notebook after its first cell, pointed at a copy of the extractions."""
    import panel as pn

    pn.extension()
    shutil.copytree(aggregated.parent / "aggregate_results", tmp_path / "aggregate_results")
    return open_notebook(tmp_path, configured)


def label(widget, session, start, stop, left):
    """Label frames ``start`` to ``stop`` of ``session`` the way the instructions say."""
    widget.session_select_dropdown.value = session
    widget.frame_num_slider.value = start
    press(widget.start_button)
    for _ in range(stop - start):
        press(widget.forward_button)
    press(widget.face_left_button if left else widget.face_right_button)


def test_labeled_ranges_reach_the_trained_classifier(notebook, tmp_path):
    """Two labeled ranges are what training trains on, and the saved classifier works.

    Training uses each labeled frame, its 180 degree rotation, a shifted copy
    of both and a cleaned copy of both: six frames per labeled frame.
    """
    widget = notebook["FF"]
    sessions = list(widget.sessions)
    label(widget, sessions[0], 10, 40, left=False)
    label(widget, sessions[-1], 100, 120, left=True)
    assert widget.curr_total_selected_frames == 50

    run_cell(SAVE, notebook)
    run_cell(TRAIN, notebook)

    with np.load(notebook["trained"].training_data) as data:
        assert len(data["frames"]) == 6 * 50

    # The classifier is read back the way the notebook's next cell reads it.
    from moseq2_extract.extract.proc import get_flips

    with h5py.File(widget.sessions[sessions[0]], "r") as f:
        frames = f["frames"][:20]
    flips = get_flips(frames, flip_file=str(tmp_path / "flip-classifier.pkl"))
    assert flips.shape == (20,)


def test_a_right_facing_range_is_trained_as_it_was_shown(notebook):
    """The frames the user saw are the frames training reads, the right way round."""
    widget = notebook["FF"]
    session = list(widget.sessions)[0]
    label(widget, session, 5, 15, left=False)
    run_cell(SAVE, notebook)
    from moseq2_app.flip.train import create_training_dataset
    from moseq2_common.frames import CleaningOptions

    with np.load(create_training_dataset(widget.training_data_path, CleaningOptions())) as data:
        trained = data["frames"][:10]
    with h5py.File(widget.sessions[session], "r") as f:
        np.testing.assert_array_equal(trained, f["frames"][5:15])


#: Cleaning settings unlike extraction's defaults in each way the flip notebook's
#: own mapping lost: the tail and cable filters' shapes, the cable erosion, the
#: median kernel and the height threshold.
CLEANING = {"spatial_filter_size": [5], "tail_filter_shape": "rectangle", "tail_filter_size": [7, 7],
            "cable_filter_shape": "ellipse", "cable_filter_size": [3, 3], "cable_filter_iters": 1,
            "min_height": 12}


def cleaned_as_extraction_cleans(frames, config):
    """``frames`` cleaned the way ``extract_chunk`` cleans a chunk with ``config``'s settings.

    The height threshold, then ``clean_frames`` with the structuring elements
    ``select_strel`` builds from the shapes and sizes.
    """
    from moseq2_common.frames import select_strel
    from moseq2_extract.extract.proc import clean_frames, threshold_chunk

    strels = {"strel_tail": select_strel(config["tail_filter_shape"], tuple(config["tail_filter_size"])),
              "strel_min": select_strel(config["cable_filter_shape"], tuple(config["cable_filter_size"]))}
    return clean_frames(
        threshold_chunk(frames.copy(), config["min_height"], config["max_height"]),
        prefilter_space=tuple(config["spatial_filter_size"]),
        prefilter_time=tuple(config["temporal_filter_size"]),
        strel_tail=strels["strel_tail"],
        iters_tail=config["tail_filter_iters"],
        strel_min=strels["strel_min"],
        iters_min=config["cable_filter_iters"],
        frame_dtype=frames.dtype,
    )


def train_with_the_script(frame_ranges_file, *args):
    """Run ``scripts/train_flip_classifier.py`` on the saved labels, from the command line."""
    import importlib.util

    from click.testing import CliRunner

    path = Path(__file__).resolve().parents[2] / "scripts" / "train_flip_classifier.py"
    spec = importlib.util.spec_from_file_location("train_flip_classifier", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    result = CliRunner().invoke(script.main, [frame_ranges_file, "--classifier-type", "rf", *args])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("front_end", ["notebook", "script", "script_config_file"])
def test_flip_training_cleans_frames_as_extraction_does(notebook, tmp_path, configured, front_end):
    """The cleaned third of the training set is what extraction makes of the same frames.

    Extraction cleans each frame before the classifier sees it, so a classifier
    trained on frames cleaned any other way is applied to frames unlike its
    training set. The notebook, and the script given ``--config-file``, read
    extraction's settings from the config file, here set away from the defaults
    in every cleaning setting; the script without one cleans with extraction's
    defaults.
    """
    from moseq2_common.yaml import read_yaml, write_yaml

    widget = notebook["FF"]
    session = list(widget.sessions)[0]
    label(widget, session, 10, 40, left=False)
    run_cell(SAVE, notebook)
    whole = dict(read_yaml(str(configured)))
    # extraction's settings, which the cleaning settings are among
    config = dict(whole["extract"])
    if front_end == "notebook":
        config.update(CLEANING)
        notebook["config_path"] = str(tmp_path / "config.yaml")
        write_yaml({**whole, "extract": config}, notebook["config_path"])
        run_cell(TRAIN, notebook)
    elif front_end == "script_config_file":
        config.update(CLEANING)
        write_yaml({**whole, "extract": config}, str(tmp_path / "config.yaml"))
        train_with_the_script(widget.training_data_path, "--config-file", str(tmp_path / "config.yaml"))
    else:
        train_with_the_script(widget.training_data_path)

    with h5py.File(widget.sessions[session], "r") as f:
        labeled = f["frames"][10:40]
    both_ways = np.concatenate((labeled, np.rot90(labeled, 2, axes=(1, 2))), axis=0)
    # pixels at the height threshold itself, which extraction keeps
    assert (both_ways == config["min_height"]).any()
    with np.load(Path(widget.training_data_path).with_name("training_data.npz")) as data:
        cleaned = data["frames"][2 * len(both_ways):]
    np.testing.assert_array_equal(cleaned, cleaned_as_extraction_cleans(both_ways, config))


def test_training_with_no_labeled_frames_says_what_to_do(notebook):
    """Running the save and training cells before adding any range.

    Numpy's ``need at least one array to concatenate`` says nothing about the
    cause. The message names the file training read and tells the user to add
    a range first.
    """
    run_cell(SAVE, notebook)
    path = notebook["FF"].training_data_path
    with pytest.raises(ValueError) as raised:
        run_cell(TRAIN, notebook)
    message = str(raised.value)
    assert path in message
    assert "Start Range" in message
    assert "Facing Left or Facing Right" in message


#: What the widget says while what it shows differs from the saved file, and
#: once the two agree again.
UNSAVED = "You have unsaved changes to your labels. Hit save when done editing to keep them."
SAVED = "Flip labels saved."


def saved_ranges(widget):
    """The ranges the file holds, by session, as ``(left, start, stop)``."""
    with open(widget.training_data_path, "rb") as f:
        _, ranges = pickle.load(f)
    return {
        session: [(left, r.start, r.stop) for left, r in labeled]
        for session, labeled in ranges.items()
        if labeled
    }


def status(widget):
    """The widget's status line: its text and its colour."""
    html = widget.status.object
    text = re.sub(r"<[^>]+>", "", html)
    colour = re.search(r"color:\s*(\w+)", html)
    return text, colour.group(1) if colour else None


def test_saved_labels_are_back_after_a_restart(notebook, tmp_path, configured):
    """Restart the kernel, run the cells from the top, then the save cell.

    The widget opens with the ranges the file holds, so they are on screen
    to keep labeling from, and saving again writes them back rather than an
    empty set.
    """
    widget = notebook["FF"]
    sessions = list(widget.sessions)
    label(widget, sessions[0], 10, 40, left=False)
    label(widget, sessions[-1], 100, 120, left=True)
    run_cell(SAVE, notebook)
    before = saved_ranges(widget)
    assert before == {sessions[0]: [(False, 10, 40)], sessions[-1]: [(True, 100, 120)]}

    restarted = open_notebook(tmp_path, configured)
    widget = restarted["FF"]
    assert widget.curr_total_selected_frames == 50
    assert len(widget.selected_ranges.options) == 2
    assert status(widget) == ("", None)

    run_cell(SAVE, restarted)
    assert saved_ranges(widget) == before


def test_an_unsaved_range_is_not_written(notebook, tmp_path, configured):
    """Adding a range changes what the widget holds, not the file.

    A range added by mistake is undone by not saving: after a restart the
    file, and so the widget, hold only what was saved.
    """
    widget = notebook["FF"]
    sessions = list(widget.sessions)
    label(widget, sessions[0], 10, 40, left=False)
    run_cell(SAVE, notebook)
    label(widget, sessions[0], 50, 60, left=True)
    assert saved_ranges(widget) == {sessions[0]: [(False, 10, 40)]}

    widget = open_notebook(tmp_path, configured)["FF"]
    assert widget.curr_total_selected_frames == 30
    assert saved_ranges(widget) == {sessions[0]: [(False, 10, 40)]}


def test_a_deleted_range_is_gone_from_the_file_once_saved(notebook):
    """Delete a range, then save with the widget's button."""
    widget = notebook["FF"]
    sessions = list(widget.sessions)
    label(widget, sessions[0], 10, 40, left=False)
    label(widget, sessions[-1], 100, 120, left=True)
    press(widget.save_button)

    widget.selected_ranges.value = [widget.selected_ranges.options[0]]
    press(widget.delete_selection_button)
    assert widget.curr_total_selected_frames == 20
    assert saved_ranges(widget) == {
        sessions[0]: [(False, 10, 40)],
        sessions[-1]: [(True, 100, 120)],
    }

    press(widget.save_button)
    assert saved_ranges(widget) == {sessions[-1]: [(True, 100, 120)]}


def test_the_status_says_whether_the_labels_are_saved(notebook):
    """Red after every edit, green after either way of saving."""
    widget = notebook["FF"]
    session = list(widget.sessions)[0]

    label(widget, session, 10, 40, left=False)
    assert status(widget) == (UNSAVED, "red")
    press(widget.save_button)
    assert status(widget) == (SAVED, "green")

    widget.selected_ranges.value = [widget.selected_ranges.options[0]]
    press(widget.delete_selection_button)
    assert status(widget) == (UNSAVED, "red")
    run_cell(SAVE, notebook)
    assert status(widget) == (SAVED, "green")


def test_ranges_for_a_session_no_longer_shown_are_named(notebook, tmp_path, configured, capsys):
    """The file holds ranges for an extraction that is no longer in input_dir.

    The widget cannot show them, so it says which, keeps the file as it is,
    and marks the labels unsaved: saving writes what is on screen.
    """
    widget = notebook["FF"]
    session = list(widget.sessions)[0]
    label(widget, session, 10, 40, left=False)
    press(widget.save_button)
    with open(widget.training_data_path, "rb") as f:
        paths, ranges = pickle.load(f)
    gone = tmp_path / "elsewhere" / "gone_results_00.h5"
    paths["gone_results_00.h5"] = gone
    ranges["gone_results_00.h5"] = [(True, range(0, 5))]
    with open(widget.training_data_path, "wb") as f:
        pickle.dump((paths, ranges), f)

    widget = open_notebook(tmp_path, configured)["FF"]
    printed = capsys.readouterr().out
    assert "gone_results_00.h5" in printed
    assert str(gone) in printed
    assert widget.training_data_path in printed
    assert widget.curr_total_selected_frames == 30
    assert status(widget) == (UNSAVED, "red")
    assert "gone_results_00.h5" in saved_ranges(widget)

    press(widget.save_button)
    assert saved_ranges(widget) == {session: [(False, 10, 40)]}


def test_training_cleans_with_the_config_parameters(notebook, tmp_path, configured):
    """The cleaning cell reads the config; the training cell cleans with what it read.

    The notebook asks for the extraction's own parameters here, because the
    classifier is applied to frames extraction cleaned. A config that erodes
    the mouse twice for a cable gives a training set whose cleaned third is
    eroded twice too.
    """
    import cv2
    import yaml

    from moseq2_extract.extract.proc import clean_frames

    config = yaml.safe_load(configured.read_text())
    config["extract"]["cable_filter_iters"] = 2
    eroding = tmp_path / "config.yaml"
    eroding.write_text(yaml.safe_dump(config))
    notebook["config_path"] = str(eroding)

    widget = notebook["FF"]
    session = list(widget.sessions)[0]
    label(widget, session, 5, 15, left=False)
    run_cell(SAVE, notebook)
    run_cell(TRAIN, notebook)

    with h5py.File(widget.sessions[session], "r") as f:
        labeled = f["frames"][5:15]
    both_ways = np.concatenate((labeled, np.rot90(labeled, 2, axes=(1, 2))), axis=0)
    extraction = config["extract"]
    expected = clean_frames(
        np.where(both_ways > extraction["min_height"], both_ways, 0),
        extraction["spatial_filter_size"],
        extraction["temporal_filter_size"],
        strel_tail=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(extraction["tail_filter_size"])),
        iters_tail=extraction["tail_filter_iters"],
        strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, tuple(extraction["cable_filter_size"])),
        iters_min=2,
    )
    with np.load(notebook["trained"].training_data) as data:
        np.testing.assert_array_equal(data["frames"][-len(expected) :], expected)


#: What every message about unsaved labels tells the user to do.
SAVE_AND_RETRAIN = "click Save Labels (or run the notebook's save cell), then run training again"


def test_training_labels_never_saved_says_to_save(notebook):
    """Label a range and run training without saving: the ranges file was never written.

    Python's bare ``No such file or directory`` does not say that the ranges
    on screen reach training only once saved.
    """
    widget = notebook["FF"]
    label(widget, list(widget.sessions)[0], 10, 40, left=False)
    with pytest.raises(FileNotFoundError) as raised:
        run_cell(TRAIN, notebook)
    message = str(raised.value)
    assert widget.training_data_path in message
    assert "only once they are saved" in message
    assert SAVE_AND_RETRAIN in message


def test_training_on_an_empty_save_says_to_save(notebook):
    """Run the save cell before labeling, label a range, run training without saving again.

    The file holds no ranges, so training stops; the message says the ranges on
    screen are not saved yet, not only to add some.
    """
    widget = notebook["FF"]
    run_cell(SAVE, notebook)
    label(widget, list(widget.sessions)[0], 10, 40, left=False)
    with pytest.raises(ValueError) as raised:
        run_cell(TRAIN, notebook)
    message = str(raised.value)
    assert widget.training_data_path in message
    assert "only once they are saved" in message
    assert SAVE_AND_RETRAIN in message


def test_training_on_older_saved_labels_warns(notebook, recwarn):
    """Save one range, label another, run training without saving again.

    Training reads the file, so it trains on the saved range alone. It warns
    that the widget holds unsaved changes and says what it is training on;
    once the labels are saved, it trains on both and does not warn.
    """
    widget = notebook["FF"]
    session = list(widget.sessions)[0]
    label(widget, session, 10, 40, left=False)
    run_cell(SAVE, notebook)
    label(widget, session, 50, 60, left=True)

    with pytest.warns(UserWarning) as warned:
        run_cell(TRAIN, notebook)
    unsaved = [str(w.message) for w in warned if "unsaved" in str(w.message)]
    assert len(unsaved) == 1
    message = unsaved[0]
    assert widget.training_data_path in message
    assert "2 range(s) (40 frames)" in message
    assert "1 range(s) (30 frames)" in message
    assert SAVE_AND_RETRAIN in message
    with np.load(notebook["trained"].training_data) as data:
        assert len(data["frames"]) == 6 * 30

    run_cell(SAVE, notebook)
    recwarn.clear()
    run_cell(TRAIN, notebook)
    assert not [w for w in recwarn if "unsaved" in str(w.message)]
    with np.load(notebook["trained"].training_data) as data:
        assert len(data["frames"]) == 6 * 40
