"""Labeling frames and training a flip classifier, as the notebook does it.

The Flip-Classifier-Training notebook opens a labeling widget on the
``aggregate_results/`` folder the extraction notebook wrote. A user picks a
session, moves to a frame, clicks Start Range, steps Forward, and clicks
Facing Left or Facing Right. Each finished range is written to
``flip-training-frame-ranges.p`` beside that folder. The next cells save the
ranges again, read the cleaning parameters from the extraction config, and
train a classifier on the frames the file names.

These tests run the notebook's own cells, with only the three paths a user
types replaced, against the extraction chain's output copied into the test's
own folder. The cells that would change the extractions in place are not run.
"""

from __future__ import annotations

import pickle
import re
import shutil

import h5py
import numpy as np
import pytest

from cells import code_cell

NOTEBOOK = "Flip-Classifier-Training-Notebook.ipynb"

#: The notebook's code cells, numbered from 1 as ``code_cell`` counts them.
SETUP, SHOW, SAVE, CLEAN, TRAIN = 1, 2, 3, 4, 5


def press(button):
    """Click a Panel button the way the browser does.

    The labeling widget wires its buttons with ``on_click``, which listens for
    ``clicks`` rather than the ``value`` that ``widgets.click`` sets. A click
    in the browser reaches the button as a ``ButtonClick`` event, which
    ``_process_event`` turns into both.
    """
    button._process_event(None)


def run_cell(number, namespace, **paths):
    """Run one of the notebook's code cells in ``namespace``.

    ``paths`` replaces the value of each named assignment in the cell, which is
    how a user fills in the placeholder paths before running it.
    """
    source = code_cell(NOTEBOOK, number)
    for name, value in paths.items():
        source, count = re.subn(
            rf"^{name} = .*$", f"{name} = {str(value)!r}", source, flags=re.MULTILINE
        )
        assert count == 1, f"cell {number} no longer assigns {name}"
    exec(source, namespace)
    return namespace


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
    run_cell(CLEAN, notebook)
    run_cell(TRAIN, notebook)

    with np.load(notebook["train_path"]) as data:
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
    run_cell(CLEAN, notebook)
    from moseq2_app.flip.train import create_training_dataset

    with np.load(create_training_dataset(widget.training_data_path)) as data:
        trained = data["frames"][:10]
    with h5py.File(widget.sessions[session], "r") as f:
        np.testing.assert_array_equal(trained, f["frames"][5:15])


def test_training_with_no_labeled_frames_says_what_to_do(notebook):
    """Running the save and training cells before adding any range.

    Numpy's ``need at least one array to concatenate`` says nothing about the
    cause. The message names the file training read and tells the user to add
    a range first.
    """
    run_cell(SAVE, notebook)
    run_cell(CLEAN, notebook)
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
