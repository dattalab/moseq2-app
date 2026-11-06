"""The labeler's crowd movies are a cache, and its labels belong to one model.

Rendering a crowd movie for every syllable is the longest step of the Analysis
notebook, so a second run of the labeling cell reuses the movies on disk. It
reuses them only while what they were rendered from still holds: the model
(by its fingerprint), the ``crowd_movies`` settings and the sessions. The
crowd-movie folder's record (``crowd_movies.yaml``, beside the folder) says
what that was; when any of the three has changed, the cell renders the movies
again and says why.

Syllable labels are keyed by usage-sorted syllable number, so they describe one
fit of one model. ``syll_info.yaml`` records the fingerprint of the model it
labels, and a labeler opened on a model that is not that one, or is newer than
the labels, says so and asks: wipe the labels and continue, or stop so the
person can keep the old labels themselves.
"""

from __future__ import annotations

import os
import shutil

import ipywidgets as widgets
import pytest

import moseq2_app.steps as app_steps
from moseq2_app.main import label_syllables
from moseq2_app.util import current_analysis, update_config
from moseq2_common.records import fingerprint
from moseq2_common.yaml import read_yaml, write_yaml

from widgets import capture_display

#: What ``syll_info.yaml`` holds the labelled model's path and fingerprint under.
LABELS_FOR = "labels_for"

#: Few syllables, so a re-render is quick; the movies the labeled fixture made cover them.
FEW = {"max_syllables": 2, "n_explained": 99}


@pytest.fixture
def copy(tmp_path, analysis_project, labeled):
    """The labeled project, copied with its file times, as a user's own project to change."""
    root = tmp_path / "project"
    shutil.copytree(analysis_project, root)
    return current_analysis(str(root))


@pytest.fixture
def renders(monkeypatch):
    """Every crowd-movie render the labeler asks for."""
    calls = []
    real = app_steps.render_syllable_movies

    def recorder(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(app_steps, "render_syllable_movies", recorder)
    return calls


def _newer(path, seconds=120):
    stat = os.stat(path)
    os.utime(path, (stat.st_atime + seconds, stat.st_mtime + seconds))


def test_crowd_movies_are_reused_while_nothing_they_were_made_from_changed(copy, renders, capsys):
    label_syllables(str(copy.project.folder), **FEW)

    assert renders == []
    assert "again" not in capsys.readouterr().out


def test_crowd_movies_rerender_when_the_model_changes(copy, renders, capsys):
    # labels made for the earlier model stop the labeler first (the last test); here there are none
    copy.syll_info.unlink()
    _newer(copy.model)

    label_syllables(str(copy.project.folder), **FEW)

    assert len(renders) == 1
    printed = capsys.readouterr().out
    assert f"Rendering the crowd movies in {copy.crowd_movies} again: the model changed since they were made" \
        in printed


def test_crowd_movies_rerender_when_their_settings_change(copy, renders, capsys):
    with update_config(str(copy.project.config_file), section="crowd_movies") as settings:
        settings["max_examples"] = 5

    label_syllables(str(copy.project.folder), **FEW)

    assert len(renders) == 1
    assert "the crowd_movies settings changed: max_examples" in capsys.readouterr().out


def test_crowd_movies_rerender_when_the_sessions_change(copy, renders, capsys):
    from moseq2_common.bookkeeping import load

    left_out = sorted(load(copy.project).sessions)[0]
    config = read_yaml(str(copy.project.config_file))
    config["exclude"] = [left_out]
    write_yaml(config, str(copy.project.config_file))

    label_syllables(str(copy.project.folder), **FEW)

    assert len(renders) == 1
    assert "the sessions changed" in capsys.readouterr().out


def test_the_labels_name_the_model_they_label(copy):
    label_syllables(str(copy.project.folder), **FEW)

    from moseq2_app.labels import read_syllable_labels

    stored = read_yaml(str(copy.syll_info))[LABELS_FOR]
    assert stored["path"] == str(copy.model)
    assert stored["fingerprint"] == fingerprint(copy.model)
    assert LABELS_FOR not in read_syllable_labels(str(copy.syll_info))


def _choice(display):
    """The two buttons of the stale-labels prompt, by label."""
    return {w.description: w for w in display.of_type(widgets.VBox)[-1].children[1].children}


def test_labels_for_another_model_are_not_attached_silently(copy, monkeypatch):
    folder = str(copy.project.folder)
    label_syllables(folder, **FEW)
    labels = read_yaml(str(copy.syll_info))
    labels[0]["label"] = "rear"
    write_yaml(labels, str(copy.syll_info))
    kept = copy.syll_info.read_bytes()
    _newer(copy.model)

    display = capture_display(monkeypatch)
    label_syllables(folder, **FEW)

    assert display.of_type(widgets.Dropdown) == [], "the labeler opened on labels made for another model"
    shown = display.of_type(widgets.VBox)[-1].children[0].value
    assert f"The labels in {copy.syll_info} were made for {copy.model}" in shown
    assert f"{copy.model} is newer than the labels" in shown
    choice = _choice(display)
    assert sorted(choice) == ["Stop", "Wipe labels and continue"]

    choice["Stop"].click()
    assert display.of_type(widgets.Dropdown) == []
    assert copy.syll_info.read_bytes() == kept

    display.clear()
    label_syllables(folder, **FEW)
    _choice(display)["Wipe labels and continue"].click()

    from moseq2_app.labels import read_syllable_labels

    dropdown = display.one(widgets.Dropdown)
    assert list(dropdown.options)[0] == "0 - "
    assert read_syllable_labels(str(copy.syll_info))[0]["label"] == ""
    assert read_yaml(str(copy.syll_info))[LABELS_FOR]["fingerprint"] == fingerprint(copy.model)
