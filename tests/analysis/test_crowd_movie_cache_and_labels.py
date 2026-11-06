"""The labeler's crowd movies are a cache, and its labels belong to one model.

Rendering a crowd movie for every syllable is the longest step of the Analysis
notebook, so a second run of the labeling cell reuses the movies on disk. It
reuses them only while what they were rendered from still holds: the model
(the run its record names), the ``crowd_movies`` settings and the sessions. The
crowd-movie folder's record (``crowd_movies.yaml``, beside the folder) says
what that was; when any of the three has changed, the cell renders the movies
again and says why.

Syllable labels are keyed by usage-sorted syllable number, so they describe one
fit of one model. Saving them is a step with its own record,
``syll_info.record.yaml`` beside ``syll_info.yaml``, naming the model, the run
that fit it, and the save time; ``syll_info.yaml`` holds the labels only. A
labeler opened on a model fit again since, or on another model, says so and
asks: wipe the labels and continue, or stop so the person can keep the old
labels themselves.

Nothing here reads a file's modification time: a project copied elsewhere with
new file times keeps its movies and its labels, and its progress display names
nothing as out of date.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime

import ipywidgets as widgets
import pytest

import moseq2_app.steps as app_steps
from moseq2_app.main import label_syllables
from moseq2_app.util import current_analysis, update_config
from moseq2_common.yaml import read_yaml, write_yaml

from widgets import capture_display

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


def fit_again(analysis):
    """What fitting the model again leaves behind, as far as any check can tell: the model carrying the
    record of a new learn-model run, its inputs the project's settings and every session's extraction."""
    from moseq2_common.bookkeeping import load, session_h5
    from moseq2_common.records import start_record, write_record

    project = analysis.project
    sessions = [session_h5(project, s) for s in load(project).sessions.values()]
    record = start_record("learn-model", {}, [project.config_file, *sessions], project=project)
    write_record(analysis.model, record.finish({"files": [str(analysis.model)]}))


def refit_unlabelled(analysis):
    """The model fit again and its old labels deleted, as after wiping them: the start of a labeling."""
    analysis.syll_info.unlink(missing_ok=True)
    labels_record(analysis).unlink(missing_ok=True)
    fit_again(analysis)


def fit_date(model):
    """When the run that wrote ``model`` ended, as the labeler's message gives it."""
    from moseq2_common.records import run_of

    return datetime.fromisoformat(run_of(model)["ended"]).strftime("%Y-%m-%d %H:%M")


def labels_record(analysis):
    return analysis.syll_info.with_name("syll_info.record.yaml")


def copied_with_new_times(analysis, destination):
    """``analysis``'s project copied to ``destination`` with every file's time moved a day on."""
    shutil.copytree(analysis.project.folder, destination)
    for folder, _, names in os.walk(destination):
        for name in names:
            path = os.path.join(folder, name)
            stat = os.stat(path)
            os.utime(path, (stat.st_atime + 86400, stat.st_mtime + 86400))
    return current_analysis(str(destination))


def test_crowd_movies_are_reused_while_nothing_they_were_made_from_changed(copy, renders, capsys):
    label_syllables(str(copy.project.folder), **FEW)

    assert renders == []
    assert "again" not in capsys.readouterr().out


def test_crowd_movies_rerender_when_the_model_changes(copy, renders, capsys):
    # labels made for the earlier model stop the labeler first (the last test); here there are none
    copy.syll_info.unlink()
    fit_again(copy)

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


def test_saving_labels_writes_a_record_naming_the_model_run(copy):
    """Saving the labels writes ``syll_info.record.yaml`` beside them: the save step, its time, and the
    model it labels with the run that fit it. ``syll_info.yaml`` holds the labels only."""
    from moseq2_common.records import read_record, run_of

    refit_unlabelled(copy)
    label_syllables(str(copy.project.folder), **FEW)

    assert "labels_for" not in read_yaml(str(copy.syll_info))
    assert labels_record(copy).is_file()
    record = read_record(copy.syll_info)
    assert record == read_yaml(str(labels_record(copy)))
    assert record["step"] == "save-syllable-labels"
    assert record["started"] and record["ended"]
    [model] = record["inputs"]
    assert model["path"] == str(copy.model)
    assert model["relative_path"] == copy.model.relative_to(copy.project.folder).as_posix()
    assert model["run"] == run_of(copy.model)


def test_a_copied_project_with_new_file_times_is_not_out_of_date(copy, tmp_path, renders, capsys, monkeypatch):
    """Copied with ``shutil.copytree`` and every file time changed, the project keeps its crowd movies and
    its labels, and its progress display names nothing to run again."""
    from moseq2_app.project import progress

    refit_unlabelled(copy)
    label_syllables(str(copy.project.folder), **FEW)
    renders.clear()
    capsys.readouterr()

    moved = copied_with_new_times(copy, tmp_path / "moved")
    display = capture_display(monkeypatch)
    label_syllables(str(moved.project.folder), **FEW)

    printed = capsys.readouterr().out
    assert renders == [], printed
    assert display.of_type(widgets.Dropdown), "the labeler did not open on the copied labels"
    assert "again" not in printed and "were made for" not in printed, printed
    assert "again" not in progress(moved.project)


def _choice(display):
    """The two buttons of the stale-labels prompt, by label."""
    return {w.description: w for w in display.of_type(widgets.VBox)[-1].children[1].children}


def test_labels_for_a_model_fit_again_are_not_attached_silently(copy, monkeypatch):
    folder = str(copy.project.folder)
    refit_unlabelled(copy)
    label_syllables(folder, **FEW)
    labelled_on = fit_date(copy.model)
    labels = read_yaml(str(copy.syll_info))
    labels[0]["label"] = "rear"
    write_yaml(labels, str(copy.syll_info))
    kept = copy.syll_info.read_bytes()
    fit_again(copy)

    display = capture_display(monkeypatch)
    label_syllables(folder, **FEW)

    assert display.of_type(widgets.Dropdown) == [], "the labeler opened on labels made for another fit"
    shown = display.of_type(widgets.VBox)[-1].children[0].value
    assert (f"The labels in {copy.syll_info} were made for {copy.model} as fit on {labelled_on}, and "
            f"{copy.model} was fit again on {fit_date(copy.model)}.") in shown, shown
    choice = _choice(display)
    assert sorted(choice) == ["Stop", "Wipe labels and continue"]

    choice["Stop"].click()
    assert display.of_type(widgets.Dropdown) == []
    assert copy.syll_info.read_bytes() == kept

    display.clear()
    label_syllables(folder, **FEW)
    _choice(display)["Wipe labels and continue"].click()

    from moseq2_app.labels import read_syllable_labels
    from moseq2_common.records import read_record, run_of

    dropdown = display.one(widgets.Dropdown)
    assert list(dropdown.options)[0] == "0 - "
    assert read_syllable_labels(str(copy.syll_info))[0]["label"] == ""
    assert read_record(copy.syll_info)["inputs"][0]["run"] == run_of(copy.model)
