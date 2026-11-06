"""``make-crowd-movies``, the syllable labeler and the crowd-movie comparison tool render
with one set of defaults, ``moseq2_viz.options.CrowdMovieOptions``.

The defaults: examples of at least 4 frames, 20 examples, 10 frames of padding, a colour
range of 10 to 120 mm, and as many workers as there are CPUs available to the process
(its CPU affinity where the operating system reports one, as on Linux and inside a SLURM
allocation smaller than the node; the machine's count on macOS and Windows, which do not).
A setting in config.yaml's ``crowd_movies`` section changes it for all three. Each front
end still chooses what it renders: the syllables, and for the comparison tool how the
movies are grouped and the number of examples its slider sets.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from moseq2_common.yaml import read_yaml

#: The defaults the three front ends share.
DEFAULTS = {"min_dur": 4, "max_examples": 20, "pad": 10, "min_height": 10, "max_height": 120}

#: What each front end sets for itself: which syllables, grouped how. The wrapper adds
#: ``crowd_syllables``.
CHOSEN = {"max_syllable", "specific_syllable", "separate_by", "session_names", "crowd_syllables"}


def _movie(crowd_movie_dir, syllable):
    return str(Path(crowd_movie_dir) / f"syllable_sorted-id-{syllable:02d}_(usage)_original-id-{syllable:02d}.mp4")


@pytest.fixture
def project(tmp_path, analysis_project_1_6):
    """A copy of the analysis project's settings and bookkeeping, reading the analysis project's files.

    Only the two yaml files are copied: the sessions they list are found through the
    bookkeeping file's paths, which are made absolute, so the model and sessions are read
    where they are and the labeler's outputs land in the copy.
    """
    from moseq2_common.yaml import write_yaml

    root = tmp_path / "project"
    (root / "models").mkdir(parents=True)
    shutil.copy2(analysis_project_1_6 / "config.yaml", root / "config.yaml")
    shutil.copy2(analysis_project_1_6 / "models" / "model.p", root / "models" / "model.p")
    bookkeeping = read_yaml(str(analysis_project_1_6 / "moseq2-project.yaml"))
    for session in bookkeeping["sessions"].values():
        session["folder"] = str(analysis_project_1_6 / session["folder"])
    bookkeeping["pca"] = {name: str(analysis_project_1_6 / path) for name, path in bookkeeping["pca"].items()}
    write_yaml(bookkeeping, str(root / "moseq2-project.yaml"))
    return root


@pytest.fixture
def rendered(monkeypatch, tmp_path):
    """The settings each call of the crowd-movie wrapper is handed, in call order."""
    import moseq2_viz.steps as viz_steps

    calls = []
    groups = []
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"")

    def wrapper(index_file, model_path, output_dir, config_data):
        calls.append(dict(config_data))
        if config_data["specific_syllable"] is None:
            # make-crowd-movies leaves the count to the wrapper, which this stub does not compute
            return {"all": [_movie(output_dir, s) for s in range(config_data["max_syllable"] or 3)]}
        return {group: [str(movie)] for group in groups}

    monkeypatch.setattr(viz_steps, "make_crowd_movies_wrapper", wrapper)
    # the comparison tool's groups, filled in once the tool knows them
    return calls, groups


def _cpus():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def test_three_crowd_movie_front_ends_share_one_set_of_defaults(project, analysis_project_1_6, rendered, tmp_path,
                                                                monkeypatch, displayed):
    from moseq2_app.main import label_syllables
    from moseq2_app.viz import controller
    from moseq2_app.viz.controller import CrowdMovieComparison
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project
    from moseq2_viz.cli import cli
    from moseq2_viz.options import CrowdMovieOptions

    calls, groups = rendered
    monkeypatch.setattr(controller, "get_video_info", lambda path: {"dims": (100, 100)})

    reference = CrowdMovieOptions().as_dict()
    assert {k: reference[k] for k in DEFAULTS} == DEFAULTS
    assert reference["processes"] == _cpus()

    # make-crowd-movies, with nothing typed but the files
    result = CliRunner().invoke(cli, ["make-crowd-movies", str(analysis_project_1_6 / "models" / "model.p"),
                                      "--project-file", str(analysis_project_1_6 / "moseq2-project.yaml"),
                                      "-o", str(tmp_path / "cli")], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    # the Analysis notebook's labeler cell
    label_syllables(str(project), max_syllables=3)

    # the comparison tool, previewing a syllable by group at its slider's number of examples
    comparison = CrowdMovieComparison(
        crowd_movie_settings={}, index=session_index(Project.at(project)),
        model_path=str(project / "models" / "model.p"),
        syll_info={i: {"label": f"s{i}", "desc": "", "crowd_movie_path": ""} for i in range(3)},
        output_dir=str(tmp_path / "comparison") + "/", get_pdfs=False,
    )
    groups.extend(comparison.groups)
    comparison.crowd_movie_preview(comparison.cm_syll_select.value, "group", comparison.num_examples.value)

    assert len(calls) == 3, calls
    shared = sorted(set(reference) - CHOSEN)
    for front_end, settings in zip(("make-crowd-movies", "labeler", "comparison"), calls):
        assert {k: settings[k] for k in shared} == {k: reference[k] for k in shared}, front_end


def test_the_crowd_movies_section_reaches_the_labeler_and_the_comparison_tool(project, rendered, tmp_path,
                                                                             monkeypatch, displayed):
    """The Analysis notebook's labeler and comparison tool render with the settings config.yaml's
    ``crowd_movies`` section holds, as ``make-crowd-movies`` does."""
    import ipywidgets

    from moseq2_app.main import interactive_crowd_movie_comparison, label_syllables
    from moseq2_app.util import update_config
    from moseq2_app.viz import controller
    from moseq2_app.viz.controller import CrowdMovieComparison

    calls, groups = rendered
    monkeypatch.setattr(controller, "get_video_info", lambda path: {"dims": (100, 100)})
    with update_config(str(project / "config.yaml"), section="crowd_movies") as section:
        section.update({"max_examples": 7, "min_dur": 6})

    label_syllables(str(project), max_syllables=3)
    interactive_crowd_movie_comparison(str(project), str(tmp_path / "comparison"), get_pdfs=False)
    tool = next(callback.__self__ for button in displayed.of_type(ipywidgets.Button)
                for callback in button._click_handlers.callbacks
                if isinstance(getattr(callback, "__self__", None), CrowdMovieComparison))
    groups.extend(tool.groups)
    tool.crowd_movie_preview(tool.cm_syll_select.value, "group", tool.num_examples.value)

    labeler, compared = calls[0], calls[-1]
    assert (labeler["max_examples"], labeler["min_dur"]) == (7, 6)
    assert compared["min_dur"] == 6


@pytest.mark.parametrize("affinity", [True, False], ids=["affinity", "no-affinity"])
def test_crowd_movie_workers_follow_the_cpus_available_to_the_process(affinity, rendered, tmp_path, monkeypatch):
    """Inside a 3-CPU allocation of a 64-CPU machine, 3 workers; where the OS reports no
    affinity (macOS, Windows), the machine's count."""
    from moseq2_viz.cli import cli
    from moseq2_viz.options import CrowdMovieOptions

    calls, _ = rendered
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    if affinity:
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 5, 9}, raising=False)
        expected = 3
    else:
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        expected = 64

    assert CrowdMovieOptions().processes == expected

    from configs import write_bookkeeping

    (tmp_path / "model.p").write_text("placeholder")  # an empty input stops a step
    write_bookkeeping(tmp_path)
    project_file = str(tmp_path / "moseq2-project.yaml")
    result = CliRunner().invoke(cli, ["make-crowd-movies", str(tmp_path / "model.p"), "--project-file", project_file,
                                      "--specific-syllable", "0", "-o", str(tmp_path / "out")],
                                catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert calls[-1]["processes"] == expected

    # a number typed on the command line is used as given
    result = CliRunner().invoke(cli, ["make-crowd-movies", str(tmp_path / "model.p"), "--project-file", project_file,
                                      "--specific-syllable", "0", "--processes", "2", "-o", str(tmp_path / "out")],
                                catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert calls[-1]["processes"] == 2
