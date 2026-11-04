"""``make-crowd-movies``, the syllable labeler and the crowd-movie comparison tool render
with one set of defaults, ``moseq2_viz.options.CrowdMovieOptions``.

The defaults: examples of at least 4 frames, 20 examples, 10 frames of padding, a colour
range of 10 to 120 mm, and as many workers as there are CPUs available to the process
(its CPU affinity where the operating system reports one, as on Linux and inside a SLURM
allocation smaller than the node; the machine's count on macOS and Windows, which do not).
Each front end still chooses what it renders: the syllables, how the movies are grouped,
and the number of examples the labeler's notebook cell or the comparison tool's slider sets.
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
def project(tmp_path, analysis_project):
    """A copy of the analysis project, with its progress file pointing into the copy."""
    shutil.copytree(analysis_project, tmp_path / "project")
    progress = tmp_path / "project" / "progress.yaml"
    progress.write_text(progress.read_text().replace(str(analysis_project), str(tmp_path / "project")))
    return read_yaml(str(progress))


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


def test_three_crowd_movie_front_ends_share_one_set_of_defaults(project, rendered, tmp_path, monkeypatch, displayed):
    from moseq2_app.gui.wrappers import interactive_syllable_labeler_wrapper
    from moseq2_app.viz import controller
    from moseq2_app.viz.controller import CrowdMovieComparison
    from moseq2_viz.cli import cli
    from moseq2_viz.options import CrowdMovieOptions

    calls, groups = rendered
    monkeypatch.setattr(controller, "get_video_info", lambda path: {"dims": (100, 100)})

    reference = CrowdMovieOptions().as_dict()
    assert {k: reference[k] for k in DEFAULTS} == DEFAULTS
    assert reference["processes"] == _cpus()

    # make-crowd-movies, with nothing typed but the files
    result = CliRunner().invoke(cli, ["make-crowd-movies", project["index_file"], project["model_path"],
                                      "-o", str(tmp_path / "cli")], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    # the Analysis notebook's labeler cell, with the number of examples the cell sets
    interactive_syllable_labeler_wrapper(
        project["model_path"], project["config_file"], project["index_file"],
        str(tmp_path / "crowd_movies") + "/", str(tmp_path / "syll_info.yaml"), str(tmp_path),
        max_syllables=3, max_examples=20,
    )

    # the comparison tool, previewing a syllable by group at its slider's number of examples
    comparison = CrowdMovieComparison(
        config_data=read_yaml(project["config_file"]), index_path=project["index_file"],
        model_path=project["model_path"],
        syll_info={i: {"label": f"s{i}", "desc": "", "crowd_movie_path": ""} for i in range(3)},
        output_dir=str(tmp_path / "comparison") + "/", get_pdfs=False,
    )
    groups.extend(comparison.groups)
    comparison.crowd_movie_preview(comparison.cm_syll_select.value, "group", comparison.num_examples.value)

    assert len(calls) == 3, calls
    shared = sorted(set(reference) - CHOSEN)
    for front_end, settings in zip(("make-crowd-movies", "labeler", "comparison"), calls):
        assert {k: settings[k] for k in shared} == {k: reference[k] for k in shared}, front_end


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

    for name in ("index.yaml", "model.p"):
        (tmp_path / name).write_text("")
    result = CliRunner().invoke(cli, ["make-crowd-movies", str(tmp_path / "index.yaml"), str(tmp_path / "model.p"),
                                      "--specific-syllable", "0", "-o", str(tmp_path / "out")],
                                catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert calls[-1]["processes"] == expected

    # a number typed on the command line is used as given
    result = CliRunner().invoke(cli, ["make-crowd-movies", str(tmp_path / "index.yaml"), str(tmp_path / "model.p"),
                                      "--specific-syllable", "0", "--processes", "2", "-o", str(tmp_path / "out")],
                                catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert calls[-1]["processes"] == 2
