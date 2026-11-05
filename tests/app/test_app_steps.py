"""The notebook-only steps: the crowd movies of the syllable labeler and the comparison tool,
and the arena tool's settings.

The syllable labeler and the crowd-movie comparison tool render their movies through
``moseq2_app.steps.render_syllable_movies``, with ``CrowdMovieOptions``' defaults and
what each tool chooses: the syllables, the grouping and the number of examples. The arena tool resolves a
recording's settings with ``ExtractOptions``, as extraction does.
"""

from __future__ import annotations

import shutil
from dataclasses import fields
from pathlib import Path

import pytest

from moseq2_common.yaml import read_yaml, write_yaml

#: How many syllables the labeler is opened with here.
LABELED = 5


def _movie(crowd_movie_dir, syllable):
    return str(Path(crowd_movie_dir) / f"syllable_sorted-id-{syllable:02d}_(usage)_original-id-{syllable:02d}.mp4")


@pytest.fixture
def project(tmp_path, analysis_project, analysis_paths):
    """A copy of the analysis project, with its progress file pointing into the copy."""
    shutil.copytree(analysis_project, tmp_path, dirs_exist_ok=True)
    progress = tmp_path / "progress.yaml"
    progress.write_text(progress.read_text().replace(str(analysis_project), str(tmp_path)))
    return read_yaml(str(progress))


def test_the_labeler_and_the_comparison_tool_render_through_the_step(project, tmp_path, monkeypatch, displayed):
    import moseq2_app.steps as app_steps
    from moseq2_app.gui.wrappers import interactive_syllable_labeler_wrapper
    from moseq2_app.viz import controller
    from moseq2_app.viz.controller import CrowdMovieComparison
    from moseq2_viz.options import CrowdMovieOptions

    calls = []
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"")

    def render(model_path, index_file, crowd_movie_dir, opts, *, syllables):
        calls.append({"opts": opts, "syllables": list(syllables)})
        if len(calls) == 1:
            return {"all": [_movie(crowd_movie_dir, s) for s in syllables]}
        return {group: [str(movie)] for group in comparison.groups}

    monkeypatch.setattr(app_steps, "render_syllable_movies", render)
    monkeypatch.setattr(controller, "get_video_info", lambda path: {"dims": (100, 100)})

    interactive_syllable_labeler_wrapper(
        project["model_path"], project["config_file"], project["index_file"],
        str(tmp_path / "crowd_movies") + "/", str(tmp_path / "syll_info.yaml"), str(tmp_path),
        max_syllables=LABELED, max_examples=20,
    )
    assert len(calls) == 1
    labeler = calls[0]
    assert isinstance(labeler["opts"], CrowdMovieOptions)
    assert labeler["syllables"] == list(range(LABELED))
    assert (labeler["opts"].max_examples, labeler["opts"].separate_by, labeler["opts"].min_dur) == (20, "default", 4)
    assert labeler["opts"].select_median_duration_instances is False

    comparison = CrowdMovieComparison(
        config_data=read_yaml(project["config_file"]), index_path=project["index_file"],
        model_path=project["model_path"],
        syll_info={i: {"label": f"s{i}", "desc": "", "crowd_movie_path": ""} for i in range(LABELED)},
        output_dir=str(tmp_path / "comparison") + "/", get_pdfs=False,
    )
    # the notebook binds the preview to the syllable dropdown
    comparison.cm_syll_select.index = 2
    comparison.crowd_movie_preview(comparison.cm_syll_select.value, "group", 7)
    assert len(calls) == 2
    compared = calls[1]
    assert isinstance(compared["opts"], CrowdMovieOptions)
    assert compared["syllables"] == [2]
    assert (compared["opts"].max_examples, compared["opts"].separate_by, compared["opts"].min_dur) == (7, "groups", 4)


@pytest.fixture
def arena_project(tmp_path, raw_sessions):
    from moseq2_extract.gui import generate_config_command

    root = tmp_path / "project"
    shutil.copytree(f"{raw_sessions}/saline_example_0", root / "saline_example_0")
    generate_config_command(str(root / "config.yaml"), camera_type="k2")
    return root


def _plain(value):
    return [_plain(v) for v in value] if isinstance(value, (list, tuple)) else value


def test_the_arena_tool_builds_extract_options(arena_project):
    """The settings the tool computes its arena and preview extraction with are the recording's
    ``ExtractOptions``, as extraction resolves them: a setting config.yaml leaves out, or holds
    as null, takes the class default."""
    import panel as pn

    from moseq2_app.roi.widget import ArenaMaskWidget
    from moseq2_extract.options import ExtractOptions
    from moseq2_extract.session_config import extract_settings, recording_options

    config_file = arena_project / "config.yaml"
    config = read_yaml(str(config_file))
    del config["extract"]["use_cc"]
    config["extract"]["centroid_hampel_span"] = None
    write_yaml(config, str(config_file))

    pn.extension()
    widget = ArenaMaskWidget(str(arena_project))
    widget.session_data.crop_size = 100
    widget.set_session_config_vars()
    folder = widget.session_data.path

    settings = widget.recording_config(folder)
    expected = recording_options(extract_settings(str(config_file)), widget.session_config[folder]).as_dict()
    declared = [f.name for f in fields(ExtractOptions)]
    assert {k: _plain(settings.get(k, "<missing>")) for k in declared} == {k: _plain(expected[k]) for k in declared}
    assert (settings["use_cc"], settings["centroid_hampel_span"]) == (True, 0)
    assert _plain(settings["crop_size"]) == [100, 100]
