"""A crowd-movie folder holds the movies and nothing describing them.

What the movies were rendered from (the model's fingerprint, the settings and
the sessions) is in the folder's record, ``crowd_movies.yaml`` beside the
folder, which is the one description of it and what the labeler's cache reads.
"""

from __future__ import annotations


def test_a_crowd_movie_render_leaves_no_info_file(analysis_project_1_6, tmp_path):
    from moseq2_common.bookkeeping import session_index
    from moseq2_common.project import Project
    from moseq2_viz.cli import make_crowd_movies
    from moseq2_viz.helpers.wrappers import make_crowd_movies_wrapper

    project = analysis_project_1_6
    # click fills in every option's default, which is what the wrapper expects
    config = make_crowd_movies.make_context(
        "make-crowd-movies",
        [str(project / "models" / "model.p"), "--project-file", str(project / "moseq2-project.yaml"),
         "--specific-syllable", "0", "--max-examples", "3", "--processes", "1"],
    ).params
    config.pop("project_file")
    output = tmp_path / "crowd_movies"
    make_crowd_movies_wrapper(session_index(Project.at(project)), config.pop("model_path"), str(output), config)

    assert list(output.glob("*.mp4")), sorted(p.name for p in output.iterdir())
    assert not (output / "info.yaml").exists()
