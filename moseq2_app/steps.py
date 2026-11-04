"""
The notebook-only pipeline steps of moseq2-app.

``render_syllable_movies`` renders the crowd movies the syllable labeler and the
crowd-movie comparison tool show, through moseq2-viz's ``make_crowd_movies`` step.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from moseq2_viz import steps as viz_steps
from moseq2_viz.options import CrowdMovieOptions


def render_syllable_movies(model_path: str, index_file: str, crowd_movie_dir: str, opts: CrowdMovieOptions, *,
                           syllables: Sequence[int], shim_flat_config: Mapping | None = None) -> dict:
    """
    Write a crowd movie of each of ``syllables`` into ``crowd_movie_dir``.

    ``syllables`` is one syllable, or the first n syllables (``range(n)``), in usage order
    when ``opts.sort`` is set. ``shim_flat_config`` (R2, removed by PR 4.1) is the tool's
    whole settings dict, passed under the options.

    Args:
    model_path (str): path to the trained model.
    index_file (str): path to the index file.
    crowd_movie_dir (str): folder the movies are written to.
    opts (CrowdMovieOptions): the crowd-movie settings.
    syllables (Sequence[int]): the syllables to render.
    shim_flat_config (Mapping): the tool's whole settings dict.

    Returns:
    (dict): the movies written, ``{"all": [...]}`` or one list per group or session.
    """
    syllables = list(syllables)
    if len(syllables) == 1:
        opts = replace(opts, specific_syllable=syllables[0])
    elif syllables == list(range(len(syllables))):
        opts = replace(opts, max_syllable=len(syllables), specific_syllable=None)
    else:
        raise ValueError(f"Crowd movies are rendered for one syllable or for the first n syllables, "
                         f"and {syllables} is neither. Pass one syllable, or range(n).")
    return viz_steps.make_crowd_movies(index_file, model_path, crowd_movie_dir, opts,
                                       shim_flat_config=shim_flat_config).movies
