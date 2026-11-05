"""
The notebook-only pipeline steps of moseq2-app.

``render_syllable_movies`` renders the crowd movies the syllable labeler and the
crowd-movie comparison tool show, through moseq2-viz's ``make_crowd_movies`` step.
``train_flip_classifier`` trains a flip classifier on the frames labeled in the
Flip-Classifier-Training notebook, cleaned as extraction cleans them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from moseq2_app.flip.train import create_training_dataset, save_classifier, train_classifier
from moseq2_common.frames import CleaningOptions
from moseq2_viz import steps as viz_steps
from moseq2_viz.options import CrowdMovieOptions


def render_syllable_movies(model_path: str, index_file: str, crowd_movie_dir: str, opts: CrowdMovieOptions, *,
                           syllables: Sequence[int]) -> dict:
    """
    Write a crowd movie of each of ``syllables`` into ``crowd_movie_dir``.

    ``syllables`` is one syllable, or the first n syllables (``range(n)``), in usage order
    when ``opts.sort`` is set.

    Args:
    model_path (str): path to the trained model.
    index_file (str): path to the index file.
    crowd_movie_dir (str): folder the movies are written to.
    opts (CrowdMovieOptions): the crowd-movie settings.
    syllables (Sequence[int]): the syllables to render.

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
    return viz_steps.make_crowd_movies(index_file, model_path, crowd_movie_dir, opts).movies


@dataclass(frozen=True)
class FlipTrainingResult:
    """What ``train_flip_classifier`` wrote."""

    training_data: Path
    classifier: Path


def train_flip_classifier(frame_ranges_file: str, out_path: str, cleaning: CleaningOptions, *,
                          classifier: str = "RF") -> FlipTrainingResult:
    """
    Train a flip classifier on the labeled frames in ``frame_ranges_file`` and save it to ``out_path``.

    The training set, written beside ``frame_ranges_file`` as ``training_data.npz``,
    includes a copy of the labeled frames cleaned as extraction cleans them with
    ``cleaning``, so pass the cleaning settings the classifier's recordings are
    extracted with.

    Args:
    frame_ranges_file (str): the saved frame ranges (``flip-training-frame-ranges.p``).
    out_path (str): where the classifier is saved.
    cleaning (CleaningOptions): extraction's cleaning settings.
    classifier (str): ``RF`` for a random forest, ``SVM`` for a support vector machine.

    Returns:
    (FlipTrainingResult): the training set and the classifier written.
    """
    training_data = create_training_dataset(frame_ranges_file, cleaning)
    save_classifier(train_classifier(training_data, classifier=classifier), out_path)
    return FlipTrainingResult(training_data=Path(training_data), classifier=Path(out_path))
