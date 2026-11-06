"""
The notebook-only pipeline steps of moseq2-app.

``render_syllable_movies`` renders the crowd movies the syllable labeler and the
crowd-movie comparison tool show, through moseq2-viz's ``make_crowd_movies`` step.
``train_flip_classifier`` trains a flip classifier on the frames labeled in the
Flip-Classifier-Training notebook, each session's cleaned as extraction cleans that
recording. The classifier's record (``moseq2_common.records``) is written beside it as
``<classifier>.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from moseq2_app.flip.train import (create_training_dataset, save_classifier, session_cleaning, train_classifier,
                                   training_settings)
from moseq2_common.records import check_inputs, start_record, write_record
from moseq2_viz import steps as viz_steps
from moseq2_viz.options import CrowdMovieOptions


def syllable_movie_options(opts: CrowdMovieOptions, syllables: Sequence[int]) -> CrowdMovieOptions:
    """``opts`` set to render ``syllables``: one syllable, or the first n syllables (``range(n)``)."""
    syllables = list(syllables)
    if len(syllables) == 1:
        return replace(opts, specific_syllable=syllables[0])
    if syllables == list(range(len(syllables))):
        return replace(opts, max_syllable=len(syllables), specific_syllable=None)
    raise ValueError(f"Crowd movies are rendered for one syllable or for the first n syllables, "
                     f"and {syllables} is neither. Pass one syllable, or range(n).")


def render_syllable_movies(model_path: str, index: dict, crowd_movie_dir: str, opts: CrowdMovieOptions, *,
                           syllables: Sequence[int]) -> dict:
    """
    Write a crowd movie of each of ``syllables`` into ``crowd_movie_dir``.

    ``syllables`` is one syllable, or the first n syllables (``range(n)``), in usage order
    when ``opts.sort`` is set.

    Args:
    model_path (str): path to the trained model.
    index (dict): the project's sessions (``moseq2_common.bookkeeping.session_index``).
    crowd_movie_dir (str): folder the movies are written to.
    opts (CrowdMovieOptions): the crowd-movie settings.
    syllables (Sequence[int]): the syllables to render.

    Returns:
    (dict): the movies written, ``{"all": [...]}`` or one list per group or session.
    """
    return viz_steps.make_crowd_movies(index, model_path, crowd_movie_dir,
                                       syllable_movie_options(opts, syllables)).movies


@dataclass(frozen=True)
class FlipTrainingResult:
    """What ``train_flip_classifier`` wrote."""

    training_data: Path
    classifier: Path


def train_flip_classifier(frame_ranges_file: str, out_path: str, config_file: str | None = None, *,
                          classifier: str = "RF") -> FlipTrainingResult:
    """
    Train a flip classifier on the labeled frames in ``frame_ranges_file`` and save it to ``out_path``.

    The training set, written beside ``frame_ranges_file`` as ``training_data.npz``,
    includes a copy of the labeled frames cleaned as extraction cleans them. Each
    session's frames are cleaned with the settings its recording is extracted with:
    ``config_file``'s ``extract`` section, with the recording's entry in the project's
    session_config.yaml over it (each labeled file's recording is the one the project's
    moseq2-project.yaml lists for it) (``moseq2_app.flip.train.session_cleaning``), or
    extraction's defaults without ``config_file``. A session whose extraction ran with
    other cleaning settings is named, with both values, and trained with the settings
    it has now. The classifier carries the crop size of the labeled frames and the
    cleaning settings it was trained with, which extraction checks before it uses it
    (``moseq2_extract.helpers.wrappers.validate_flip_classifier``).

    Args:
    frame_ranges_file (str): the saved frame ranges (``flip-training-frame-ranges.p``).
    out_path (str): where the classifier is saved.
    config_file (str): the project's config.yaml; None for extraction's defaults.
    classifier (str): ``RF`` for a random forest, ``SVM`` for a support vector machine.

    Returns:
    (FlipTrainingResult): the training set and the classifier written.
    """
    check_inputs("train-flip-classifier", {frame_ranges_file: (
        "Label frames in the Flip-Classifier-Training notebook's labeling widget and save them again.")})
    record = start_record("train-flip-classifier", {"classifier": classifier},
                          [path for path in (config_file, frame_ranges_file) if path])
    cleaning = session_cleaning(frame_ranges_file, config_file)
    record.note("cleaning", {name: opts.as_dict() for name, opts in cleaning.items()})
    training_data = create_training_dataset(frame_ranges_file, cleaning)
    save_classifier(train_classifier(training_data, classifier=classifier), out_path,
                    training_settings(frame_ranges_file, cleaning))
    result = FlipTrainingResult(training_data=Path(training_data), classifier=Path(out_path))
    if result.classifier.is_file():
        write_record(result.classifier,
                     record.finish({"files": [str(result.classifier), str(result.training_data)]}))
    return result
