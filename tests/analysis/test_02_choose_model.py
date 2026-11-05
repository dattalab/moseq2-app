"""Cell 2: choose the model to analyse, and give its analysis a folder.

    analysis = choose_model(str(project.folder))

A project can hold several models, fit at different kappas and left
together in ``models/``. Everything the analysis notebook produces belongs
to exactly one of them: the crowd movies, the syllable labels, the figures,
the dataframes. Keeping them apart is what the analysis folder is for: the
model's own folder beside it in ``models/``, named after it.

The cell analyses the project's current model, the one the extraction and
modeling notebook's best-model cell picked (or the last one fitted). It
copies nothing: the analysis reads the model where it was fit, through
``moseq2-project.yaml``, so a model fit again under the same name is what
the next analysis reads.
"""

from __future__ import annotations

from pathlib import Path

from moseq2_app.util import choose_model, current_analysis
from moseq2_common.bookkeeping import load


def test_the_current_model_is_analysed_where_it_was_fit(analysis_project, analysis):
    """1. The model the project names as current, in ``models/``, and no copy of it."""
    assert analysis.model == Path(analysis_project) / "models" / "model.p"
    assert [p.name for p in (Path(analysis_project) / "models").rglob("*.p")] == ["model.p"]


def test_the_analysis_folder_is_named_after_the_model(analysis_project, analysis):
    """2. ``models/model/`` for ``models/model.p``, made by the cell."""
    assert analysis.folder == Path(analysis_project) / "models" / "model"
    assert analysis.folder.is_dir()


def test_every_output_path_sits_inside_the_analysis_folder(analysis):
    """3. Why the folder exists.

    If any of these resolved outside the model's own folder, two models
    analysed in the same project would write over each other.
    """
    for path in (analysis.plots, analysis.crowd_movies, analysis.syll_info):
        assert analysis.folder in path.parents, f"{path} is not under {analysis.folder}"


def test_the_choice_is_recorded_in_the_bookkeeping_file(analysis):
    """4. A user who restarts the kernel keeps it: the later cells read it from ``moseq2-project.yaml``."""
    models = load(analysis.project).models
    assert models["current"] == "models/model.p"
    assert models["analysis"] == {"models/model.p": "models/model"}
    assert current_analysis(str(analysis.project.folder)) == analysis


def test_running_the_cell_again_changes_nothing(analysis):
    """5. Re-running the notebook from the top is safe."""
    bookkeeping = Path(analysis.project.bookkeeping_file)
    before = bookkeeping.read_bytes()
    assert choose_model(str(analysis.project.folder)) == analysis
    assert bookkeeping.read_bytes() == before
