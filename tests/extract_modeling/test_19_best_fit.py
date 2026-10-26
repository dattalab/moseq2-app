"""Cell 19: choose between the models that were fit.

    output_file = join(progress_paths['base_dir'], 'model_vs_pc_changepoints')
    objective = 'duration (mean match)'

    best_model_fit = get_best_fit_model(progress_paths, output_file, plot_all=True, objective=objective)

    progress_paths = update_progress(progress_filepath, 'model_path',
                                     best_model_fit[f'best model - {objective}'])

The last cell, and the reason cell 16 exists. Kappa decides how long the
model expects a syllable to last, and there is no way to know the right value
in advance, so a user fits several and compares. This cell does the
comparing: it reads every model in the folder, measures the syllable
durations each produced, and scores them against the model-free changepoints.

The chosen model's path is written back into the progress file, which is what
the analysis notebook opens. That makes this cell the handover between the
two notebooks, and the only one whose output is a decision rather than a file.

With a single model in the folder the comparison is vacuous, so the chain
fits a second at a kappa far from the first. That is what a user comparing
kappas would have in front of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moseq2_app.gui.progress import restore_progress_vars
from moseq2_viz.gui import get_best_fit_model

from tests.conftest import SECOND_KAPPA

#: The objectives the notebook names. The first is the default; the comment
#: in the cell says to switch to the third when choosing a final model.
OBJECTIVES = ("duration (mean match)", "jsd", "median_loglikelihood")

DEFAULT_OBJECTIVE = OBJECTIVES[0]

#: Written beside the models, one row per model compared.
STATS = "model_kappa_scan_stats.csv"


@pytest.fixture(scope="session")
def comparison(two_models, model_settings, project):
    """Run the cell once over both models."""
    output_file = str(project / "model_vs_pc_changepoints")
    chosen = get_best_fit_model(
        model_settings, output_file, plot_all=True, objective=DEFAULT_OBJECTIVE
    )
    return chosen, project


def test_the_cell_chooses_one_of_the_models_that_were_fit(comparison, two_models):
    """1. The cell's decision.

    The choice has to be a model that exists, which sounds trivial until the
    folder holds a checkpoint or a half-written file and the comparison picks
    it.
    """
    chosen, _ = comparison
    best = chosen[f"best model - {DEFAULT_OBJECTIVE}"]

    assert best in {str(path) for path in two_models.values()}


def test_both_models_were_compared(comparison, two_models, project):
    """2. Nothing in the folder is skipped.

    A model silently left out of the comparison would be invisible: the cell
    would still name a winner and a user would never know it had a choice.
    """
    _, project = comparison
    stats = pd.read_csv(project / "models" / STATS)

    assert len(stats) == len(two_models)
    assert sorted(stats["file_name"]) == sorted(str(p) for p in two_models.values())
    assert sorted(stats["model_kappa"]) == sorted(
        [float(SECOND_KAPPA), float(2400)]
    )


def test_each_model_is_measured_against_the_changepoints(comparison, project):
    """3. What the comparison is made of.

    Each model contributes the mean, median and mode of its syllable
    durations, set beside the same three taken from the model-free
    changepoints. The second three are a property of the data, so they are
    the same for every row.
    """
    _, project = comparison
    stats = pd.read_csv(project / "models" / STATS)

    for column in ("model_mean", "model_median", "model_mode", "pc_mean", "pc_median", "pc_mode"):
        assert column in stats.columns
        assert np.isfinite(stats[column]).all(), column
        assert (stats[column] > 0).all(), column

    for column in ("pc_mean", "pc_median", "pc_mode"):
        assert stats[column].nunique() == 1, f"{column} differs between models"


def test_the_chosen_model_is_the_closest_on_the_objective(comparison, project):
    """4. The choice is the one the objective implies.

    'duration (mean match)' means the model whose mean syllable duration is
    nearest the changepoints'. Recomputing that from the stats file is how we
    know the cell chose rather than took the first row.
    """
    chosen, project = comparison
    stats = pd.read_csv(project / "models" / STATS)

    distance = (stats["model_mean"] - stats["pc_mean"]).abs()
    expected = stats.loc[distance.idxmin(), "file_name"]

    assert chosen[f"best model - {DEFAULT_OBJECTIVE}"] == expected


def test_every_objective_names_a_model(two_models, model_settings, project):
    """5. The cell's comment tells a user to switch objectives.

    It says to use median_loglikelihood when picking a final model, so all
    three have to work, not just the default.
    """
    for objective in OBJECTIVES:
        chosen = get_best_fit_model(
            model_settings,
            str(project / "model_vs_pc_changepoints"),
            plot_all=True,
            objective=objective,
        )
        best = chosen[f"best model - {objective}"]
        assert best in {str(path) for path in two_models.values()}, objective


def test_the_comparison_figure_is_drawn(comparison, project):
    """6. What a user looks at before accepting the answer.

    The cell plots every model's duration distribution against the
    changepoints', which is how an answer that is technically closest but
    obviously wrong gets caught.
    """
    from PIL import Image

    _, project = comparison
    for suffix in ("png", "pdf"):
        assert (project / f"model_vs_pc_changepoints.{suffix}").exists()

    figure = Image.open(project / "model_vs_pc_changepoints.png").convert("L")
    pixels = np.asarray(figure)
    assert figure.size[0] >= 400 and figure.size[1] >= 300
    assert (pixels < 250).mean() > 0.01, "the comparison plot is blank"


def test_the_choice_is_handed_to_the_analysis_notebook(comparison, progress_file):
    """7. The handover between the two notebooks.

    The cell writes the chosen model back into the progress file, which is
    the only thing the analysis notebook is given. A choice that was made and
    not recorded would leave that notebook opening whichever model happened
    to be named last.
    """
    chosen, _ = comparison
    best = chosen[f"best model - {DEFAULT_OBJECTIVE}"]

    from moseq2_app.gui.progress import update_progress

    update_progress(str(progress_file), "model_path", best)
    recorded = restore_progress_vars(str(progress_file), init=False, overwrite=False)
    assert recorded["model_path"] == best
