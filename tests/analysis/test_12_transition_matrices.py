"""Cell 12: how one syllable leads to the next, per group.

    normalize = 'bigram'
    model_data = parse_model_results(progress_paths['model_path'])
    model_data['labels'] = relabel_by_usage(model_data['labels'], count='usage')[0]
    label_group, uuids = get_trans_graph_groups(model_data)
    group = list(set(label_group))
    trans_mats, usages = get_group_trans_mats(model_data['labels'], label_group, group,
                                              max_syllables, normalize=normalize)
    fig, ax = plt.subplots(1, len(group), figsize=(12, 8), sharex=False, sharey=True)
    ...
    fig.savefig(join(progress_paths['plot_path'], f'{normalize}_tansition_matrix.{ext}'))

Usage says how often a syllable happens; this says what it follows and
what follows it. Two groups can use the same syllables equally often and
string them together quite differently, and that is a result usage alone
cannot show.

One matrix per group, syllable by syllable, drawn side by side on a
shared colour scale so the two are comparable by eye. ``normalize``
chooses what a cell means: ``bigram`` makes the whole matrix a
distribution over transitions, ``rows`` makes each row the distribution
of what follows that syllable, ``columns`` the distribution of what
precedes it.

The cell is written out longhand in the notebook rather than called, so
the tests transcribe it. That is also why its two defects are the
notebook's rather than the library's.
"""

from __future__ import annotations

import os
from os.path import join
from pathlib import Path

import numpy as np
import pytest

from moseq2_viz.model.trans_graph import (
    get_group_trans_mats,
    get_trans_graph_groups,
)
from moseq2_viz.model.util import parse_model_results, relabel_by_usage

#: The normalization the cell ships with.
NORMALIZE = "bigram"

#: What the cell names the figure. The spelling is the notebook's.
FIGURE = f"{NORMALIZE}_tansition_matrix"


def draw_matrices(trans_mats, group, max_syllables, normalize, plot_path):
    """The cell's plotting half, transcribed.

    Kept in one place so the test for a single-group experiment runs the
    same code as the test for the figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, len(group), figsize=(12, 8), sharex=False, sharey=True)
    title_map = dict(bigram="Bigram", columns="Incoming", rows="Outgoing")

    color_lim = max([x.max() for x in trans_mats])

    for i, g in enumerate(group):
        h = ax[i].imshow(
            trans_mats[i][:max_syllables, :max_syllables], cmap="cubehelix", vmax=color_lim
        )
        if i == 0:
            ax[i].set_ylabel("Incoming syllable")
        cb = fig.colorbar(h, ax=ax[i], fraction=0.046, pad=0.04)
        cb.set_label(f"{title_map[normalize]} transition probability")
        ax[i].set_xlabel("Outgoing syllable")
        ax[i].set_title(g)

    os.makedirs(plot_path, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(join(plot_path, f"{normalize}_tansition_matrix.{ext}"))
    return fig


@pytest.fixture(scope="module")
def figure(analysis_paths, labeled, transitions):
    """The cell's plotting half, run for real."""
    import matplotlib.pyplot as plt

    plt.close("all")
    drawn = draw_matrices(
        transitions["trans_mats"],
        transitions["group"],
        labeled,
        NORMALIZE,
        analysis_paths["plot_path"],
    )
    yield drawn, Path(analysis_paths["plot_path"])
    plt.close("all")


@pytest.fixture
def image(figure):
    from PIL import Image as PILImage

    _, plots = figure
    return PILImage.open(plots / f"{FIGURE}.png").convert("L")


def test_every_modelled_session_is_assigned_its_group(transitions, sorted_index):
    """1. Where the per-group split comes from.

    The grouping is read out of the model rather than the index, because
    the model holds only the sessions it was fit on. If the two
    disagreed, the matrices would be averaging the wrong animals
    together.
    """
    expected = {uuid: entry["group"] for uuid, entry in sorted_index["files"].items()}
    assigned = dict(zip(transitions["uuids"], transitions["label_group"]))

    assert set(assigned) <= set(expected)
    for uuid, group in assigned.items():
        assert group == expected[uuid], f"{uuid} is {group} in the model, {expected[uuid]} in the index"

    assert len(transitions["group"]) > 1, "one group; there is nothing to compare"


def test_there_is_one_square_matrix_per_group(transitions, labeled):
    """2. Syllable by syllable, for each condition.

    The matrix has to be square and cover every syllable being analysed:
    the cell slices it to ``max_syllables`` and indexes both axes by
    syllable number.
    """
    assert len(transitions["trans_mats"]) == len(transitions["group"])
    for name, matrix in zip(transitions["group"], transitions["trans_mats"]):
        assert matrix.shape == (labeled, labeled), f"{name} is {matrix.shape}"
        assert np.isfinite(matrix).all(), f"{name} holds non-finite entries"
        assert (matrix >= 0).all(), f"{name} holds a negative probability"


def test_bigram_normalisation_makes_the_whole_matrix_a_distribution(transitions):
    """3. What a cell of the figure means under the default.

    With ``bigram`` every entry is the share of all transitions that went
    from one syllable to another, so the matrix sums to one and two
    groups are directly comparable on the shared colour scale the cell
    sets.
    """
    for name, matrix in zip(transitions["group"], transitions["trans_mats"]):
        assert matrix.sum() == pytest.approx(1.0), name


def test_row_normalisation_asks_a_different_question(analysis_paths, labeled, transitions):
    """4. ``normalize`` is a real choice.

    Row-normalized, each row is the distribution over what follows that
    syllable, so every row sums to one regardless of how often the
    syllable occurred. That answers "what does the mouse do next" rather
    than "what does it do most".
    """
    model_data = parse_model_results(analysis_paths["model_path"])
    model_data["labels"] = relabel_by_usage(model_data["labels"], count="usage")[0]
    label_group, _ = get_trans_graph_groups(model_data)

    by_row, _ = get_group_trans_mats(
        model_data["labels"], label_group, transitions["group"], labeled, normalize="rows"
    )

    for name, matrix in zip(transitions["group"], by_row):
        totals = matrix.sum(axis=1)
        used = totals > 0
        np.testing.assert_allclose(totals[used], 1.0, atol=1e-9, err_msg=name)


def test_each_group_brings_the_usage_counts_the_graph_scales_nodes_by(transitions, labeled):
    """5. Cell 13 exports these and cell 14 draws nodes sized by them."""
    assert len(transitions["usages"]) == len(transitions["group"])
    for name, usage in zip(transitions["group"], transitions["usages"]):
        assert set(usage) >= set(range(labeled)), f"{name} is missing syllables"
        assert all(count >= 0 for count in usage.values()), name


def test_the_figure_shows_one_panel_per_group(figure, transitions):
    """6. Side by side, titled, on one colour scale.

    The panels are compared by eye, which only works if they share the
    scale the cell computes across both.
    """
    drawn, _ = figure
    panels = [axis for axis in drawn.axes if axis.get_title()]
    assert [axis.get_title() for axis in panels] == list(transitions["group"])

    limits = {axis.images[0].get_clim() for axis in panels if axis.images}
    assert len(limits) == 1, f"the panels are on different colour scales: {limits}"


def test_the_figure_is_written_large_enough_and_is_not_blank(figure, image):
    """7. The cell's only deliverable."""
    _, plots = figure
    for extension in ("png", "pdf"):
        path = plots / f"{FIGURE}.{extension}"
        assert path.exists(), f"{path.name} was not written"
        assert path.stat().st_size > 0, f"{path.name} is empty"

    width, height = image.size
    assert width >= 400 and height >= 300, f"the figure is {width}x{height}"

    pixels = np.asarray(image)
    assert (pixels < 250).mean() > 0.01, "the figure is blank"
    assert pixels.std() > 5, "the figure has no contrast"


@pytest.mark.xfail(
    reason="the cell writes `{normalize}_tansition_matrix`, so the figure a user "
    "goes looking for under `transition` is not there"
)
def test_the_figure_is_named_after_what_it_shows(figure):
    """8. A user finds this file by its name.

    Everything else the notebook writes is named for what it holds. This
    one is misspelled, so it does not turn up beside the others and a
    search for it finds nothing.
    """
    _, plots = figure
    assert (plots / f"{NORMALIZE}_transition_matrix.png").exists()


@pytest.mark.xfail(
    reason="plt.subplots(1, 1) returns a bare Axes rather than an array of one, "
    "and the cell indexes ax[i], so a pilot experiment with a single condition "
    "dies with TypeError: 'Axes' object is not subscriptable"
)
def test_an_experiment_with_one_condition_still_plots(
    analysis_paths, labeled, transitions, tmp_path
):
    """9. The first thing anyone runs is a pilot.

    A project with one group is the normal starting point: a handful of
    animals, no comparison yet, just a look at what the model found.
    Every cell before this one handles it. This one builds its axes from
    the number of groups and then assumes there is more than one.
    """
    one_group = transitions["group"][:1]
    matrices = transitions["trans_mats"][:1]

    draw_matrices(matrices, one_group, labeled, NORMALIZE, str(tmp_path))
