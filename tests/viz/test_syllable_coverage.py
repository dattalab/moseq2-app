"""How many syllables cover a share of a model's usage.

With ``max_syllables = None`` the syllable-labelling step keeps the fewest
syllables, best used first, that cover ``explained_variance`` percent of all
syllable bouts, and the later analysis steps are cut off at that number. A
model fit with more than a hundred states can use more than a hundred.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from moseq2_viz.model.util import compute_syllable_explained_variance, get_num_syllables_for_coverage


def test_coverage_counts_every_state_the_model_used():
    """120 syllables used equally: 99% of bouts takes 119 of them.

    Each syllable is one bout, numbered as relabel_by_usage leaves them.
    """
    labels = [np.arange(120)]

    assert get_num_syllables_for_coverage(labels, n_explained=99) == 119


def test_the_coverage_figure_marks_the_crossing_at_the_count(tmp_path):
    """The curve's x axis is the number of syllables, so the line drawn at the count sits on the crossing.

    Four syllables take 50%, 30%, 15% and 5% of the bouts. Three are needed
    for 90%: two reach 80%, three reach 95%.
    """
    model = {"labels": [np.array([0, 1] * 6 + [0, 2] * 3 + [0, 3])]}

    n = compute_syllable_explained_variance(model, save_dir=str(tmp_path), n_explained=90)
    curve, line = plt.gcf().axes[0].lines
    plt.close("all")

    coverage = dict(zip(curve.get_xdata(), curve.get_ydata()))
    assert n == 3
    assert list(coverage) == [1, 2, 3, 4]
    assert line.get_xdata()[0] == n
    assert coverage[n] >= 90 > coverage[n - 1]
