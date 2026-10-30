"""How many syllables cover a share of a model's usage.

With ``max_syllables = None`` the syllable-labelling step keeps the fewest
syllables, best used first, that cover ``explained_variance`` percent of all
syllable bouts, and the later analysis steps are cut off at that number. A
model fit with more than a hundred states can use more than a hundred.
"""

from __future__ import annotations

import numpy as np

from moseq2_viz.model.util import get_num_syllables_for_coverage


def test_coverage_counts_every_state_the_model_used():
    """120 syllables used equally: 99% of bouts takes 119 of them.

    Each syllable is one bout, numbered as relabel_by_usage leaves them.
    """
    labels = [np.arange(120)]

    assert get_num_syllables_for_coverage(labels, n_explained=99) == 119
