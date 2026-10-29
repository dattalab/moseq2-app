"""Entropy and entropy rate of syllable sequences.

``entropy_rate`` weighs each syllable's outgoing transition probabilities by
its usage, so the transition counts have to be normalised row by row first.
"""

from __future__ import annotations

import numpy as np

from moseq2_viz.info.util import entropy_rate


def test_entropy_rate_normalises_transitions_by_row_by_default():
    """Called with its defaults, it computes the rate from row probabilities."""
    labels = [np.tile([0, 0, 1, 1, 2, 2, 0, 0], 5)]
    np.testing.assert_allclose(entropy_rate(labels), entropy_rate(labels, normalize="rows"))
