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


def test_entropy_rate_is_the_usage_weighted_entropy_of_each_syllables_next_step():
    """One number per session: H = -sum_i u_i sum_j P_ij log2 P_ij.

    From 0 the mouse goes to 1 or 2 with even odds, and from either it
    returns to 0. Half of all syllables are 0, whose next step carries one
    bit, and the other half carry none, so the rate is half a bit.
    """
    branching = [np.tile([0, 1, 0, 2], 500)]
    rate = entropy_rate(branching, smoothing=0, tm_smoothing=0)
    assert np.shape(rate) == (1,)
    np.testing.assert_allclose(rate, [0.5], atol=1e-3)

    cycle = [np.tile([0, 1, 2], 500)]
    np.testing.assert_allclose(entropy_rate(cycle, smoothing=0, tm_smoothing=0), [0.0], atol=1e-12)
