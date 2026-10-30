"""Renumbering syllables by how much they are used.

Every analysis cell renumbers a model's states with ``relabel_by_usage`` so
that syllable 0 is the most used. A model fit with more than a hundred
states can use state numbers above 99.
"""

from __future__ import annotations

import numpy as np

from moseq2_viz.model.util import relabel_by_usage


def test_states_numbered_above_99_are_ranked_with_the_rest():
    """The most used state becomes syllable 0 whatever its raw number."""
    raw = np.array([150, 150, 150, 3, 3, 150, 150, 3, 3])
    labels, ordering = relabel_by_usage([raw], count="frames")

    assert ordering[:2] == [150, 3]
    np.testing.assert_array_equal(labels[0], np.where(raw == 150, 0, 1))
