"""Turning syllable counts into usage fractions.

``normalize_usages`` divides each syllable's count by the total, which the
transition graphs use to size their nodes. A dictionary with no usage at all
has no fractions to give, so every syllable stays at zero rather than the
division failing.
"""

from __future__ import annotations

import numpy as np

from moseq2_viz.model.util import normalize_usages


def test_counts_become_fractions_of_the_total():
    assert normalize_usages({0: 3, 1: 1}) == {0: 0.75, 1: 0.25}


def test_no_usage_at_all_stays_zero():
    fractions = normalize_usages({0: 0.0, 1: 0.0})
    assert list(fractions.values()) == [0.0, 0.0]
    assert np.isfinite(list(fractions.values())).all()
