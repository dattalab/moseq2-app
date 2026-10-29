"""Removing single-frame jumps from the centroid and angle tracks.

``feature_hampel_filter`` runs when ``centroid_hampel_span`` or
``angle_hampel_span`` is set above 0 (both are 0 in a generated config). A
value further than ``sig`` standard deviations from the median of its
window, with the standard deviation estimated as 1.4826 times the median
absolute deviation, is replaced by that median. The test depends only on
deviations, so where the mouse is in the arena makes no difference, and the
two filters run independently of each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from moseq2_extract.extract.proc import feature_hampel_filter

FRAMES = 101
JUMP = 50


@pytest.mark.parametrize("axis", [0, 1], ids=["x", "y"])
@pytest.mark.parametrize("offset", [0, 200], ids=["near the origin", "200 px away"])
def test_a_centroid_jump_is_removed_on_either_axis_anywhere(axis, offset):
    """One frame 40 noise SDs off is pulled back to its neighbours."""
    centroid = np.random.RandomState(7).normal(0, 0.3, (FRAMES, 2)) + offset
    centroid[JUMP, axis] += 12
    filtered = feature_hampel_filter(
        {"centroid": centroid, "orientation": np.zeros(FRAMES)},
        centroid_hampel_span=5, centroid_hampel_sig=3, angle_hampel_span=0,
    )["centroid"]
    assert abs(filtered[JUMP, axis] - offset) < 2


def test_the_filter_does_not_depend_on_position():
    """Shifting the whole track shifts the result and changes nothing else."""
    centroid = np.random.RandomState(7).normal(0, 0.3, (FRAMES, 2))
    centroid[JUMP] = 12

    def run(track):
        return feature_hampel_filter(
            {"centroid": track, "orientation": np.zeros(FRAMES)},
            centroid_hampel_span=5, centroid_hampel_sig=3, angle_hampel_span=0,
        )["centroid"]

    np.testing.assert_allclose(run(centroid.copy()), run(centroid.copy() + 200) - 200, atol=1e-12)


def test_the_angle_filter_runs_without_the_centroid_filter():
    """An angle jump is removed with the centroid filter left off."""
    angles = np.random.RandomState(3).normal(0.1, 0.01, FRAMES)
    angles[JUMP] = 2
    filtered = feature_hampel_filter(
        {"centroid": np.zeros((FRAMES, 2)), "orientation": angles},
        centroid_hampel_span=0, angle_hampel_span=5, angle_hampel_sig=3,
    )["orientation"]
    assert filtered[JUMP] < 1
