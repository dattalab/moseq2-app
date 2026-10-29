"""Fitting the arena floor with RANSAC.

``plane_ransac`` fits the floor from the background pixels inside
``bg_roi_depth_range``, keeping only flat pixels when the gradient filter is
on. Its candidate planes come from its own seeded generator, so the same
background gives the same floor every time and the rest of the program's
random draws are left alone. When too few pixels survive the filter, or no
candidate plane covers enough of them, the error names the config keys that
decide the outcome.
"""

from __future__ import annotations

import numpy as np
import pytest

from moseq2_extract.extract.roi import plane_ransac


def test_a_filter_that_leaves_no_pixels_names_its_settings():
    """The gradient filter keeping no pixels points at its threshold."""
    y, x = np.mgrid[:30, :40]
    floor = 700 + 0.3 * x - 0.2 * y
    with pytest.raises(ValueError, match='"bg_roi_gradient_threshold"'):
        plane_ransac(floor, mask=np.zeros(floor.shape, dtype=bool))


def test_a_floor_no_plane_fits_names_the_depth_range():
    """A floor too noisy for the tolerance points at the depth range."""
    np.random.seed(0)
    floor = np.random.uniform(655, 745, (30, 40))
    with pytest.raises(ValueError, match='"bg_roi_depth_range" \\(650, 750\\)'):
        plane_ransac(floor, noise_tolerance=1)


def floor():
    y, x = np.mgrid[:30, :40]
    return 700 + 0.3 * x - 0.2 * y + np.random.RandomState(3).normal(0, 2, (30, 40))


def test_the_same_background_gives_the_same_floor():
    """Two fits of one image agree exactly, whatever was drawn in between."""
    first = plane_ransac(floor(), iters=30)
    np.random.seed(1)
    np.random.random(100)
    second = plane_ransac(floor(), iters=30)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_fitting_leaves_the_global_random_state_alone():
    """Code that draws random numbers after the fit draws what it would have."""
    np.random.seed(84)
    before = np.random.get_state()
    plane_ransac(floor(), iters=30)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
