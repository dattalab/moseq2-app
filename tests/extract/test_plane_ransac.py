"""Finding the arena floor when no plane can be fit.

``plane_ransac`` fits the floor from the background pixels inside
``bg_roi_depth_range``, keeping only flat pixels when the gradient filter is
on. When too few pixels survive the filter, or no candidate plane covers
enough of them, the error names the config keys that decide the outcome.
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
