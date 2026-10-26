"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pytest


def _floor():
    y, x = np.mgrid[:30, :40]
    return 700 + .3 * x - .2 * y + np.random.RandomState(3).normal(0, 2, (30, 40))


def test_ransac_empty_mask_reports_input_problem():
    from moseq2_extract.extract.roi import plane_ransac
    with pytest.raises(ValueError, match="(?i)(mask|points|inlier)"):
        plane_ransac(_floor(), mask=np.zeros((30, 40), dtype=bool), iters=2)


def test_ransac_no_accepted_plane_reports_failure():
    from moseq2_extract.extract.roi import plane_ransac
    with pytest.raises(ValueError, match="(?i)(plane|iteration|ransac)"):
        plane_ransac(_floor(), iters=0)
