"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("sigma", [0, None])
def test_changepoints_can_disable_smoothing(sigma):
    from moseq2_pca.util import get_changepoints
    _, curve = get_changepoints(np.sin(np.arange(120)[None, :] / 6), sigma=sigma)
    assert np.isfinite(curve[10:-10]).all()
