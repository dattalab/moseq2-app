"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py


def test_small_area_qc_is_scale_invariant():
    from moseq2_app.roi.validation import count_frames_with_small_areas
    values = pd.DataFrame({"area_px": [100.] * 19 + [70.]})
    assert count_frames_with_small_areas(values) == 1
    assert count_frames_with_small_areas(values * 5) == 1
