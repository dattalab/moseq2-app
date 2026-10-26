"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py


def test_stationary_count_cannot_be_negative():
    from moseq2_app.roi.validation import count_stationary_frames
    assert count_stationary_frames(pd.DataFrame({"velocity_2d_mm": [1., 2., 3.]})) == 0
