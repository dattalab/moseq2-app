"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd
import h5py


def test_integer_timestamps_can_be_rescaled():
    from moseq2_app.roi.validation import check_timestamp_error_percentage
    integers = np.arange(10) * 33
    assert check_timestamp_error_percentage(integers) == check_timestamp_error_percentage(integers.astype(float))
