"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


def test_usage_relabeling_includes_raw_states_above_99():
    from moseq2_viz.model.util import relabel_by_usage
    original = np.array([150, 150, 150, 3, 3, 150, 150, 3, 3])
    labels, ordering = relabel_by_usage([original], count="frames")
    assert ordering[:2] == [150, 3]
    np.testing.assert_array_equal(labels[0], np.where(original == 150, 0, 1))
