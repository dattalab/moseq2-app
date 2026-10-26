"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


def test_coverage_is_a_count_including_threshold_crossing():
    from moseq2_viz.model.util import get_num_syllables_for_coverage
    assert get_num_syllables_for_coverage([np.zeros(30, dtype=int)], n_explained=99) == 1
