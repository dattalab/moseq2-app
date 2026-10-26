"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest


@pytest.mark.parametrize("fraction", [.2, .8])
def test_training_split_partitions_every_input_row(fraction):
    from moseq2_model.helpers.data import get_training_data_splits
    data = OrderedDict([("synthetic", np.arange(100)[:, None])])
    train, validation = get_training_data_splits(fraction, data)
    np.testing.assert_array_equal(np.concatenate([train["synthetic"], validation["synthetic"]]), data["synthetic"])
    assert len(train["synthetic"]) == int(100 * fraction)


@pytest.mark.parametrize("fraction", [0, 1, -.1, 1.1])
def test_training_split_rejects_degenerate_fractions(fraction):
    from moseq2_model.helpers.data import get_training_data_splits
    with pytest.raises(ValueError):
        get_training_data_splits(fraction, {"synthetic": np.arange(10)[:, None]})
