"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd


def test_best_fit_does_not_overwrite_median_summary_with_mean(monkeypatch):
    import moseq2_viz.model.util as util
    reference = np.array([.1, .1, .9])
    monkeypatch.setattr(util, "load_changepoint_distribution", lambda path: reference)
    models = {
        "synthetic-a": {"changepoints": np.array([.1, .1, .1]), "loglikes": 0.,
                         "model_parameters": {"kappa": "synthetic-a"}},
        "synthetic-b": {"changepoints": np.array([.2, .3, .6]), "loglikes": 1.,
                         "model_parameters": {"kappa": "synthetic-b"}},
    }
    info, _ = util.get_best_fit("synthetic-unused", models)
    assert info["min duration median match (seconds)"] == pytest.approx(0.)
    assert info["duration distances median"]["synthetic-a"] == pytest.approx(0.)
    assert "duration distances mean" in info
