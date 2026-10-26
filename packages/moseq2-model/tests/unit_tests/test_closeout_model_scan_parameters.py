"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest


def test_scan_command_forwards_existing_fit_options():
    from moseq2_model.util import get_parameter_strings
    config = dict(npcs=10, num_iter=50, index=None, separate_trans=False,
                  robust=False, e_step=False, hold_out=False, max_states=20,
                  ncpus=0, cluster_type="local", nlags=5, whiten="each",
                  alpha=100., gamma=500., var_name="synthetic_scores",
                  hold_out_seed=17, noise_level=.2, percent_split=.3)
    parameters, _ = get_parameter_strings(config)
    for option in ["--nlags 5", "--whiten each", "--alpha 100.0", "--gamma 500.0",
                   "--var-name synthetic_scores", "--hold-out-seed 17",
                   "--noise-level 0.2", "--percent-split 0.3"]:
        assert option in parameters
