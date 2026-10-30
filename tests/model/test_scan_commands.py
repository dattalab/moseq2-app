"""The learn-model commands a kappa scan writes.

``moseq2-model kappa-scan`` takes the same fitting options as
``learn-model`` and writes one ``learn-model`` command per kappa, built by
``get_parameter_strings``. Whatever the scan was given has to reach each of
those commands, or every model in the scan is fit with the defaults instead.
"""

from __future__ import annotations

import shlex

from moseq2_model.cli import cli
from moseq2_model.util import get_parameter_strings

#: Fitting options the scan accepts, each set away from its default.
FITTING = dict(
    nlags=5,
    whiten="each",
    alpha=100.0,
    gamma=500.0,
    var_name="synthetic_scores",
    hold_out_seed=17,
    noise_level=0.2,
    percent_split=30,
)


def test_the_scans_fitting_options_reach_each_learn_model_command(tmp_path):
    """learn-model, parsing the command the scan wrote, sees the scan's values."""
    config = dict(
        npcs=10, num_iter=50, index=None, separate_trans=False, robust=False,
        e_step=False, hold_out=False, max_states=20, ncpus=0, cluster_type="local",
        **FITTING,
    )
    parameters, _ = get_parameter_strings(config)
    scores = tmp_path / "pca_scores.h5"
    scores.touch()

    learn = cli.commands["learn-model"]
    ctx = learn.make_context(
        "learn-model", [str(scores), str(tmp_path / "model.p"), *shlex.split(parameters)]
    )

    assert {key: ctx.params[key] for key in FITTING} == FITTING
