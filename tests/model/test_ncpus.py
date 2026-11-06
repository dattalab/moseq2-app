"""What ``--ncpus`` does in moseq2-model.

The jax-moseq backend fits a model with whatever CPUs the job or machine
gives the process; nothing in a fit reads a CPU count. ``--ncpus`` matters in
one place: ``kappa-scan`` on SLURM puts it in each job's ``sbatch -c`` (8 when
it is 0). Its help says so, ``learn-model`` given a non-default ``--ncpus``
says in one line that it does not change a fit, and the value reaches neither
the model nor the training-set split.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict

import numpy as np
import h5py
import pytest
from click.testing import CliRunner

import moseq2_model.cli as model_cli
import moseq2_model.steps as model_steps
from moseq2_model.helpers.data import prepare_model_metadata
from moseq2_model.train.util import train_model
from moseq2_model.util import create_command_strings

NOTE = (
    "--ncpus sets the CPUs each kappa-scan job requests on SLURM. A single fit "
    "uses the CPUs the job or machine gives it, and --ncpus does not change that."
)


@pytest.mark.parametrize("command", ["learn-model", "kappa-scan"])
def test_the_help_says_what_ncpus_sets(command):
    option = next(p for p in model_cli.cli.commands[command].params if p.name == "ncpus")
    assert "each kappa-scan job" in option.help and "SLURM" in option.help
    assert "resampling" not in option.help


@pytest.fixture
def learn(tmp_path, monkeypatch):
    """Run learn-model's command line with the fit replaced by a recorder."""
    received = []
    monkeypatch.setattr(model_steps, "learn_model_wrapper",
                        lambda i, d, config: received.append(config))
    scores = tmp_path / "pca_scores.h5"
    h5py.File(scores, "w").close()  # a placeholder: the fit is stubbed, and an empty input stops a step

    def run(*args):
        result = CliRunner().invoke(
            model_cli.cli, ["learn-model", str(scores), str(tmp_path / "model.p"), *args],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        return result.output

    return run


def test_learn_model_says_ncpus_does_not_change_a_fit(learn):
    assert learn("--ncpus", "4").count(NOTE) == 1


def test_learn_model_is_quiet_about_ncpus_by_default(learn):
    assert "--ncpus" not in learn()


def test_the_scan_requests_ncpus_from_slurm_and_does_not_pass_it_on(tmp_path):
    config = dict(npcs=10, num_iter=50, index=None, separate_trans=False, robust=False,
                  e_step=False, hold_out=False, max_states=20, ncpus=4, cluster_type="slurm",
                  memory="5GB", partition="short", wall_time="3:00:00", prefix="")
    models = str(tmp_path / "models")
    command = create_command_strings("pca_scores.h5", models, config, [100]).splitlines()[-1]
    assert command.startswith("sbatch -c 4 ")
    assert "--ncpus" not in command

    command = create_command_strings("pca_scores.h5", models, {**config, "ncpus": 0}, [100]).splitlines()[-1]
    assert command.startswith("sbatch -c 8 ")


def test_ncpus_is_not_capped_at_the_number_of_training_sessions():
    config = dict(kappa=100, hold_out=False, nfolds=2, hold_out_seed=0, ncpus=8, gamma=1, alpha=1,
                  nlags=1, robust=False, max_states=2, separate_trans=False, whiten="all",
                  noise_level=0, percent_split=0)
    rng = np.random.RandomState(0)
    data = OrderedDict(a=rng.normal(size=(40, 2)), b=rng.normal(size=(40, 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare_model_metadata(data, {"groups": dict.fromkeys(data, "g")}, config)
    assert config["ncpus"] == 8


class Recorder:
    """A model that records how it is resampled."""

    nlags = 1
    states_list = ()

    def __init__(self):
        self.calls = []

    def resample_model(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def log_likelihood(self):
        return 0.0


def test_training_does_not_pass_a_cpu_count_to_the_model():
    model = Recorder()
    train_model(model, num_iter=2, progress_kwargs=dict(disable=True))
    assert model.calls == [((), {}), ((), {})]
