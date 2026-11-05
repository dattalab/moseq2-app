"""What ``learn-model`` does with a ``kappa`` it cannot fit at.

``learn-model`` fits one model at one kappa: a number, or the number of
frames when ``kappa`` is null. The notebook's model cell writes
``kappa: scan`` into ``config.yaml`` when a scan is set up, and handing that
file to ``learn-model --config-file`` used to reach JAX as a string, which
stopped deep in the fit naming neither the file nor the setting. It stops
before anything is loaded, naming the file, the value and what to do.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import moseq2_model.cli as model_cli
import moseq2_model.steps as model_steps
from moseq2_common.yaml import write_yaml


@pytest.fixture
def fits(monkeypatch):
    """The settings each fit is handed, instead of fitting."""
    received = []
    monkeypatch.setattr(model_steps, "learn_model_wrapper", lambda i, d, config: received.append(config))
    return received


def learn_model(tmp_path, settings):
    scores = tmp_path / "pca_scores.h5"
    scores.touch()
    config = tmp_path / "config.yaml"
    write_yaml(settings, str(config))
    result = CliRunner().invoke(
        model_cli.cli,
        ["learn-model", str(scores), str(tmp_path / "model.p"), "--config-file", str(config)],
    )
    return config, result


def test_a_kappa_that_is_not_a_number_stops_learn_model_naming_the_file(tmp_path, fits):
    config, result = learn_model(tmp_path, {"kappa": "scan"})
    assert result.exit_code != 0
    assert "'scan'" in result.output and str(config) in result.output
    assert "moseq2-model kappa-scan" in result.output
    assert "JAX" not in result.output
    assert not fits


@pytest.mark.parametrize("kappa", [1000, 1e6, None])
def test_a_number_or_null_kappa_is_fitted(tmp_path, fits, kappa):
    _, result = learn_model(tmp_path, {"kappa": kappa})
    assert result.exit_code == 0, result.output
    assert fits[0]["kappa"] == kappa


def test_ncpus_from_the_config_file_prints_no_note(tmp_path, fits):
    """Each kappa-scan job's settings file holds the scan's ``ncpus``; nobody
    typed ``--ncpus`` to that job, so it says nothing about the flag."""
    _, result = learn_model(tmp_path, {"kappa": 1000, "ncpus": 4})
    assert result.exit_code == 0, result.output
    assert "--ncpus" not in result.output
