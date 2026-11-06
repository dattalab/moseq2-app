"""An option typed on ``train-pca``'s command line outranks the config file,
including when the typed value equals the option's default (moseq2-extract#24
names the extract side). An option not typed takes the config file's value."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import moseq2_pca.cli as pca_cli
import moseq2_pca.steps as pca_steps
from configs import write_bookkeeping, write_session_labels
from moseq2_common.yaml import write_yaml


@pytest.fixture
def received(monkeypatch, tmp_path):
    """What ``train-pca`` hands its wrapper, instead of training, in a project (``tmp_path``) of one session."""
    write_bookkeeping(tmp_path, ["aggregate_results/session_0"])
    write_session_labels(tmp_path, ["aggregate_results/session_0"])
    calls = []

    def record(input_dir, config, output_dir, output_file):
        calls.append(dict(config))
        return config

    monkeypatch.setattr(pca_steps, "train_pca_wrapper", record)
    return calls


@pytest.mark.parametrize(
    "typed, expected",
    [(("--rank", "25"), 25), (("--rank", "30"), 30), ((), 50)],
    ids=["typed-default", "typed-other", "not-typed"],
)
def test_train_pca_rank(typed, expected, tmp_path, received):
    config = tmp_path / "config.yaml"
    write_yaml({"pca": {"rank": 50}}, str(config))

    result = CliRunner().invoke(
        pca_cli.cli,
        ["train-pca", "-i", str(tmp_path), "-o", str(tmp_path / "_pca"),
         "--config-file", str(config), *typed],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    assert received[0]["rank"] == expected


# A config value that is falsy (0, 0.0, False) still replaces the option's
# default. ``min_height: 0`` keeps every pixel above the floor where the default
# zeroes those below 10 mm, and ``sigma: 0`` turns off the changepoint
# smoothing, which ``get_changepoints`` handles as its own case. A typed option
# still wins over either.


@pytest.mark.parametrize(
    "typed, expected",
    [((), 0), (("--min-height", "15"), 15), (("--min-height", "10"), 10)],
    ids=["not-typed", "typed-other", "typed-default"],
)
def test_train_pca_zero_min_height_from_config(typed, expected, tmp_path, received):
    config = tmp_path / "config.yaml"
    write_yaml({"pca": {"min_height": 0}}, str(config))

    result = CliRunner().invoke(
        pca_cli.cli,
        ["train-pca", "-i", str(tmp_path), "-o", str(tmp_path / "_pca"),
         "--config-file", str(config), *typed],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    assert received[0]["min_height"] == expected


@pytest.fixture
def changepoints_received(monkeypatch, tmp_path):
    """What ``compute-changepoints`` hands its wrapper, instead of computing, in a project (``tmp_path``) of one
    session."""
    write_bookkeeping(tmp_path, ["aggregate_results/session_0"])
    write_session_labels(tmp_path, ["aggregate_results/session_0"])
    calls = []

    def record(input_dir, config, output_dir, output_file):
        calls.append(dict(config))
        return config

    monkeypatch.setattr(pca_steps, "compute_changepoints_wrapper", record)
    return calls


@pytest.mark.parametrize(
    "config_values, typed, expected",
    [
        ({"sigma": 0, "threshold": 0.0}, (), {"sigma": 0, "threshold": 0.0}),
        ({"sigma": 0}, ("--sigma", "2"), {"sigma": 2, "threshold": 0.5}),
        ({"sigma": None}, (), {"sigma": 3.5, "threshold": 0.5}),
    ],
    ids=["falsy-config", "typed-over-falsy", "null-config"],
)
def test_compute_changepoints_falsy_config(
    config_values, typed, expected, tmp_path, changepoints_received
):
    config = tmp_path / "config.yaml"
    write_yaml({"pca": config_values}, str(config))

    result = CliRunner().invoke(
        pca_cli.cli,
        ["compute-changepoints", "-i", str(tmp_path), "-o", str(tmp_path / "_pca"),
         "--config-file", str(config), *typed],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    got = changepoints_received[0]
    assert {k: got[k] for k in expected} == expected
