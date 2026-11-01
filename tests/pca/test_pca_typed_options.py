"""An option typed on ``train-pca``'s command line outranks the config file,
including when the typed value equals the option's default (moseq2-extract#24
names the extract side; ``moseq2_pca.util.command_with_config`` is the same
merge). An option not typed takes the config file's value."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import moseq2_pca.cli as pca_cli
from moseq2_pca._yaml import write_yaml


@pytest.fixture
def received(monkeypatch):
    """What ``train-pca`` hands its wrapper, instead of training."""
    calls = []

    def record(input_dir, config, output_dir, output_file):
        calls.append(dict(config))
        return config

    monkeypatch.setattr(pca_cli, "train_pca_wrapper", record)
    monkeypatch.setattr(pca_cli, "combine_new_config", lambda *a, **k: None)
    return calls


@pytest.mark.parametrize(
    "typed, expected",
    [(("--rank", "25"), 25), (("--rank", "30"), 30), ((), 50)],
    ids=["typed-default", "typed-other", "not-typed"],
)
def test_train_pca_rank(typed, expected, tmp_path, received):
    config = tmp_path / "config.yaml"
    write_yaml({"rank": 50}, str(config))

    result = CliRunner().invoke(
        pca_cli.cli,
        ["train-pca", "-i", str(tmp_path), "-o", str(tmp_path / "_pca"),
         "--config-file", str(config), *typed],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    assert received[0]["rank"] == expected
