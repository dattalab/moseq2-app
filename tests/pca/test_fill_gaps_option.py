"""``apply-pca`` always fills dropped frames when a recording has timestamps.

Every later step treats a row of PC scores as one frame step, so scores
without the empty rows would make the model and the changepoints assume
nothing was dropped. There is no option to turn it off; typing the former
``--fill-gaps`` stops with that explanation. A config file that still holds
``fill_gaps`` (``apply-pca`` used to write every option it ran with back into
it) is accepted, since it says nothing the command does not already do.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import moseq2_pca.cli as pca_cli
from moseq2_common.yaml import write_yaml


@pytest.fixture
def received(monkeypatch):
    """What ``apply-pca`` hands its wrapper, instead of computing scores."""
    calls = []

    def record(input_dir, config, output_dir, output_file):
        calls.append(dict(config))
        return config, True

    monkeypatch.setattr(pca_cli, "apply_pca_wrapper", record)
    monkeypatch.setattr(pca_cli, "combine_new_config", lambda *a, **k: None)
    return calls


def apply_pca(tmp_path, *args):
    return CliRunner().invoke(
        pca_cli.cli,
        ["apply-pca", "-i", str(tmp_path), "-o", str(tmp_path / "_pca"), *args],
    )


def test_fill_gaps_is_not_a_documented_option():
    result = CliRunner().invoke(pca_cli.cli, ["apply-pca", "--help"])
    assert result.exit_code == 0
    assert "fill-gaps" not in result.output


@pytest.mark.parametrize("value", ["True", "False"])
def test_typing_fill_gaps_stops_with_why(tmp_path, received, value):
    result = apply_pca(tmp_path, "--fill-gaps", value)
    assert result.exit_code != 0
    assert "always" in result.output and "timestamps" in result.output
    assert "Remove --fill-gaps" in result.output
    assert received == []


def test_a_config_file_holding_fill_gaps_is_accepted(tmp_path, received):
    config = tmp_path / "config.yaml"
    write_yaml({"fill_gaps": True}, str(config))
    result = apply_pca(tmp_path, "--config-file", str(config))
    assert result.exit_code == 0, result.output
    assert len(received) == 1 and "fill_gaps" not in received[0]
