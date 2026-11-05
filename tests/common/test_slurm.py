"""The SLURM options and the sbatch line, in one place.

``batch-extract`` and ``kappa-scan`` each declared the same seven SLURM options
and built the same ``sbatch`` line in their own words. ``SlurmOptions`` holds
the seven, with one ``--get-cmd`` default for both, and ``sbatch_command``
writes the line both commands write.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_model.cli as model_cli
from moseq2_common.slurm import SlurmOptions, sbatch_command


def test_sbatch_command_matches_batch_extracts_line():
    assert sbatch_command("moseq2-extract extract --config-file c.yaml d.dat; ", ncpus=1,
                          memory="5GB", partition="short", wall_time="3:00:00", prefix="") == (
        'sbatch -c 1 --mem=5GB -p short -t 3:00:00 '
        '--wrap "moseq2-extract extract --config-file c.yaml d.dat; "'
    )


def test_sbatch_command_matches_kappa_scans_line():
    assert sbatch_command("moseq2-model learn-model s.h5 m.p --kappa 10", ncpus=8, memory="5GB",
                          partition="short", wall_time="3:00:00", prefix="module load x; ") == (
        'sbatch -c 8 --mem=5GB -p short -t 3:00:00 '
        '--wrap "module load x; moseq2-model learn-model s.h5 m.p --kappa 10"'
    )


def test_slurm_options_declare_the_shared_seven():
    assert SlurmOptions().as_dict() == {
        "cluster_type": "local", "prefix": "", "memory": "5GB",
        "wall_time": "3:00:00", "partition": "short", "get_cmd": True, "run_cmd": False,
    }


def _batch_extract(tmp_path, monkeypatch, args):
    """``batch-extract``'s ``get_cmd``, with the step replaced by a stand-in."""
    seen = {}

    def stand_in(input_folder, opts, **_):
        seen["opts"] = opts
        return SimpleNamespace(failures=[])

    monkeypatch.setattr(extract_cli.steps, "batch_extract", stand_in)
    config = tmp_path / "config.yaml"
    config.write_text("extract:\n  cluster_type: slurm\n")
    result = CliRunner().invoke(extract_cli.cli, ["batch-extract", str(tmp_path), "--config-file", str(config),
                                                  *args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return seen["opts"].get_cmd


def _kappa_scan(tmp_path, monkeypatch, args):
    """``kappa-scan``'s ``get_cmd``, with the step replaced by a stand-in."""
    seen = {}
    monkeypatch.setattr(model_cli.steps, "kappa_scan",
                        lambda input_file, output_dir, opts, **_: seen.setdefault("opts", opts))
    scores = tmp_path / "pca_scores.h5"
    scores.touch()
    result = CliRunner().invoke(model_cli.cli, ["kappa-scan", str(scores), str(tmp_path / "models"),
                                                "--cluster-type", "slurm", *args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return seen["opts"].get_cmd


@pytest.mark.parametrize("submitter", [_batch_extract, _kappa_scan], ids=["batch-extract", "kappa-scan"])
def test_both_submitters_list_their_commands_by_default(submitter, tmp_path, monkeypatch):
    """``batch-extract`` and ``kappa-scan`` print the commands they write unless told not to.

    In 1.5.0 ``batch-extract`` always printed them (``--get-cmd`` defaulted to
    True and could not be turned off) and ``kappa-scan`` printed them only with
    ``--get-cmd``, while the notebook's scan always did.
    """
    assert submitter(tmp_path, monkeypatch, []) is True
    assert submitter(tmp_path, monkeypatch, ["--get-cmd"]) is True
    assert submitter(tmp_path, monkeypatch, ["--no-get-cmd"]) is False
