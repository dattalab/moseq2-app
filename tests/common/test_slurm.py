"""The SLURM options and the sbatch line, in one place.

``batch-extract`` and ``kappa-scan`` each declared the same seven SLURM options
and built the same ``sbatch`` line in their own words. ``SlurmOptions`` holds
the seven, with one ``--get-cmd`` default for both, and ``sbatch_command``
writes the line both commands write, the job's command quoted as ``--wrap``'s
one argument.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_model.cli as model_cli
from moseq2_common.slurm import SlurmOptions, sbatch_command
from moseq2_common.yaml import read_yaml


def test_sbatch_command_matches_batch_extracts_line():
    assert sbatch_command("moseq2-extract extract --config-file c.yaml d.dat; ", ncpus=1,
                          memory="5GB", partition="short", wall_time="3:00:00", prefix="") == (
        "sbatch -c 1 --mem=5GB -p short -t 3:00:00 "
        "--wrap 'moseq2-extract extract --config-file c.yaml d.dat; '"
    )


def test_sbatch_command_matches_kappa_scans_line():
    assert sbatch_command("moseq2-model learn-model s.h5 m.p --kappa 10", ncpus=8, memory="5GB",
                          partition="short", wall_time="3:00:00", prefix="module load x; ") == (
        "sbatch -c 8 --mem=5GB -p short -t 3:00:00 "
        "--wrap 'module load x; moseq2-model learn-model s.h5 m.p --kappa 10'"
    )


def test_slurm_options_declare_the_shared_seven():
    assert SlurmOptions().as_dict() == {
        "cluster_type": "local", "prefix": "", "memory": "5GB",
        "wall_time": "3:00:00", "partition": "short", "get_cmd": True, "run_cmd": False,
    }


def test_kappa_scan_jobs_request_15gb_and_6_hours_by_default(tmp_path, monkeypatch):
    """kappa-scan's jobs request what the notebook's scan requested in 1.5.0; batch-extract's keep 5GB, 3 hours.

    In 1.5.0 the notebook's kappa-scan jobs took ``memory`` and ``wall_time``
    from PCA's settings in the flat config.yaml (15GB, 06:00:00). The model
    section declares those as its own defaults, on the command line, in the
    notebook and in the config.yaml ``generate-config`` writes.
    """
    from moseq2_extract.options import BatchExtractOptions
    from moseq2_model.options import KappaScanOptions

    assert (KappaScanOptions().memory, KappaScanOptions().wall_time) == ("15GB", "06:00:00")
    assert (BatchExtractOptions().memory, BatchExtractOptions().wall_time) == ("5GB", "3:00:00")

    seen = {}
    monkeypatch.setattr(model_cli.steps, "kappa_scan",
                        lambda input_file, output_dir, opts, **_: seen.setdefault("opts", opts))
    monkeypatch.chdir(tmp_path)
    scores = tmp_path / "pca_scores.h5"
    scores.touch()
    result = CliRunner().invoke(model_cli.cli, ["kappa-scan", str(scores), str(tmp_path / "models"),
                                                "--cluster-type", "slurm"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert (seen["opts"].memory, seen["opts"].wall_time) == ("15GB", "06:00:00")

    config = tmp_path / "config.yaml"
    result = CliRunner().invoke(extract_cli.cli, ["generate-config", "-o", str(config)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    written = read_yaml(str(config))
    assert (written["model"]["memory"], written["model"]["wall_time"]) == ("15GB", "06:00:00")
    assert (written["extract"]["memory"], written["extract"]["wall_time"]) == ("5GB", "3:00:00")


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
