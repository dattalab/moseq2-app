"""The SLURM options and the sbatch line, in one place.

``SlurmOptions`` holds the seven SLURM options ``kappa-scan`` takes, with its
``--get-cmd`` default, and ``sbatch_command`` writes the line ``kappa-scan``
writes and ``moseq2 extract`` runs, the job's command quoted as ``--wrap``'s one
argument. Where extraction runs, on a cluster included, is its
``computing.extract`` block, and on a cluster ``moseq2 extract`` always says
each job it submits.
"""

from __future__ import annotations


from click.testing import CliRunner

import moseq2_extract.cli as extract_cli
import moseq2_model.cli as model_cli
from moseq2_common.slurm import SlurmOptions, sbatch_command
from moseq2_common.yaml import read_yaml, write_yaml
from slurm_fakes import fake_slurm  # noqa: F401 -- the fixture


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
    """kappa-scan's jobs request what the notebook's scan requested in 1.5.0; extraction's 16GB and auto.

    ``auto`` gives each extraction job three times its recording's length, at
    least an hour.

    In 1.5.0 the notebook's kappa-scan jobs took ``memory`` and ``wall_time``
    from PCA's settings in the flat config.yaml (15GB, 06:00:00). The model
    section declares those as its own defaults, on the command line, in the
    notebook and in the config.yaml ``generate-config`` writes.
    """
    from moseq2_extract.options import ExtractComputing
    from moseq2_model.options import KappaScanOptions

    assert (KappaScanOptions().memory, KappaScanOptions().wall_time) == ("15GB", "06:00:00")
    assert (ExtractComputing().memory, ExtractComputing().wall_time) == ("16GB", "auto")

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
    computing = written["computing"]["extract"]
    assert (computing["memory"], computing["wall_time"]) == ("16GB", "auto")
    assert "memory" not in written["extract"] and "wall_time" not in written["extract"]


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


def test_kappa_scan_lists_its_commands_by_default(tmp_path, monkeypatch):
    """``kappa-scan`` prints the commands it writes unless told not to.

    In 1.5.0 ``kappa-scan`` printed them only with ``--get-cmd``, while the
    notebook's scan always did.
    """
    assert _kappa_scan(tmp_path, monkeypatch, []) is True
    assert _kappa_scan(tmp_path, monkeypatch, ["--get-cmd"]) is True
    assert _kappa_scan(tmp_path, monkeypatch, ["--no-get-cmd"]) is False


def test_moseq2_extract_always_says_each_job_it_submits(tmp_path, monkeypatch, fake_slurm):
    """On a cluster, ``moseq2 extract`` says each job it submits, and has no switch to stop it."""
    from moseq2_app.cli import cli

    fake_slurm.hold()
    (tmp_path / "session_1").mkdir()
    (tmp_path / "session_1" / "depth.avi").write_bytes(b"\0")  # a placeholder: nothing is extracted on a cluster
    (tmp_path / "flip.npz").write_bytes(b"\0")  # a placeholder head/tail classifier, so nothing is downloaded
    config = tmp_path / "config.yaml"
    result = CliRunner().invoke(extract_cli.cli, ["generate-config", "-o", str(config)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    data = read_yaml(str(config))
    data["extract"]["flip_classifier"] = str(tmp_path / "flip.npz")
    # a time per job given, since `auto` reads the placeholder recording's length
    data["computing"]["extract"].update(cluster_type="slurm", wall_time="3:00:00")
    write_yaml(data, str(config))
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["extract"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "session_1/depth.avi: submitted as job 1001." in result.stderr, result.stderr
    assert len(fake_slurm.calls("sbatch")) == 1
    for switch in ("--get-cmd", "--no-get-cmd"):
        refused = CliRunner().invoke(cli, ["extract", switch])
        assert refused.exit_code == 2 and f"No such option '{switch}'" in refused.output, refused.output
