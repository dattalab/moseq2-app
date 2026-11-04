"""Each modeling step is one function in ``moseq2_model.steps``, called by both front ends.

``learn-model``, ``kappa-scan`` and ``apply-model`` on the command line and
``learn_model_command`` and ``apply_model_command`` in the notebook resolve
their settings into one options object and hand it to the same step, which
runs the work. A step also runs from a script, on its options class's
defaults, with no ``config.yaml`` at all.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_model.cli as model_cli
from moseq2_common.yaml import read_yaml, write_yaml

SESSIONS = {"s0": 120, "s1": 140, "s2": 160}

#: The script the kappa-scan step writes for the declared defaults with cluster_type slurm, ncpus 0, three
#: models and a prefix, on the scores below (420 frames), with the paths relative to the working folder.
#: Each command passes its fit's settings file to --config-file.
SCAN_SCRIPT = (
    'set -e\n'
    'sbatch -c 8 --mem=5GB -p short -t 3:00:00 --wrap "module load x; moseq2-model learn-model pca_scores.h5 '
    'models/model-000-1.p --config-file models/model-000-1-settings.yaml"\n'
    'sbatch -c 8 --mem=5GB -p short -t 3:00:00 --wrap "module load x; moseq2-model learn-model pca_scores.h5 '
    'models/model-001-100.p --config-file models/model-001-100-settings.yaml"\n'
    'sbatch -c 8 --mem=5GB -p short -t 3:00:00 --wrap "module load x; moseq2-model learn-model pca_scores.h5 '
    'models/model-002-10000.p --config-file models/model-002-10000-settings.yaml"'
)


@pytest.fixture
def steps():
    import moseq2_model.steps

    return moseq2_model.steps


@pytest.fixture
def options():
    import moseq2_model.options

    return moseq2_model.options


@pytest.fixture
def scores(tmp_path):
    """Three sessions of random 10-PC scores."""
    rng = np.random.default_rng(0)
    path = tmp_path / "pca_scores.h5"
    with h5py.File(path, "w") as f:
        for key, n in SESSIONS.items():
            f[f"scores/{key}"] = rng.normal(size=(n, 10)).astype("float32")
            f[f"scores_idx/{key}"] = np.arange(n, dtype="float32")
    return path


@pytest.fixture
def called(monkeypatch, steps):
    """Each step replaced by a recorder of its options object."""
    calls = {"learn_model": [], "kappa_scan": [], "apply_model": []}

    def learn_model(input_file, dest_file, opts, **kwargs):
        calls["learn_model"].append(opts)
        return steps.LearnModelResult(model_file=dest_file, figure_file=None)

    def kappa_scan(input_file, output_dir, opts, **kwargs):
        calls["kappa_scan"].append(opts)
        return steps.KappaScanResult(script=opts.out_script, commands="")

    def apply_model(model_file, pc_file, dest_file, opts, **kwargs):
        calls["apply_model"].append(opts)
        return steps.ApplyModelResult(applied_file=dest_file)

    for name, recorder in (("learn_model", learn_model), ("kappa_scan", kappa_scan),
                           ("apply_model", apply_model)):
        monkeypatch.setattr(steps, name, recorder)
    return calls


def test_every_model_command_calls_its_step(tmp_path, scores, called, options):
    model = tmp_path / "model.p"
    model.touch()
    for args in (["learn-model", str(scores), str(tmp_path / "out.p")],
                 ["kappa-scan", str(scores), str(tmp_path / "models")],
                 ["apply-model", str(model), str(scores), str(tmp_path / "applied.p")]):
        result = CliRunner().invoke(model_cli.cli, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output

    classes = {"learn_model": options.LearnModelOptions, "kappa_scan": options.KappaScanOptions,
               "apply_model": options.ApplyModelOptions}
    for name, cls in classes.items():
        assert len(called[name]) == 1, name
        assert type(called[name][0]) is cls, name


def test_the_notebook_functions_call_the_steps(tmp_path, scores, called, options):
    from moseq2_model.gui import apply_model_command, learn_model_command

    config = tmp_path / "config.yaml"
    index = tmp_path / "moseq2-index.yaml"
    write_yaml({"files": []}, str(index))
    paths = {"scores_path": str(scores), "model_path": str(tmp_path / "models" / "model.p"),
             "config_file": str(config), "index_file": str(index),
             "base_model_path": str(tmp_path / "models")}

    write_yaml({"kappa": 1000, "num_iter": 7}, str(config))
    learn_model_command(paths)
    write_yaml({"kappa": "scan", "scan_scale": "log", "n_models": 4}, str(config))
    learn_model_command(paths)
    apply_model_command(paths, str(tmp_path / "models" / "model.p"))

    assert [type(o) for o in called["learn_model"]] == [options.LearnModelOptions]
    assert (called["learn_model"][0].kappa, called["learn_model"][0].num_iter) == (1000, 7)
    assert [type(o) for o in called["kappa_scan"]] == [options.KappaScanOptions]
    assert called["kappa_scan"][0].n_models == 4
    assert called["kappa_scan"][0].out_script == str(tmp_path / "models" / "train_out.sh")
    assert [type(o) for o in called["apply_model"]] == [options.ApplyModelOptions]


def test_learn_model_runs_from_a_script_without_a_config_file(tmp_path, scores, steps, options):
    result = steps.learn_model(str(scores), str(tmp_path / "m.p"),
                               options.LearnModelOptions(num_iter=2, progressbar=False))

    assert result.model_file.is_file()
    assert list(tmp_path.rglob("config.yaml")) == []


def test_kappa_scan_commands_are_unchanged(tmp_path, scores, steps, options, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opts = options.KappaScanOptions(cluster_type="slurm", ncpus=0, n_models=3, prefix="module load x; ",
                                    out_script="models/train_out.sh")

    result = steps.kappa_scan("pca_scores.h5", "models", opts)

    assert result.script.read_text().splitlines() == SCAN_SCRIPT.splitlines()
    assert result.commands == SCAN_SCRIPT
    for i, kappa in enumerate((1, 100, 10000)):
        settings = read_yaml(str(tmp_path / "models" / steps.KAPPA_SCAN_SETTINGS_NAME.format(i, kappa)))
        assert settings == options.LearnModelOptions(kappa=float(kappa)).as_dict()


def test_learn_model_reads_config_file(tmp_path, scores, steps, monkeypatch):
    received = []
    monkeypatch.setattr(steps, "learn_model_wrapper", lambda i, d, config: received.append(config))
    config = tmp_path / "config.yaml"
    write_yaml({"kappa": 1000, "num_iter": 7}, str(config))

    def learn(*args):
        result = CliRunner().invoke(
            model_cli.cli, ["learn-model", str(scores), str(tmp_path / "model.p"), "--config-file", str(config), *args],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        return received.pop()

    fit = learn()
    assert (fit["kappa"], fit["num_iter"]) == (1000, 7)

    fit = learn("--kappa", "50")
    assert (fit["kappa"], fit["num_iter"]) == (50, 7)
