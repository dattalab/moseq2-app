"""The learn-model commands a kappa scan writes.

``moseq2-model kappa-scan``, and the notebook's model cell with ``kappa`` set
to ``scan``, write one ``learn-model`` command per kappa. Each fit's options
are written to its own settings file beside its model, and its command runs
``learn-model --config-file`` on that file. Whatever the scan was given has to
reach each fit, or every model in the scan is fit with the defaults instead.
"""

from __future__ import annotations

import dataclasses
import shlex

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_model.steps as model_steps
from moseq2_common.yaml import write_yaml
from moseq2_model.cli import cli
from moseq2_model.options import LearnModelOptions

#: A value away from its default for each learn-model option, the scan's fitting options first.
#: ``var_name`` stays ``scores``: the scan reads the scores itself to choose its kappas.
SCAN_FITTING = dict(
    check_every=7, robust=True, separate_trans=True, nlags=5, noise_level=0.2, alpha=100.0, gamma=500.0,
    load_groups=False, percent_split=30, progressbar=False, whiten="each", npcs=8, max_states=20,
    save_model=False, e_step=True, num_iter=50, ncpus=4, nfolds=3, hold_out_seed=17,
    hold_out=True,
)
#: The learn-model options a kappa scan does not take on the command line; the notebook's config.yaml holds them.
LEARN_ONLY = dict(checkpoint_freq=10, use_checkpoint=True, default_group="saline", verbose=True)


def _declared(values):
    """``values`` limited to the options learn-model declares."""
    names = {f.name for f in dataclasses.fields(LearnModelOptions)}
    return {k: v for k, v in values.items() if k in names}


@pytest.fixture
def scan_inputs(tmp_path):
    rng = np.random.default_rng(0)
    scores = tmp_path / "pca_scores.h5"
    with h5py.File(scores, "w") as f:
        for key, n in {"s0": 120, "s1": 140, "s2": 160}.items():
            f[f"scores/{key}"] = rng.normal(size=(n, 10)).astype("float32")
            f[f"scores_idx/{key}"] = np.arange(n, dtype="float32")
    index = tmp_path / "moseq2-index.yaml"
    write_yaml({"files": []}, str(index))
    return scores, index


def _fits(script, monkeypatch):
    """Run each learn-model command in ``script`` through learn-model's command line; each fit's options and index."""
    fits = []
    monkeypatch.setattr(model_steps, "learn_model",
                        lambda input_file, dest_file, opts, index=None, **_: fits.append((opts, index)))
    for line in script.splitlines():
        if not line.startswith("moseq2-model learn-model "):
            assert line == "set -e", line
            continue
        result = CliRunner().invoke(cli, shlex.split(line)[1:], catch_exceptions=False)
        assert result.exit_code == 0, result.output
    return fits


@pytest.mark.parametrize("front_end", ["cli", "notebook"])
def test_kappa_scan_jobs_carry_every_learn_model_option(front_end, tmp_path, scan_inputs, monkeypatch):
    """Each fit a kappa scan writes runs with every learn-model option the scan was given.

    The command line's kappa-scan takes the fitting options; the notebook's scan
    reads them, and the options only learn-model has, from config.yaml.
    """
    scores, index = scan_inputs
    models = tmp_path / "models"
    given = _declared(SCAN_FITTING) if front_end == "cli" else _declared({**SCAN_FITTING, **LEARN_ONLY})
    defaults = LearnModelOptions()
    assert all(getattr(defaults, k) != v for k, v in given.items())

    if front_end == "cli":
        kappa_scan = cli.commands["kappa-scan"]
        typed = []
        for name, value in given.items():
            param = next(p for p in kappa_scan.params if p.name == name)
            typed += [param.opts[-1]] if param.is_flag else [param.opts[-1], str(value)]
        result = CliRunner().invoke(cli, ["kappa-scan", str(scores), str(models), "-i", str(index), "--n-models", "3",
                                          *typed], catch_exceptions=False)
        assert result.exit_code == 0, result.output
    else:
        from moseq2_model.gui import learn_model_command

        config = tmp_path / "config.yaml"
        write_yaml({**given, "kappa": "scan", "scan_scale": "log", "n_models": 3}, str(config))
        learn_model_command({"scores_path": str(scores), "model_path": str(models / "model.p"),
                             "config_file": str(config), "index_file": str(index),
                             "base_model_path": str(models)}, get_cmd=False)

    fits = _fits((models / "train_out.sh").read_text(), monkeypatch)

    kappas = [opts.kappa for opts, _ in fits]
    assert len(kappas) == 3 and kappas == sorted(set(kappas)), kappas
    for opts, fit_index in fits:
        expected = dataclasses.replace(defaults, **given, kappa=opts.kappa)
        got, want = opts.as_dict(), expected.as_dict()
        assert {k: (got[k], want[k]) for k in want if got[k] != want[k]} == {}, "(fit's value, scan's value)"
        assert fit_index == str(index)
