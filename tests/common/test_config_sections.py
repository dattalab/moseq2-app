"""``config.yaml`` has one section per stage, and each step reads only its own.

A 1.6 ``config.yaml`` holds ``camera_type`` at its top level and the settings
of each stage in a section of their own: ``extract``, ``pca``, ``model`` and
``crowd_movies``. A name that means different things in two stages
(``max_height``, ``chunk_size``) or a compute resource (``cluster_type``) has a
value per stage, so setting it for extraction leaves PCA alone. Both front ends
of a step resolve the same section the same way, with no difference in value or
type between the command line and the notebook.

A key in a section that no step of that stage declares is ignored, and the step
prints one message naming every such key with the declared name closest to it,
then runs. A one-run switch (``run_cmd``, ``get_cmd``, ``verbose``,
``progress_bar``) in a section, ``camera_type`` inside a section and a setting at
the top level outside the sections are ignored the same way, each with a message
saying where the line belongs. A ``config.yaml`` without sections was written by
1.5.0 or earlier and stops with the message that says how to finish that project.

``generate-config`` writes the sections and, for an Azure Kinect or Orbbec Femto
project (``--camera-type azure``, or every recording an ``.mkv``), that camera's
preset values in them. A job settings file a kappa scan writes names the project
it belongs to, so a fit run from it finds the project and not the ``models/``
folder the file sits in.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from moseq2_common.yaml import read_yaml, write_yaml


def sectioned(path: Path, **sections) -> Path:
    """A 1.6 config.yaml at ``path`` holding ``sections`` (and ``camera_type: auto``)."""
    data = {"camera_type": "auto"}
    data.update(sections)
    write_yaml(data, str(path))
    return path


@pytest.fixture
def pca_steps(monkeypatch):
    """moseq2_pca's three steps replaced by recorders of the options each is handed."""
    import moseq2_pca.steps as steps

    calls = {"train_pca": [], "apply_pca": [], "compute_changepoints": []}
    results = {"train_pca": steps.TrainPCAResult, "apply_pca": steps.ApplyPCAResult,
               "compute_changepoints": steps.ChangepointsResult}

    def recorder(name):
        def record(input_dir, output_dir, output_file, opts, **kwargs):
            calls[name].append(opts)
            # a result naming no file written
            return results[name](**{f.name: None for f in dataclasses.fields(results[name])})
        return record

    for name in calls:
        monkeypatch.setattr(steps, name, recorder(name))
    return calls


def run_pca(command, project, config_file, *typed):
    from moseq2_pca.cli import cli

    return CliRunner().invoke(cli, [command, "-i", str(project / "aggregate_results"),
                                    "-o", str(project / "_pca"), "--config-file", str(config_file), *typed])


def test_pca_height_range_is_independent_of_extraction(tmp_path, pca_steps):
    """Extraction's height range stays extraction's: PCA keeps its own default."""
    config = sectioned(tmp_path / "config.yaml", extract={"max_height": 110}, pca={"min_height": 5})

    result = run_pca("train-pca", tmp_path, config)

    (opts,) = pca_steps["train_pca"]
    assert opts.min_height == 5
    assert opts.max_height == 120
    assert result.exit_code == 0, result.output


def test_choosing_slurm_for_extraction_leaves_pca_local(tmp_path, pca_steps):
    """``cluster_type`` is set per stage: SLURM extraction does not send PCA to SLURM."""
    from moseq2_common.options import read_settings
    from moseq2_extract.options import EXTRACT_STAGE, BatchExtractOptions
    from moseq2_pca.options import PCA_STAGE, TrainPCAOptions

    config = sectioned(tmp_path / "config.yaml", extract={"cluster_type": "slurm"})

    extraction = BatchExtractOptions.resolve(
        config=read_settings(config, BatchExtractOptions, stage_classes=EXTRACT_STAGE))
    pca = TrainPCAOptions.resolve(config=read_settings(config, TrainPCAOptions, stage_classes=PCA_STAGE))

    assert extraction.cluster_type == "slurm"
    assert pca.cluster_type == "local"
    result = run_pca("train-pca", tmp_path, config)
    assert result.exit_code == 0, result.output
    assert pca_steps["train_pca"][0].cluster_type == "local"


def test_unrecognized_settings_are_listed_and_ignored(tmp_path, pca_steps):
    """Keys no PCA step declares are named in one message, each with its closest name, and train-pca runs."""
    config = sectioned(tmp_path / "config.yaml",
                       pca={"chunk_size": 250, "gausfilter_space": [2, 2], "zzz_unheard_of": 1})

    result = run_pca("train-pca", tmp_path, config)

    assert result.exit_code == 0, result.output
    (opts,) = pca_steps["train_pca"]
    assert opts.chunk_size == 250
    assert opts.gaussfilter_space == (1.5, 1)
    message = (f"{config} has settings in its `pca` section that no pca step reads, so they are ignored: "
               "`gausfilter_space` (did you mean `gaussfilter_space`?), `zzz_unheard_of`. "
               "Correct or delete those lines to silence this message.")
    assert message in result.output, result.output
    assert result.output.count("no pca step reads") == 1, result.output


def test_a_one_run_switch_in_config_is_ignored_naming_its_flag(tmp_path, monkeypatch):
    """``verbose`` applies to one run; in config.yaml it is ignored, and the message names its flag."""
    import moseq2_model.steps as model_steps
    from moseq2_model.cli import cli

    fits = []
    monkeypatch.setattr(model_steps, "learn_model", lambda input_file, dest_file, opts, **kwargs: fits.append(opts))
    config = sectioned(tmp_path / "config.yaml", model={"verbose": True, "nlags": 2})
    scores = tmp_path / "pca_scores.h5"
    scores.write_bytes(b"")

    result = CliRunner().invoke(cli, ["learn-model", str(scores), str(tmp_path / "model.p"),
                                      "--config-file", str(config)])

    assert result.exit_code == 0, result.output
    (opts,) = fits
    assert opts.verbose is False
    assert opts.nlags == 2
    for named in ("`verbose`", "`model`", str(config), "--verbose", "verbose=", "ignored"):
        assert named in result.output, (named, result.output)


def test_camera_type_inside_a_section_is_ignored(tmp_path, capsys):
    """camera_type is read from the top level only; a copy inside a section is named and ignored."""
    from moseq2_common.options import read_settings
    from moseq2_extract.options import EXTRACT_STAGE, ExtractOptions

    config = sectioned(tmp_path / "config.yaml", extract={"camera_type": "azure", "chunk_size": 250})
    settings = read_settings(config, ExtractOptions, stage_classes=EXTRACT_STAGE)
    assert settings["camera_type"] == "auto"
    assert settings["chunk_size"] == 250

    write_yaml({"extract": {"camera_type": "azure"}, "pca": {}}, str(config))
    assert "camera_type" not in read_settings(config, ExtractOptions, stage_classes=EXTRACT_STAGE)

    printed = capsys.readouterr().out
    assert printed.count("`camera_type` in the `extract` section") == 2, printed
    for named in (str(config), "ignored", "top level"):
        assert named in printed, (named, printed)


def test_a_setting_outside_the_sections_is_ignored(tmp_path, pca_steps):
    """A setting at the top level beside the sections is named and ignored, and the step runs."""
    config = tmp_path / "config.yaml"
    write_yaml({"camera_type": "auto", "pca": {"min_height": 5}, "chunk_size": 250}, str(config))

    result = run_pca("train-pca", tmp_path, config)

    assert result.exit_code == 0, result.output
    (opts,) = pca_steps["train_pca"]
    assert opts.chunk_size == 4000
    assert opts.min_height == 5
    for named in (str(config), "`chunk_size`", "top level", "ignored"):
        assert named in result.output, (named, result.output)


def test_command_line_pca_on_an_mkv_project_uses_the_azure_filters(tmp_path, pca_steps):
    """generate-config in a folder of .mkv recordings writes the Azure preset, which train-pca reads."""
    from moseq2_extract.cli import cli as extract_cli

    for name in ("session_0", "session_1"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "depth.mkv").write_bytes(b"")
    config = tmp_path / "config.yaml"

    generated = CliRunner().invoke(extract_cli, ["generate-config", "-o", str(config)])
    assert generated.exit_code == 0, generated.output
    result = run_pca("train-pca", tmp_path, config)

    (opts,) = pca_steps["train_pca"]
    assert opts.gaussfilter_space == (2.25, 1.5)
    assert opts.tailfilter_size == (13, 13)
    assert result.exit_code == 0, result.output


# --------------------------------------------------------------------------
# One resolution for both front ends
# --------------------------------------------------------------------------

def _same(cli_opts, notebook_opts):
    """Field by field, values and types."""
    assert type(cli_opts) is type(notebook_opts)
    for field in dataclasses.fields(cli_opts):
        a, b = getattr(cli_opts, field.name), getattr(notebook_opts, field.name)
        assert a == b and type(a) is type(b), (field.name, a, b)


@pytest.fixture
def project(tmp_path):
    """A project folder with a generated config.yaml and the files the notebook functions check for."""
    from moseq2_extract.cli import cli as extract_cli

    result = CliRunner().invoke(extract_cli, ["generate-config", "-o", str(tmp_path / "config.yaml")])
    assert result.exit_code == 0, result.output
    (tmp_path / "aggregate_results").mkdir()
    (tmp_path / "_pca").mkdir()
    (tmp_path / "models").mkdir()
    write_yaml({"files": []}, str(tmp_path / "moseq2-index.yaml"))
    (tmp_path / "_pca" / "pca_scores.h5").write_bytes(b"")
    (tmp_path / "_pca" / "changepoints.h5").write_bytes(b"")
    return tmp_path


def progress_paths(project: Path) -> dict:
    return {
        "base_dir": str(project), "config_file": str(project / "config.yaml"),
        "train_data_dir": str(project / "aggregate_results"), "pca_dirname": str(project / "_pca"),
        "index_file": str(project / "moseq2-index.yaml"), "scores_path": str(project / "_pca" / "pca_scores.h5"),
        "model_path": str(project / "models" / "model.p"), "base_model_path": str(project / "models"),
        "changepoints_path": str(project / "_pca" / "changepoints.h5"), "plot_path": str(project),
    }


def test_both_front_ends_resolve_the_same_values(project, pca_steps, monkeypatch):
    """Each step with two front ends gets equal options from both, types included."""
    import moseq2_extract.steps as extract_steps
    import moseq2_model.steps as model_steps
    import moseq2_viz.steps as viz_steps
    from moseq2_extract.cli import cli as extract_cli
    from moseq2_extract.gui import extract_found_sessions
    from moseq2_model.cli import cli as model_cli
    from moseq2_model.gui import learn_model_command
    from moseq2_pca.gui import apply_pca_command, compute_changepoints_command, train_pca_command
    from moseq2_viz.cli import cli as viz_cli
    from moseq2_viz.gui import get_best_fit_model

    paths = progress_paths(project)
    config = project / "config.yaml"

    # PCA
    train_pca_command(paths, paths["pca_dirname"], "pca")
    assert run_pca("train-pca", project, config).exit_code == 0
    apply_pca_command(paths, "pca_scores")
    assert run_pca("apply-pca", project, config).exit_code == 0
    compute_changepoints_command(paths["train_data_dir"], paths, "changepoints")
    assert run_pca("compute-changepoints", project, config).exit_code == 0
    for name, (notebook, cli) in pca_steps.items():
        assert notebook.timeout == 5 and type(notebook.timeout) is int, name
        _same(cli, notebook)

    # extraction
    batches = []
    monkeypatch.setattr(extract_steps, "batch_extract", lambda input_dir, opts, **kwargs: batches.append(opts)
                        or extract_steps.BatchExtractResult(found=[], extracted=[], failures=[], script=None))
    extract_found_sessions(str(project), str(config), extract_all=True, skip_extracted=True)
    result = CliRunner().invoke(extract_cli, ["batch-extract", str(project), "--config-file", str(config),
                                              "--skip-completed"])
    assert result.exit_code == 0, result.output
    _same(batches[1], batches[0])

    # modeling
    fits = []
    monkeypatch.setattr(model_steps, "learn_model", lambda input_file, dest_file, opts, **kwargs: fits.append(opts))
    learn_model_command(paths)
    result = CliRunner().invoke(model_cli, ["learn-model", paths["scores_path"], paths["model_path"],
                                            "--config-file", str(config)])
    assert result.exit_code == 0, result.output
    _same(fits[1], fits[0])

    # model selection
    best = []

    class _Figure:
        def show(self, **_):
            pass

    monkeypatch.setattr(viz_steps, "get_best_model", lambda model_dir, cp_path, output_file, opts: best.append(opts)
                        or viz_steps.BestModelResult(best_model_info={}, figure=_Figure(), figure_files=[]))
    get_best_fit_model(paths, str(project / "best"))
    result = CliRunner().invoke(viz_cli, ["get-best-model", paths["base_model_path"], paths["changepoints_path"],
                                          str(project / "best")])
    assert result.exit_code == 0, result.output
    _same(best[1], best[0])


def test_a_depth_range_passed_explicitly_is_used_as_given():
    """A range a script passes (as one typed on the command line) is not replaced by the detected range."""
    from moseq2_extract.options import ExtractOptions, RoiOptions

    for cls in (RoiOptions, ExtractOptions):
        assert cls.resolve(explicit={"bg_roi_depth_range": (600, 700)}).manual_set_depth_range is True
        assert cls.resolve(config={"bg_roi_depth_range": (600, 700)}).manual_set_depth_range is False
        assert cls.resolve(explicit={"bg_roi_depth_range": (600, 700),
                                     "manual_set_depth_range": False}).manual_set_depth_range is False


# --------------------------------------------------------------------------
# Job files, and 1.5.0 projects
# --------------------------------------------------------------------------

def test_a_job_file_names_its_project(project):
    """A kappa scan's per-fit settings file belongs to the project, not to models/."""
    from moseq2_common.project import find_project
    from moseq2_model.gui import learn_model_command

    rng = np.random.default_rng(0)
    scores = project / "_pca" / "pca_scores.h5"
    with h5py.File(scores, "w") as f:
        for key, n in {"s0": 120, "s1": 140}.items():
            f[f"scores/{key}"] = rng.normal(size=(n, 10)).astype("float32")
            f[f"scores_idx/{key}"] = np.arange(n, dtype="float32")
    config = read_yaml(str(project / "config.yaml"))
    config["model"].update({"kappa": "scan", "n_models": 2})
    write_yaml(config, str(project / "config.yaml"))

    learn_model_command(progress_paths(project))

    job_files = sorted((project / "models").glob("model-*-settings.yaml"))
    assert len(job_files) == 2
    for job_file in job_files:
        found = find_project(str(job_file))
        assert found is not None and found.folder.resolve() == project.resolve(), (job_file, found)


def test_a_flat_config_stops_with_the_old_project_message(tmp_path, pca_steps):
    """A config.yaml without sections is a 1.5.0 project: the step stops and says how to finish it."""
    from moseq2_common.project import OldProjectError
    from moseq2_pca.gui import train_pca_command

    config = tmp_path / "config.yaml"
    write_yaml({"chunk_size": 250, "max_height": 110, "gaussfilter_space": [1.5, 1]}, str(config))

    result = run_pca("train-pca", tmp_path, config)

    assert result.exit_code != 0
    assert 'python -m pip install "moseq2-app==1.5.0"' in result.output, result.output
    assert str(config) in result.output
    assert pca_steps["train_pca"] == []
    with pytest.raises(OldProjectError, match="1.5.0"):
        train_pca_command(progress_paths(tmp_path), str(tmp_path / "_pca"), "pca")
