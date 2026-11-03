"""The YAML files the packages write keep their bytes.

Every package writes YAML through ``moseq2_common.yaml``. Before that module
existed, extract, pca, viz and app each carried their own copy of one writer,
and a user's existing files, configs, indexes, session results, the PCA
parameter yaml and crowd-movie info, are in that writer's format. These tests
hold each file the notebook chain writes to it.

Stored expected files cannot do this, since the files hold temporary paths and
uuids. The reference is that writer itself, kept here, applied to the data the
real file holds: the file must be exactly what it produces for that data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

from moseq2_common.yaml import read_yaml


def legacy_bytes(path, scratch):
    """The bytes the pre-moseq2-common writer produces for the data in ``path``.

    This is the writer extract, pca, viz and app each carried as ``_yaml.py``:
    a fresh safe, pure-Python ``YAML`` with ``allow_unicode = False`` and
    ``width = 82``, dumped into a file opened in text mode. It is kept here as
    the reference the packages' output is compared with.
    """
    data = read_yaml(str(path))
    parser = YAML(typ="safe", pure=True)
    parser.allow_unicode = False
    parser.width = 82
    with open(scratch, "w") as handle:
        parser.dump(data, handle)
    return Path(scratch).read_bytes()


def assert_written_as_before(path, tmp_path):
    path = Path(path)
    assert path.read_bytes() == legacy_bytes(path, tmp_path / "legacy.yaml"), path


@pytest.mark.parametrize("camera_type", ["k2", "azure"])
def test_generated_config_is_written_as_before(tmp_path, camera_type):
    from moseq2_extract.gui import generate_config_command

    config = tmp_path / "config.yaml"
    generate_config_command(str(config), camera_type=camera_type)
    assert_written_as_before(config, tmp_path)


def test_crowd_movie_info_is_written_as_before(tmp_path):
    from moseq2_viz.io.video import write_crowd_movie_info_file

    model_fit = {
        "model_parameters": {
            "model_class": "ARHMM",
            "kappa": np.int64(212414),
            "gamma": 1000.0,
            "alpha": 5.7,
        }
    }
    write_crowd_movie_info_file("model.p", model_fit, "moseq2-index.yaml", str(tmp_path))
    assert_written_as_before(tmp_path / "info.yaml", tmp_path)


def test_session_config_is_written_as_before(arena, tmp_path):
    assert_written_as_before(arena, tmp_path)


def test_results_yaml_is_written_as_before(extracted, tmp_path):
    results = [p for p in Path(extracted).rglob("results_00.yaml") if p.parent.name == "proc"]
    assert results, f"no proc/results_00.yaml under {extracted}"
    for path in results:
        assert_written_as_before(path, tmp_path)


def test_index_is_written_as_before(index_as_aggregated, aggregated, grouped, tmp_path):
    assert_written_as_before(index_as_aggregated, tmp_path)
    assert_written_as_before(aggregated, tmp_path)


def test_aggregated_results_are_written_as_before(aggregated, tmp_path):
    results = sorted((Path(aggregated).parent / "aggregate_results").glob("*.yaml"))
    assert results, f"no yaml in {Path(aggregated).parent / 'aggregate_results'}"
    for path in results:
        assert_written_as_before(path, tmp_path)


def test_pca_parameters_are_written_as_before(pca, tmp_path):
    assert_written_as_before(Path(pca) / "pca.yaml", tmp_path)


def test_project_config_is_written_as_before(configured, pca, tmp_path):
    assert_written_as_before(configured, tmp_path)
