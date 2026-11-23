"""Where each step runs is a block of its own under config.yaml's ``computing`` section.

``moseq2_common.computing``: a step that does long work reads ``computing.<step>``
(local or SLURM, cores, partition, memory, time, prefix), so two steps sharing
their other settings can run in different places. ``generate-config`` writes a
block for every installed step; the SLURM settings are no longer in ``extract``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from moseq2_common.computing import ComputingOptions, available_cores, gigabytes, hours
from moseq2_common.options import read_settings, save_settings
from moseq2_common.project import Project
from moseq2_common.yaml import read_yaml, write_config


@dataclass(frozen=True, kw_only=True)
class _OtherComputing(ComputingOptions):
    STEP = "train-pca"


def test_each_step_reads_its_own_computing_block(tmp_path):
    from moseq2_extract.options import ExtractComputing

    config = tmp_path / "config.yaml"
    write_config({"extract": {}, "computing": {"extract": {"cluster_type": "slurm", "memory": "32GB"},
                                               "train-pca": {"cores": 2}}}, str(config))

    assert read_settings(config, ExtractComputing, stage_classes=()) == {"cluster_type": "slurm", "memory": "32GB"}
    assert read_settings(config, _OtherComputing, stage_classes=()) == {"cores": 2}
    resolved = ExtractComputing.resolve(config=read_settings(config, ExtractComputing, stage_classes=()))
    assert (resolved.cluster_type, resolved.memory, resolved.wall_time) == ("slurm", "32GB", "auto")


def test_a_generated_config_holds_a_computing_block_per_installed_step(tmp_path):
    from moseq2_extract.steps import generate_config

    config = generate_config(str(tmp_path / "config.yaml"), "auto", overwrite=True)
    data = read_yaml(str(config))

    assert data["computing"]["extract"] == {"cluster_type": "local", "cores": None, "partition": "short",
                                            "memory": "16GB", "wall_time": "auto", "prefix": ""}
    slurm_keys = {"cluster_type", "prefix", "memory", "wall_time", "partition", "ncpus", "get_cmd", "run_cmd",
                  "extract_out_script"}
    assert not slurm_keys & set(data["extract"])
    assert not {"threads", "compress_threads"} & set(data["extract"])


def test_saved_computing_settings_land_in_their_block(tmp_path):
    from moseq2_extract.options import ExtractComputing, ExtractOptions

    project = Project.at(tmp_path)
    write_config({"camera_type": "auto", "extract": {"crop_size": [80, 80]}, "exclude": []}, str(project.config_file))

    save_settings(project, ExtractComputing, {"cluster_type": "slurm", "memory": "24GB"})
    save_settings(project, ExtractOptions, {"crop_size": (100, 100), "camera_type": "kinect"})

    data = read_yaml(str(project.config_file))
    assert data["computing"] == {"extract": {"cluster_type": "slurm", "memory": "24GB"}}
    assert data["extract"] == {"crop_size": [100, 100]}
    assert data["camera_type"] == "kinect"
    assert data["exclude"] == []


def test_cores_default_to_what_this_process_may_use():
    assert available_cores() == len(os.sched_getaffinity(0))


def test_conversions():
    assert gigabytes("16GB") == 16
    assert gigabytes("512MB") == 0.5
    assert hours("1:30:00") == 1.5
    assert hours("2-00:00:00") == 48
    assert hours("auto") is None
    with pytest.raises(ValueError):
        gigabytes("lots")
