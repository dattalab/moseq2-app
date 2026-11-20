"""Extraction's project layer gives each step its project's folders, settings files and record.

``moseq2_extract.project`` resolves where a recording's outputs go in the
project (its session's folder in ``aggregate_results/``) and the uuid
``moseq2-project.yaml`` gave the session, starts the run's record with the
project (so its inputs are described relative to the project folder and include
``config.yaml`` and ``session_config.yaml``), and hands both to the file layer.
Batch extraction and aggregation are the project's alone, and writing a config
file or naming a flip classifier in one stops on a 1.5.0 project. What each
writes is what the step wrote when it took the project itself.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from fixtures import fetch_raw_sessions

from moseq2_common.bookkeeping import Session, load
from moseq2_common.options import read_settings
from moseq2_common.project import OldProjectError, Project
from moseq2_common.records import read_record
from moseq2_common.yaml import read_yaml, write_yaml
from moseq2_extract import project as project_steps
from moseq2_extract.options import (EXTRACT_STAGE, AggregateOptions, BatchExtractOptions, ExtractOptions,
                                    RoiOptions)
from moseq2_extract.recordings import session_outputs
from moseq2_extract.session_config import recording_entry

SESSION = "saline_example_0"
KEY = f"{SESSION}/depth.avi"
#: The uuid ``moseq2-project.yaml`` lists the session under.
LISTED_UUID = "5e55105-listed"

FLIP_BASE = "https://moseq-data.s3.amazonaws.com/flip-classifiers/"
FLIP_LIST = [
    ("large mice with fibers (K2)", "flip_classifier_k2_largemicewithfiber.npz"),
    ("adult male c57s (K2)", "flip_classifier_k2_c57_10to13weeks.npz"),
    ("mice with Inscopix cables (K2)", "flip_classifier_k2_inscopix.npz"),
    ("adult male c57s (Azure)", "flip_classifier_azure_temp.npz"),
]


def call(project, recording, opts, **kwargs):
    """``find_roi`` in ``project``, through the project layer."""
    return project_steps.find_roi(project, recording, opts, **kwargs)


def call_extract(project, recording, opts, output_dir=None):
    """``extract`` in ``project``, through the project layer."""
    return project_steps.extract(project, recording, opts, output_dir=output_dir)


def call_batch_extract(project, input_dir, opts, **kwargs):
    """``batch_extract`` in ``project``, through the project layer."""
    return project_steps.batch_extract(project, input_dir, opts, **kwargs)


def call_aggregate_results(project, opts, **kwargs):
    """``aggregate_results`` in ``project``, through the project layer."""
    return project_steps.aggregate_results(project, opts, **kwargs)


def call_generate_config(output_file, camera_type):
    """``generate_config`` writing a project's config file, through the project layer."""
    return project_steps.generate_config(output_file, camera_type, overwrite=True)


def call_download_flip_file(config_file, output_dir, selection):
    """``download_flip_file`` naming the classifier in ``config_file``, through the project layer."""
    return project_steps.download_flip_file(config_file, output_dir, selection)


def copy_session(folder: Path, name: str = SESSION, source: str = SESSION) -> Path:
    """A copy of the fixture recording ``source`` in ``folder``/``name``; its ``depth.avi``."""
    session = folder / name
    session.mkdir(parents=True)
    for file in ("depth.avi", "depth_ts.txt", "metadata.json"):
        shutil.copy(Path(fetch_raw_sessions()) / source / file, session / file)
    return session / "depth.avi"


@pytest.fixture
def project(tmp_path):
    """A project holding one fixture recording, a Kinect config.yaml, a session_config.yaml entry for the
    recording, and a moseq2-project.yaml listing nothing yet."""
    folder = tmp_path / "proj"
    copy_session(folder)
    project_steps.generate_config(str(folder / "config.yaml"), "kinect", overwrite=True)
    write_yaml({KEY: {"noise_tolerance": 25}}, str(folder / "session_config.yaml"))
    write_yaml({"sessions": {}, "pca": {}, "models": {"fitted": [], "current": None, "analysis": {}},
                "applied_models": []}, str(folder / "moseq2-project.yaml"))
    return Project.at(folder)


def resolved(project, recording, cls=RoiOptions):
    """The options the command line resolves for ``recording`` in ``project``."""
    return cls.resolve(config=read_settings(project.config_file, cls, stage_classes=EXTRACT_STAGE),
                       entry=recording_entry(project, str(recording)))


def list_session(project, key=KEY, uuid=LISTED_UUID):
    """List ``key`` in ``project``'s moseq2-project.yaml with ``uuid``, in its default folder."""
    write_yaml({"sessions": {key: {"recording": key, "folder": f"aggregate_results/{key.split('/')[0]}",
                                   "uuid": uuid}},
                "pca": {}, "models": {"fitted": [], "current": None, "analysis": {}}, "applied_models": []},
               str(project.bookkeeping_file))


def h5_uuid(h5_file) -> str:
    with h5py.File(h5_file, "r") as h5:
        value = h5["metadata/uuid"][()]
    return value.decode() if isinstance(value, bytes) else str(value)


def test_find_roi_in_a_project_writes_into_the_session_folder_and_records_the_project(project, monkeypatch):
    recording = project.folder / KEY
    opts = resolved(project, recording)
    assert opts.noise_tolerance == 25
    config_bytes = project.config_file.read_bytes()
    bookkeeping_bytes = project.bookkeeping_file.read_bytes()
    monkeypatch.chdir(project.folder)

    result = call(project, str(recording), opts)

    folder = session_outputs(project, KEY).folder
    assert folder == project.folder / "aggregate_results" / SESSION
    assert result.roi_files == [folder / "roi_00.tiff"]
    assert not (recording.parent / "proc").exists()
    record = read_record(folder / "roi_00.tiff")
    assert record["project"] == str(project.folder.resolve())
    assert [item["path"] for item in record["inputs"]] == [
        str(recording), str(project.config_file), str(project.session_config_file)]
    assert [item["relative_path"] for item in record["inputs"]] == [KEY, "config.yaml", "session_config.yaml"]
    assert record["options"]["noise_tolerance"] == 25
    assert "bg_roi_depth_range" in record["options"]
    assert project.config_file.read_bytes() == config_bytes
    assert project.bookkeeping_file.read_bytes() == bookkeeping_bytes


def test_find_roi_with_a_relative_config_file_writes_where_it_did(project, monkeypatch):
    """``find-roi session/depth.avi --config-file config.yaml`` from the project folder makes the project folder
    ``.``, so the session folder is the relative ``aggregate_results/<session>``, which ``get_roi_wrapper`` joins
    under the recording's folder. That is dattalab/moseq2-app#518; this pins where the images go until it is
    fixed."""
    import moseq2_extract.cli as extract_cli

    monkeypatch.chdir(project.folder)
    result = CliRunner().invoke(extract_cli.cli, ["find-roi", KEY, "--config-file", "config.yaml"],
                                catch_exceptions=False)
    assert result.exit_code == 0, result.output

    written = project.folder / SESSION / "aggregate_results" / SESSION
    assert (written / "roi_00.tiff").is_file()
    assert (written / "bground.tiff").is_file()
    assert not (project.folder / "aggregate_results").exists()


def test_extract_in_a_project_keeps_the_listed_uuid_and_records_the_project(project, monkeypatch):
    list_session(project)
    recording = project.folder / KEY
    opts = replace(resolved(project, recording, ExtractOptions), num_frames=20)
    config_bytes = project.config_file.read_bytes()
    bookkeeping_bytes = project.bookkeeping_file.read_bytes()
    monkeypatch.chdir(project.folder)

    result = call_extract(project, str(recording), opts)

    folder = project.folder / "aggregate_results" / SESSION
    assert result.h5_file == folder / "results.h5"
    assert h5_uuid(result.h5_file) == LISTED_UUID
    record = read_record(result.h5_file)
    assert record["project"] == str(project.folder.resolve())
    assert [item["path"] for item in record["inputs"][:2]] == [str(project.config_file),
                                                               str(project.session_config_file)]
    assert [item["relative_path"] for item in record["inputs"][:2]] == ["config.yaml", "session_config.yaml"]
    assert KEY in [item["relative_path"] for item in record["inputs"]]
    assert record["options"]["noise_tolerance"] == 25
    assert not (recording.parent / "proc").exists()
    assert project.config_file.read_bytes() == config_bytes
    assert project.bookkeeping_file.read_bytes() == bookkeeping_bytes


def test_extract_with_a_typed_output_dir_in_a_project_keeps_the_uuid_and_record(project, monkeypatch):
    """A typed ``--output-dir`` is relative to the recording's folder; the extraction is still the project's
    session, with its listed uuid and the project's record."""
    list_session(project)
    recording = project.folder / KEY
    opts = replace(resolved(project, recording, ExtractOptions), num_frames=20)
    monkeypatch.chdir(project.folder)

    result = call_extract(project, str(recording), opts, output_dir="proc_typed")

    assert result.h5_file == recording.parent / "proc_typed" / "results.h5"
    assert h5_uuid(result.h5_file) == LISTED_UUID
    record = read_record(result.h5_file)
    assert record["project"] == str(project.folder.resolve())
    assert [item["relative_path"] for item in record["inputs"][:2]] == ["config.yaml", "session_config.yaml"]
    assert not (project.folder / "aggregate_results").exists()


def test_extract_with_a_relative_config_file_writes_where_it_did(project, monkeypatch):
    """``extract session/depth.avi --config-file config.yaml`` from the project folder makes the project folder
    ``.``, so the session folder is the relative ``aggregate_results/<session>``, which ``extract_wrapper`` joins
    under the recording's folder. That is dattalab/moseq2-app#518; this pins where the outputs go until it is
    fixed."""
    import moseq2_extract.cli as extract_cli

    monkeypatch.chdir(project.folder)
    result = CliRunner().invoke(extract_cli.cli, ["extract", KEY, "--config-file", "config.yaml",
                                                  "--num-frames", "20"], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    written = project.folder / SESSION / "aggregate_results" / SESSION
    assert (written / "results.h5").is_file()
    assert read_record(written / "results.h5")["project"] == str(project.folder.resolve())
    assert not (project.folder / "aggregate_results").exists()


def test_batch_extract_extracts_each_recording_with_its_entry(project, monkeypatch, capsys):
    """Each recording is extracted with ``config.yaml``'s settings and its own ``session_config.yaml`` entry."""
    other = "saline_example_1"
    copy_session(project.folder, other, source=other)
    config = read_settings(project.config_file, BatchExtractOptions, stage_classes=EXTRACT_STAGE)
    opts = BatchExtractOptions.resolve(config=config, explicit={"num_frames": 20, "cluster_type": "local"})
    configured = opts.noise_tolerance
    assert configured != 25
    monkeypatch.chdir(project.folder)

    result = call_batch_extract(project, str(project.folder), opts, config_file=str(project.config_file))

    assert result.failures == []
    assert sorted(result.extracted) == [project.folder / KEY, project.folder / other / "depth.avi"]
    tolerances = {}
    for key in (KEY, f"{other}/depth.avi"):
        record = read_record(session_outputs(project, key).h5)
        assert record["project"] == str(project.folder.resolve())
        tolerances[key] = record["options"]["noise_tolerance"]
    assert tolerances == {KEY: 25, f"{other}/depth.avi": configured}
    assert capsys.readouterr().out.endswith("Extracted 2 of 2 recordings.\n")


def fake_extraction(project, name, uuid):
    """A recording ``name`` in ``project`` and the ``results.h5`` a finished extraction leaves in its session
    folder: frames deep enough to hold a mouse, and its uuid."""
    session = project.folder / name
    session.mkdir(parents=True)
    (session / "depth.avi").write_bytes(b"\0")
    folder = project.folder / "aggregate_results" / name
    folder.mkdir(parents=True)
    with h5py.File(folder / "results.h5", "w") as h5:
        h5.create_dataset("frames", data=np.full((4, 8, 8), 30, dtype="uint8"))
        h5.create_dataset("metadata/uuid", data=uuid)


def test_aggregate_results_lists_the_extracted_sessions(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "proj"
    folder.mkdir()
    project_steps.generate_config(str(folder / "config.yaml"), "kinect", overwrite=True)
    project = Project.at(folder)
    fake_extraction(project, "session_a", "uuid-a")
    fake_extraction(project, "session_b", "uuid-b")
    (folder / "session_c").mkdir()
    (folder / "session_c" / "depth.avi").write_bytes(b"\0")
    monkeypatch.chdir(folder)

    result = call_aggregate_results(project, AggregateOptions())

    assert result.sessions == ["session_a/depth.avi", "session_b/depth.avi"]
    assert load(project).sessions == {
        "session_a/depth.avi": Session(key="session_a/depth.avi", recording="session_a/depth.avi",
                                       folder="aggregate_results/session_a", uuid="uuid-a"),
        "session_b/depth.avi": Session(key="session_b/depth.avi", recording="session_b/depth.avi",
                                       folder="aggregate_results/session_b", uuid="uuid-b"),
    }
    assert capsys.readouterr().out == (f"2 session(s) listed in {project.bookkeeping_file}; it lists 2 in all.\n")


def test_generate_config_stops_on_a_1_5_project(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    write_yaml({"bg_roi_depth_range": [650, 750], "chunk_size": 1000, "camera_type": "k2"},
               str(folder / "config.yaml"))
    before = {path: path.read_bytes() for path in folder.iterdir()}

    with pytest.raises(OldProjectError):
        call_generate_config(str(folder / "config.yaml"), "auto")
    with pytest.raises(OldProjectError):
        call_generate_config(str(folder / "new.yaml"), "kinect")

    assert {path: path.read_bytes() for path in folder.iterdir()} == before


@pytest.fixture
def downloads(monkeypatch):
    """Each download asked for, as (url, file); the file gets a few placeholder bytes."""
    calls = []

    def urlretrieve(url, filename, *args, **kwargs):
        calls.append((url, filename))
        Path(filename).write_bytes(b"npz")
        return filename, None

    monkeypatch.setattr(urllib.request, "urlretrieve", urlretrieve)
    return calls


def test_download_flip_file_names_the_classifier_in_config_yaml(project, downloads):
    out = project.folder / "flips"

    written = call_download_flip_file(str(project.config_file), str(out), 1)

    expected = out / FLIP_LIST[1][1]
    assert Path(written) == expected
    assert downloads == [(FLIP_BASE + FLIP_LIST[1][1], str(expected))]
    config = read_yaml(str(project.config_file))
    assert config["extract"]["flip_classifier"] == str(expected)
    assert "flip_classifier" not in config


def test_download_flip_file_on_the_command_line_asks_as_before(project, monkeypatch, downloads):
    """``download-flip-file`` lists the classifiers, asks until it gets a listed number, downloads that one
    into the working folder and names it in ``config.yaml``."""
    import moseq2_extract.cli as extract_cli

    monkeypatch.chdir(project.folder)
    here = os.getcwd()
    expected = os.path.join(here, FLIP_LIST[1][1])

    result = CliRunner().invoke(extract_cli.cli, ["download-flip-file"], input="x\n7\n1\n", catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert result.output == (
        "".join(f"[{index}] {label} ---> {FLIP_BASE + name}\n" for index, (label, name) in enumerate(FLIP_LIST))
        + "Enter a selection Please enter a valid number listed above\n"
        + "Enter a selection Please enter a valid number listed above\n"
        + f"Enter a selection Successfully downloaded flip file to {expected}\n"
    )
    assert downloads == [(FLIP_BASE + FLIP_LIST[1][1], expected)]
    assert read_yaml(str(project.config_file))["extract"]["flip_classifier"] == expected
