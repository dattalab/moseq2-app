"""Extraction's project layer gives each step its project's folders, settings files and record.

``moseq2_extract.project`` resolves where a recording's outputs go in the
project (its session's folder in ``aggregate_results/``) and the uuid
``moseq2-project.yaml`` gave the session, starts the run's record with the
project (so its inputs are described relative to the project folder and include
``config.yaml`` and ``session_config.yaml``), and hands both to the file layer.
The Extract step (``extract``) is the project's alone: it extracts each
recording with its own entry and lists every recording it plans as a session in
``moseq2-project.yaml``, keeping the uuid an extraction already holds. Writing a
config file or naming a flip classifier in one stops on a 1.5.0 project. What
each writes is what the step wrote when it took the project itself.
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

from moseq2_app.cli import cli
from moseq2_common.bookkeeping import Session, load
from moseq2_common.options import read_settings
from moseq2_common.project import OldProjectError, Project
from moseq2_common.records import read_record
from moseq2_common.yaml import read_yaml, write_config, write_yaml
from moseq2_extract import project as project_steps
from moseq2_extract.options import EXTRACT_STAGE, ExtractOptions, RoiOptions
from moseq2_extract.recordings import session_outputs
from moseq2_extract.session_config import recording_entry
from reporting import CollectingReporter

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
    """One recording's extraction in ``project``, through the project layer (``extract_recording``)."""
    return project_steps.extract_recording(project, recording, opts, output_dir=output_dir)


#: The fixture's head/tail classifier, a copy of the published adult C57 one.
CLASSIFIER = "flip_classifier_k2_c57_10to13weeks.npz"


def name_classifier(project):
    """Copy the fixture's head/tail classifier into ``project`` and name it in config.yaml, so its
    extraction downloads nothing."""
    shutil.copy(Path(fetch_raw_sessions()) / CLASSIFIER, project.folder / CLASSIFIER)
    data = read_yaml(str(project.config_file))
    data["extract"]["flip_classifier"] = str(project.folder / CLASSIFIER)
    write_config(data, str(project.config_file))


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


def test_moseq2_extract_from_the_project_folder_writes_into_the_session_folder(project, monkeypatch):
    """``moseq2 extract session/depth.avi`` from the project folder takes the project from the working folder
    as an absolute path, so the outputs go into the session folder in the project's ``aggregate_results/``,
    not one joined under the recording's folder (dattalab/moseq2-app#518)."""
    name_classifier(project)
    monkeypatch.chdir(project.folder)
    result = CliRunner().invoke(cli, ["extract", KEY, "--num-frames", "20"], catch_exceptions=False)
    assert result.exit_code == 0, result.output

    written = project.folder / "aggregate_results" / SESSION
    assert (written / "results.h5").is_file()
    assert read_record(written / "results.h5")["project"] == str(project.folder.resolve())
    assert not (project.folder / SESSION / "aggregate_results").exists()


def test_batch_extract_extracts_each_recording_with_its_entry(project, monkeypatch):
    """Each recording is extracted with ``config.yaml``'s settings and its own ``session_config.yaml`` entry."""
    other = "saline_example_1"
    copy_session(project.folder, other, source=other)
    name_classifier(project)
    configured = ExtractOptions.resolve(read_settings(project.config_file, ExtractOptions,
                                                      stage_classes=EXTRACT_STAGE)).noise_tolerance
    assert configured != 25
    monkeypatch.chdir(project.folder)
    reporter = CollectingReporter()

    run = project_steps.extract(project, settings={"num_frames": 20, "cluster_type": "local"}, reporter=reporter)

    assert run.failed == {}
    assert sorted(run.extracted) == [KEY, f"{other}/depth.avi"]
    tolerances = {}
    for key in (KEY, f"{other}/depth.avi"):
        record = read_record(session_outputs(project, key).h5)
        assert record["project"] == str(project.folder.resolve())
        tolerances[key] = record["options"]["noise_tolerance"]
    assert tolerances == {KEY: 25, f"{other}/depth.avi": configured}
    assert reporter.statuses[-1] == "Extracted 2 of 2 recordings."


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


def test_extraction_lists_each_session_with_the_uuid_its_extraction_holds(tmp_path, monkeypatch):
    """Before extracting anything, the Extract step lists every recording it plans in ``moseq2-project.yaml``:
    one extracted already keeps the uuid its ``results.h5`` holds, and one not yet extracted gets a new uuid."""
    folder = tmp_path / "proj"
    folder.mkdir()
    project_steps.generate_config(str(folder / "config.yaml"), "kinect", overwrite=True)
    project = Project.at(folder)
    name_classifier(project)
    fake_extraction(project, "session_a", "uuid-a")
    fake_extraction(project, "session_b", "uuid-b")
    (folder / "session_c").mkdir()
    (folder / "session_c" / "depth.avi").write_bytes(b"\0")
    monkeypatch.chdir(folder)
    listed_at_extraction = []
    monkeypatch.setattr(project_steps, "extract_recording",
                        lambda project_, input_file, opts, **kwargs: listed_at_extraction.append(
                            dict(load(project_).sessions)))

    run = project_steps.extract(project)

    keys = ["session_a/depth.avi", "session_b/depth.avi", "session_c/depth.avi"]
    assert sorted(run.extracted) == keys
    sessions = load(project).sessions
    assert {key: sessions[key] for key in keys[:2]} == {
        "session_a/depth.avi": Session(key="session_a/depth.avi", recording="session_a/depth.avi",
                                       folder="aggregate_results/session_a", uuid="uuid-a"),
        "session_b/depth.avi": Session(key="session_b/depth.avi", recording="session_b/depth.avi",
                                       folder="aggregate_results/session_b", uuid="uuid-b"),
    }
    new = sessions["session_c/depth.avi"]
    assert (new.recording, new.folder) == ("session_c/depth.avi", "aggregate_results/session_c")
    assert new.uuid not in (None, "", "uuid-a", "uuid-b")
    assert sorted(sessions) == keys
    assert listed_at_extraction == [sessions] * 3


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
