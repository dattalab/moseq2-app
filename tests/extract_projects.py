"""Small projects for the extraction tests: a few of the fixture recordings, cut short.

``small_project`` copies the named recordings of the raw-sessions fixture into a
new project folder, writes its ``config.yaml`` as ``moseq2_extract.steps.generate_config``
does, and points extraction at the fixture's own head/tail classifier, so
nothing is downloaded. ``num_frames`` keeps each extraction to a few seconds.
``dat_recording`` writes a short Kinect v2 ``.dat`` recording from a fixture
``.avi``, with its timestamps and metadata.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from moseq2_common.project import Project
from moseq2_common.yaml import read_yaml, write_config

#: The fixture's head/tail classifier, a copy of the published adult C57 one.
CLASSIFIER = "flip_classifier_k2_c57_10to13weeks.npz"


def small_project(folder: Path, raw_sessions: Path, recordings=("saline_example_0",), *, num_frames: int | None = 60,
                  extract: dict | None = None, classifier: bool = True) -> Project:
    """A project in ``folder`` holding ``recordings`` from the fixture, with its ``config.yaml``."""
    from moseq2_extract.steps import generate_config

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in recordings:
        shutil.copytree(Path(raw_sessions) / name, folder / name)
    project = Project.at(folder)
    generate_config(str(project.config_file), "auto", overwrite=True)
    settings = {"num_frames": num_frames, **(extract or {})}
    if classifier:
        shutil.copy(Path(raw_sessions) / CLASSIFIER, folder / CLASSIFIER)
        settings["flip_classifier"] = str(folder / CLASSIFIER)
    data = read_yaml(str(project.config_file))
    data["extract"].update(settings)
    write_config(data, str(project.config_file))
    return project


def dat_recording(folder: Path, raw_sessions: Path, n_frames: int = 100, source: str = "saline_example_0") -> Path:
    """``folder/depth.dat``: the first ``n_frames`` of a fixture recording, with its timestamps and metadata."""
    from moseq2_extract.io.video import get_movie_info, load_movie_data

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    original = Path(raw_sessions) / source
    avi = str(original / "depth.avi")
    finfo = get_movie_info(avi)
    frames = load_movie_data(avi, range(n_frames), frame_size=finfo["dims"], finfo=finfo)
    frames.astype("<u2").tofile(folder / "depth.dat")
    lines = (original / "depth_ts.txt").read_text().splitlines(keepends=True)[:n_frames]
    (folder / "depth_ts.txt").write_text("".join(lines))
    shutil.copy(original / "metadata.json", folder / "metadata.json")
    return folder / "depth.dat"


def recorded_extraction(project: Project, key: str, *, record: bool = True, options: dict | None = None,
                        listed: bool = True) -> Path:
    """Write ``key``'s session folder as an extraction would leave it, without extracting: a ``results.h5``
    holding a uuid and, with ``record``, the record of a run with the options the project resolves for the
    recording now (``options`` over them), its head/tail detection, and the recording as its input. With
    ``listed`` the session is listed in ``moseq2-project.yaml``. Returns the ``results.h5``."""
    from dataclasses import replace

    import h5py

    from moseq2_common.bookkeeping import Session, relative_path, update
    from moseq2_common.project import settings_files
    from moseq2_common.records import record_text, start_record
    from moseq2_extract.project import head_tail_detection
    from moseq2_extract.recordings import session_outputs
    from moseq2_extract.session_config import extract_settings, read_entries, recording_entry, recording_options

    recording = Path(project.folder) / key
    outputs = session_outputs(project, key)
    outputs.folder.mkdir(parents=True, exist_ok=True)
    uuid = f"uuid-of-{key}"
    with h5py.File(outputs.h5, "w") as f:
        f.create_dataset("metadata/uuid", data=uuid)
        if record:
            detection = head_tail_detection(project)
            opts = recording_options(extract_settings(project.config_file),
                                     recording_entry(project, recording, read_entries(project)))
            opts = replace(opts, flip_classifier=str(detection.path), **(options or {}))
            run = start_record("extract", opts, settings_files(project), project=project)
            inputs = [{"path": str(recording)}]
            if detection.path.is_file():
                inputs.append(str(detection.path))
            run.add_inputs(inputs)
            f.attrs["provenance"] = record_text(run.finish({}))
    if listed:
        update(project, lambda b: b.sessions.__setitem__(key, Session(
            key=key, recording=key, folder=relative_path(project, outputs.folder), uuid=uuid)))
    return outputs.h5
