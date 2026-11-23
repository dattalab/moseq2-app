"""The extraction steps in a project: the project layer.

Each function here is one step run in a project (``moseq2_common.project.Project``).
It resolves what the project decides for the step and calls the file layer
(``moseq2_extract.steps``) with explicit paths, options and a record started
with the project, so the record names the project folder and describes each
input relative to it. The notebook's Extract step (``moseq2_app.notebook``) and
``moseq2 extract`` both call ``extract`` here with the same arguments, so the
two compute the same thing.

``extract(project)`` is the Extract step. Holding the step's lock
(``moseq2_common.project.step_lock``), it:

1. saves the settings it was given (``settings=``) where they belong: the
   ``extract`` section of ``config.yaml``, the ``computing.extract`` block, and,
   for an arena setting given with recordings named, those recordings' entries
   in ``session_config.yaml``;
2. makes sure the project's head/tail detection is at hand
   (``ensure_head_tail_detection``), downloading the published classifier into
   the project folder the first time it is needed;
3. decides what the run will do (``plan``): every included recording is new,
   up to date, or to be redone and why;
4. lists every planned recording as a session in ``moseq2-project.yaml``
   before any extraction starts, so no separate step lists them afterwards;
5. extracts what the plan says on this computer, one recording at a time with
   ``threads`` and ``compress_threads`` set to the cores it may use, a failure
   recorded and the rest extracted; or, on a SLURM cluster, writes each
   recording's job settings file in its session folder and submits one job per
   recording (``moseq2_common.slurm.submit``), each asking for ``JOB_CPUS``
   cores and running ``moseq2 run-job`` (``run_job``) through the Python that
   submitted it, and records the submission beside the recording's outputs
   (``SUBMITTED_FILE``: the job's id, when, the ``sbatch`` line run, the job's
   settings file and its log, ``JOB_LOG``). It returns once the jobs are
   submitted.

On a cluster, each job reports through a ``ProgressFileReporter`` into its
recording's ``PROGRESS_FILE``. ``cluster_runs`` reads the submission records,
the progress files and one ``squeue`` (then one ``sacct``) into where each
recording's job stands, and why one failed; a recording whose job is queued or
running is ``in progress`` in the plan, so a rerun neither submits it again nor
redoes it. ``stop`` cancels the jobs, and ``follow`` reports each job's changes
until every one has ended. Nothing depends on the process that submitted the
jobs staying alive: a notebook opened later, or ``moseq2 extract`` run again,
reads the same records.

Head/tail detection (``head_tail_detection``) is, in order: the classifier
``extract.flip_classifier`` names; the one trained in the project
(``TRAINED_FLIP_CLASSIFIER``); the published one for the project's ``animals``
and camera. ``extract_recording`` extracts one recording in the project, as a
job does, and ``find_roi`` finds one recording's arena.

``generate_config`` and ``download_flip_file`` take the project's config file
rather than a ``Project``, since writing it is what makes a folder a project and
it may have any name. Both stop on a MoSeq2 1.5.0 project
(``moseq2_common.project.check_not_old``) before writing anything.

The file layer is reached through its module (``steps.extract``), never by a
name imported from it. What each step does is reported through
``moseq2_common.report``.
"""

from __future__ import annotations

import math
import os
import shlex
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime
from os.path import abspath, basename, dirname, exists, join
from pathlib import Path
from uuid import uuid4

import h5py

from moseq2_common.bookkeeping import Session, load, relative_path, update
from moseq2_common.computing import available_cores, gigabytes, hours
from moseq2_common.errors import ExpectedError, StepStopped
from moseq2_common.options import arena_fields, check_top_level, read_config, read_settings, save_settings, \
    setting_fields
from moseq2_common.project import Project, check_not_old, read_exclude, session_key, settings_files, step_lock
from moseq2_common.records import read_record, start_record, unchanged
from moseq2_common.report import ProgressFileReporter, Reporter, checkpoint, read_progress, reporting
from moseq2_common.slurm import JobState, SubmitError, cancel, job_command, job_settings_file, job_states, \
    sbatch_command, submit
from moseq2_common.yaml import dump_block_yaml, read_yaml, write_config
from moseq2_extract import steps
from moseq2_extract.helpers.extract import EXTRACTION_BAR, extraction_summary
from moseq2_extract.options import (
    EXTRACT_STAGE,
    DiagnosticsOptions,
    ExtractComputing,
    ExtractOptions,
    ProjectOptions,
    RecordingsOptions,
    RoiOptions,
)
from moseq2_extract.recordings import find_recordings, project_keys, session_outputs
from moseq2_extract.session_config import extract_settings, read_entries, recording_entry, recording_options, \
    save_entry
from moseq2_extract.setup import project_camera
from moseq2_extract.util import filter_warnings

__all__ = ["JOB_CPUS", "JOB_LOG", "JOB_NAME", "PROGRESS_FILE", "STALLED_AFTER", "SUBMITTED_FILE",
           "TRAINED_FLIP_CLASSIFIER", "ExtractRun", "HeadTail", "HeadTailDownloadError", "PlannedRecording",
           "RecordingRun", "cluster_runs", "download_flip_file", "ensure_head_tail_detection", "extract",
           "extract_recording", "find_roi", "follow", "generate_config", "head_tail_detection",
           "included_recordings", "plan", "run_job", "stop"]

#: The head/tail classifier trained in a project (the flip-classifier notebook), in the project folder.
TRAINED_FLIP_CLASSIFIER = "flip_classifier_trained.pkl"

#: The CPU cores each extraction job on a cluster asks for; its reading and compressing threads follow.
JOB_CPUS = 4

#: A recording's submission record, in its session folder: its job's id, when and how it was submitted.
SUBMITTED_FILE = "extract-submitted.yaml"

#: A recording's progress file, in its session folder, which its cluster job rewrites as it goes.
PROGRESS_FILE = "extract-progress.json"

#: SLURM's log of a recording's job (the job's output), in its session folder.
JOB_LOG = "extract-job.log"

#: The name extraction jobs have in the cluster's queue.
JOB_NAME = "moseq2-extract"

#: Seconds after which the progress file of a job no longer in the queue, not rewritten since, counts as stopped.
STALLED_AFTER = 600


# ------------------------------------------------------------------------------- head/tail detection


@dataclass(frozen=True)
class HeadTail:
    """The head/tail detection a project's extractions use."""

    #: ``setting`` (``extract.flip_classifier``), ``trained`` (``TRAINED_FLIP_CLASSIFIER``) or ``published``.
    source: str
    #: The classifier file; a published one may not be downloaded yet.
    path: Path
    #: What it detects, as messages name it: ``adult C57 mice``, ``your trained classifier``.
    label: str
    #: Where a published classifier downloads from; None for the others.
    url: str | None = None


#: The published classifier for each camera and ``animals``. Other animals get adult C57 detection, and an
#: Azure camera gets its one classifier for every choice.
_PUBLISHED = {
    ("kinect", "c57"): "adult male c57s (K2)",
    ("kinect", "other"): "adult male c57s (K2)",
    ("kinect", "fiber"): "large mice with fibers (K2)",
    ("kinect", "inscopix"): "mice with Inscopix cables (K2)",
    ("azure", "c57"): "adult male c57s (Azure)",
    ("azure", "other"): "adult male c57s (Azure)",
    ("azure", "fiber"): "adult male c57s (Azure)",
    ("azure", "inscopix"): "adult male c57s (Azure)",
}

#: The animals each ``animals`` choice detects, as messages name them.
_DETECTS = {"c57": "adult C57 mice", "other": "adult C57 mice", "fiber": "large mice with fiber implants",
            "inscopix": "mice with Inscopix cables"}


class HeadTailDownloadError(ExpectedError):
    """The published head/tail classifier could not be downloaded."""


def head_tail_detection(project: Project) -> HeadTail:
    """The head/tail detection ``project``'s extractions use; nothing is downloaded.

    ``extract.flip_classifier`` when set (a relative path is relative to the project folder); else the
    project's trained classifier, ``TRAINED_FLIP_CLASSIFIER``, when it exists; else the published classifier
    for the project's ``animals`` and camera (``camera_type``, or the camera its recordings show when that is
    ``auto``), in the project folder under its published name.
    """
    config = extract_settings(project.config_file) if Path(project.config_file).is_file() else {}
    given = config.get("flip_classifier")
    if given:
        path = Path(given)
        if not path.is_absolute() and (Path(project.folder) / path).exists():
            path = Path(project.folder) / path
        return HeadTail("setting", path, path.name)
    trained = Path(project.folder) / TRAINED_FLIP_CLASSIFIER
    if trained.is_file():
        return HeadTail("trained", trained, "your trained classifier")
    settings = ProjectOptions.resolve(read_settings(project.config_file, ProjectOptions, stage_classes=())
                                      if Path(project.config_file).is_file() else {})
    camera = {"k2": "kinect", "realsense": "kinect"}.get(settings.camera_type, settings.camera_type)
    if camera == "auto":
        camera = project_camera(project.folder)
    url = steps.FLIP_CLASSIFIERS[_PUBLISHED[(camera, settings.animals)]]
    return HeadTail("published", Path(project.folder) / basename(url), _DETECTS[settings.animals], url)


def ensure_head_tail_detection(project: Project, *, reporter: Reporter | None = None) -> HeadTail:
    """``head_tail_detection(project)``, downloading a published classifier into the project folder when it is
    not there yet. A download that fails raises ``HeadTailDownloadError`` and leaves nothing behind."""
    with reporting(reporter) as reporter:
        detection = head_tail_detection(project)
        if detection.source != "published" or detection.path.is_file():
            return detection
        reporter.status(f"Getting head/tail detection for {detection.label} (first time only)…")
        partial = detection.path.with_name(f".{detection.path.name}.partial")
        try:
            urllib.request.urlretrieve(detection.url, partial)
            os.replace(partial, detection.path)
        except Exception as error:  # noqa: BLE001 -- every failed download is the same problem to the person
            if partial.exists():
                partial.unlink()
            raise HeadTailDownloadError(
                f"The download didn't finish ({error}). The arena tool's previews and extraction can't run "
                "without it.",
                title=f"Couldn't get head/tail detection for {detection.label}",
                fix="check this computer's internet connection, then run this cell again.",
                command_line_fix="check this computer's internet connection and run the command again.",
                help_page="Why do you need to know what my animals are?") from error
        return detection


# ------------------------------------------------------------------------------- what a run will do


@dataclass(frozen=True)
class PlannedRecording:
    """What a run will do with one recording."""

    #: The session key: the recording's path in the project.
    key: str
    recording: Path
    #: ``new`` (not extracted), ``redo``, ``done`` (up to date) or ``in progress``.
    plan: str
    #: Why it is redone, as its row says it ("its arena was adjusted"); empty otherwise.
    reason: str = ""
    #: The settings that changed, when that is the reason, for a row's tooltip and the command line.
    changed: tuple = ()


#: Why an extraction carrying no readable record is redone.
NO_RECORD = "no record of how it was made"


def included_recordings(project: Project) -> list[tuple[str, Path]]:
    """Each recording the project's extraction covers, as (session key, path), by key.

    The recordings found in the project folder (``RecordingsOptions``: their extensions, how deep), and the
    sessions ``moseq2-project.yaml`` lists whose recording is gone but whose extraction is there, less those
    ``config.yaml``'s ``exclude`` leaves out.
    """
    excluded = set(read_exclude(project))
    return [(key, recording) for key, recording in _recordings(project) if key not in excluded]


def _recordings(project: Project, search=None) -> list[tuple[str, Path]]:
    if search is None:
        search = _search(project)
    found = {session_key(project, recording): Path(recording) for recording in search.recordings}
    if Path(project.bookkeeping_file).is_file():
        everything = set(found) | set(load(project).sessions)
        for key in load(project).sessions:
            if key not in found and session_outputs(project, key, keys=everything).h5.is_file():
                found[key] = Path(project.folder) / key
    return sorted(found.items())


def _search(project: Project):
    opts = RecordingsOptions.resolve(read_settings(project.config_file, RecordingsOptions,
                                                   stage_classes=EXTRACT_STAGE))
    return find_recordings(str(project.folder), extensions=opts.extensions, max_depth=opts.max_depth)


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        return float(a) == float(b)
    return a == b


#: The settings the camera sets when ``camera_type`` is ``auto`` (``moseq2_extract.util.CAMERA_DEFAULTS``).
_CAMERA_SET = ("camera_type", "bg_roi_weights", "pixel_format", "movie_dtype")

#: Settings a run sets itself, never compared with a record: the head/tail classifier (compared as
#: detection), and the skip a project's extraction always turns off.
_NOT_COMPARED = ("flip_classifier", "skip_completed")


def _changed_settings(resolved: ExtractOptions, recorded: Mapping) -> list[str]:
    """The settings ``resolved`` holds that differ from the ``recorded`` options, leaving out the one-run
    switches and what extraction detects: the floor's depth range unless it is set by hand, and the camera's
    settings when the camera is told from the recording (``auto``, or a recording that showed another camera
    than the one stated)."""
    names = [name for name in setting_fields(ExtractOptions) if name not in _NOT_COMPARED]
    if not resolved.manual_set_depth_range:
        names.remove("bg_roi_depth_range")
    if resolved.camera_type == "auto" or recorded.get("camera_type") not in (None, resolved.camera_type):
        names = [name for name in names if name not in _CAMERA_SET]
    values = resolved.as_dict()
    return [name for name in names if not _same(_plain(values[name]), recorded.get(name))]


def _plain(value):
    return [_plain(v) for v in value] if isinstance(value, (list, tuple)) else value


def _recorded_input(record: Mapping, path: Path) -> Mapping | None:
    target = os.path.abspath(path)
    for entry in record.get("inputs") or ():
        if isinstance(entry, Mapping) and entry.get("path") == target:
            return entry
    return None


def _detection_changed(record: Mapping, detection: HeadTail) -> bool:
    used = (record.get("options") or {}).get("flip_classifier")
    if not used or Path(str(used)).name != detection.path.name:
        return True
    entry = _recorded_input(record, Path(str(used)))
    if entry is None or not detection.path.is_file() or not isinstance(entry.get("fingerprint"), Mapping):
        return False
    return not unchanged(entry, detection.path)


def _recording_changed(record: Mapping, recording: Path) -> bool:
    if not recording.exists():
        return False  # moved to cold storage: its extraction stays as it is
    entry = _recorded_input(record, recording)
    if entry is None or not isinstance(entry.get("fingerprint"), Mapping):
        return False
    return not unchanged(entry, recording)


def _redo_reason(changed: Sequence[str], entry: Mapping) -> str:
    if changed and all(name in entry for name in changed):
        return "its arena was adjusted"
    if len(changed) == 1:
        return f"{changed[0]} changed"
    return f"{len(changed)} settings changed"


def plan(project: Project, *, recordings: Sequence[str] | None = None) -> list[PlannedRecording]:
    """What a run of extraction in ``project`` would do with each included recording, by key.

    ``in progress`` (its reason says how the job stands): its cluster job is queued or running
    (``cluster_runs``). ``new``: it has no ``results.h5``. ``redo`` (with its reason): its extraction carries
    no readable record (``NO_RECORD``); the recording changed since it was extracted (a recording that is gone
    is not a change); the head/tail detection now resolved differs from the one it used; or its options as
    resolved now (``config.yaml``, its entry in ``session_config.yaml``) differ from the record's in a setting
    that is not detected or a one-run switch. ``done`` otherwise. ``recordings`` (paths or session keys) are
    listed too when they are not included, as recordings to extract. Nothing is written or downloaded.
    """
    runs = cluster_runs(project)
    detection = head_tail_detection(project)
    search = _search(project)
    excluded = set(read_exclude(project))
    listed = {key: path for key, path in _recordings(project, search) if key not in excluded}
    for name in recordings or ():
        key = _key(project, name)
        listed.setdefault(key, Path(project.folder) / key)
    everything = set(listed) | set(project_keys(project))
    config = extract_settings(project.config_file)
    entries = read_entries(project)
    out = []
    for key, recording in sorted(listed.items()):
        if key in runs and runs[key].live:
            out.append(PlannedRecording(key, recording, "in progress", _how_it_stands(runs[key])))
            continue
        h5 = session_outputs(project, key, keys=everything).h5
        if not h5.is_file():
            out.append(PlannedRecording(key, recording, "new"))
            continue
        try:
            record = read_record(h5)
        except Exception:  # noqa: BLE001 -- an unreadable record is no record
            record = None
        if not isinstance(record, Mapping) or not isinstance(record.get("options"), Mapping):
            out.append(PlannedRecording(key, recording, "redo", NO_RECORD))
            continue
        entry = recording_entry(project, recording, entries)
        resolved = recording_options(config, entry)
        if _recording_changed(record, recording):
            out.append(PlannedRecording(key, recording, "redo", "the recording changed since it was extracted"))
        elif _detection_changed(record, detection):
            out.append(PlannedRecording(key, recording, "redo", "its head/tail detection changed"))
        else:
            changed = _changed_settings(resolved, record["options"])
            if changed:
                out.append(PlannedRecording(key, recording, "redo", _redo_reason(changed, entry), tuple(changed)))
            else:
                out.append(PlannedRecording(key, recording, "done"))
    return out


def _key(project: Project, name: str | os.PathLike) -> str:
    """The session key of a recording named by its path (from the working folder) or by its key."""
    path = Path(name)
    if not path.is_absolute() and not path.exists() and (Path(project.folder) / path).exists():
        return Path(name).as_posix()
    return session_key(project, path)


def redo_line(planned: PlannedRecording) -> str:
    """What the command line says about a recording the run extracts again."""
    if planned.reason == NO_RECORD:
        return (f"{planned.key}: extracted again, because nothing records how its extraction was made (an older "
                "MoSeq2, `moseq2 standalone` or a hand copy made it).")
    reason = planned.reason
    if len(planned.changed) > 1:
        reason += f" ({', '.join(planned.changed)})"
    return f"{planned.key}: extracted again, because {reason}."


# ------------------------------------------------------------------------------- the run


@dataclass(frozen=True)
class ExtractRun:
    """What a run of extraction did."""

    #: The recordings extracted, by session key.
    extracted: list
    #: The recordings whose extraction failed, and the error each raised.
    failed: dict
    #: The recordings the plan left as they were, up to date. One whose cluster job is queued or running is in
    #: none of these lists.
    up_to_date: list
    #: On a cluster: each recording submitted, its job settings file, by session key.
    jobs: dict


#: Where a run was started, as ``step_lock`` records it, by the front end ``start_record`` detects.
_WHERE = {"cli": "command line", "notebook": "notebook", "script": "script"}


def _where() -> str:
    from moseq2_common.records import _invocation

    return _WHERE.get(_invocation()[0], "script")


#: The settings ``extract``'s ``settings=`` may hold, by the class whose section saves them.
_SAVED_CLASSES = (ExtractOptions, ExtractComputing, RecordingsOptions, DiagnosticsOptions)


def save_extract_settings(project: Project, settings: Mapping, recordings: Sequence[str] | None = None) -> None:
    """Save ``settings`` typed for a run of extraction: each where it belongs.

    Extraction's settings go to ``config.yaml``'s ``extract`` section (``camera_type`` to its top level), its
    computing settings to ``computing.extract``. With ``recordings`` named, an arena setting goes instead to
    each named recording's entry in ``session_config.yaml``, kept with the entry's other settings. A typed
    ``bg_roi_depth_range`` is saved with ``manual_set_depth_range`` on, unless that is typed too, so the range
    is used as given. One-run switches are not saved. A setting no extraction class declares raises
    ``TypeError``.
    """
    remaining = dict(settings)
    if "bg_roi_depth_range" in remaining:
        # a range typed is used as given (moseq2_extract.options, RoiOptions.resolve)
        remaining.setdefault("manual_set_depth_range", True)
    if recordings:
        arena = {name: remaining.pop(name) for name in list(remaining) if name in arena_fields(ExtractOptions)}
        if arena:
            entries = read_entries(project)
            for name in recordings:
                key = _key(project, name)
                save_entry(project, key, {**entries.get(key, {}), **_yaml_values(arena)})
    for cls in _SAVED_CLASSES:
        mine = {name: remaining.pop(name) for name in list(remaining) if name in {f.name for f in fields(cls)}}
        if mine:
            save_settings(project, cls, mine)
    if remaining:
        raise TypeError(f"Extraction has no setting {', '.join(map(repr, sorted(remaining)))}.")


def _yaml_values(values: Mapping) -> dict:
    return {name: list(value) if isinstance(value, tuple) else value for name, value in values.items()}


def _computing(project: Project) -> ExtractComputing:
    return ExtractComputing.resolve(read_settings(project.config_file, ExtractComputing, stage_classes=()))


def _list_sessions(project: Project, planned: Sequence[PlannedRecording]) -> dict:
    """Give every planned recording's session its folder and uuid in ``moseq2-project.yaml``; their uuids.

    A session the file lists keeps its uuid; one extracted already keeps the uuid its ``results.h5`` holds;
    any other gets a new one.
    """
    listed = load(project).sessions
    everything = {p.key for p in planned} | set(listed) | set(project_keys(project))
    sessions = {}
    for item in planned:
        outputs = session_outputs(project, item.key, keys=everything)
        if item.key in listed and listed[item.key].uuid:
            uuid = listed[item.key].uuid
        elif outputs.h5.is_file():
            uuid = _h5_uuid(outputs.h5) or str(uuid4())
        else:
            uuid = str(uuid4())
        sessions[item.key] = Session(key=item.key, recording=item.key, folder=relative_path(project, outputs.folder),
                                     uuid=uuid)

    def add(bookkeeping):
        for session in sessions.values():
            bookkeeping.sessions[session.key] = session

    if sessions:
        update(project, add)
    return {key: session.uuid for key, session in sessions.items()}


def _h5_uuid(h5: Path) -> str | None:
    try:
        with h5py.File(h5, "r") as f:
            value = f["metadata/uuid"][()]
    except (OSError, KeyError):
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def _options(project: Project, targets: Sequence[PlannedRecording], detection: HeadTail, threads: int) -> dict:
    """Each target's options: ``config.yaml``, its entry, then the run's own: its head/tail detection, no
    skip, and ``threads`` for reading and compressing."""
    config = extract_settings(project.config_file)
    entries = read_entries(project)
    return {item.key: replace(recording_options(config, recording_entry(project, item.recording, entries)),
                              flip_classifier=str(detection.path), skip_completed=False, threads=threads,
                              compress_threads=threads)
            for item in targets}


class _OneRecording:
    """The run's reporter while one recording is extracted: its chunk bar is the recording's own, labelled
    with its session key, so a GUI draws it in that recording's row."""

    def __init__(self, reporter: Reporter, key: str):
        self._reporter = reporter
        self._key = key

    def status(self, text: str) -> None:
        self._reporter.status(text)

    def warning(self, text: str) -> None:
        self._reporter.warning(text)

    def progress(self, label: str, total: int | None = None, *, unit: str = ""):
        return self._reporter.progress(self._key if label == EXTRACTION_BAR else label, total, unit=unit)

    def stop_requested(self) -> bool:
        return self._reporter.stop_requested()


def extract(project: Project, *, recordings: Sequence[str] | None = None, settings: Mapping | None = None,
            reporter: Reporter | None = None) -> ExtractRun:
    """Run the Extract step in ``project`` (module docstring): save ``settings``, then extract what the plan
    says, or every recording in ``recordings`` (paths or session keys) whatever its plan.

    A recording whose cluster job is queued or running is left to its job, named or not. Locally, a recording
    that fails is recorded in the run's ``failed`` and the rest are extracted, and the last status line is the
    run's summary. A stop the reporter asks for raises ``StepStopped`` at the next chunk, leaving the recording
    being extracted as it was. On a cluster, each recording's job is submitted and recorded, and the run
    returns; a job ``sbatch`` refuses raises ``SubmitError`` (W16), the jobs submitted before it staying
    submitted.
    """
    with reporting(reporter) as reporter:
        check_not_old(project.folder)
        with step_lock(project, "extract", where=_where()):
            if settings:
                save_extract_settings(project, settings, recordings)
            detection = ensure_head_tail_detection(project)
            search = _search(project)
            for line in search.report().splitlines():
                if line in search.warnings():
                    reporter.warning(line.removeprefix("Warning: "))
                else:
                    reporter.status(line)
            planned = plan(project, recordings=recordings)
            named = {_key(project, name) for name in recordings} if recordings is not None else None
            targets = [p for p in planned if p.plan != "in progress"
                       and (p.key in named if named is not None else p.plan in ("new", "redo"))]
            up_to_date = [p.key for p in planned if p not in targets and p.plan != "in progress"]
            for item in planned:
                if item.plan == "in progress":
                    reporter.status(f"{item.key}: in progress on the cluster, {item.reason}.")
            for item in targets:
                if item.plan == "redo":
                    reporter.status(redo_line(item))
            if not targets:
                reporter.status(extraction_summary(0, 0, []))
                return ExtractRun(extracted=[], failed={}, up_to_date=up_to_date, jobs={})
            uuids = _list_sessions(project, planned)
            computing = _computing(project)
            if computing.cluster_type == "slurm":
                return _submit(project, targets, detection, computing, uuids, up_to_date, reporter)
            return _extract_here(project, targets, detection, computing, up_to_date, reporter)


def _extract_here(project, targets, detection, computing, up_to_date, reporter) -> ExtractRun:
    cores = computing.cores or available_cores()
    options = _options(project, targets, detection, cores)
    extracted, failed = [], {}
    for item in targets:
        try:
            with reporting(_OneRecording(reporter, item.key)):
                _extract_recording(project, str(item.recording), options[item.key])
        except StepStopped:
            raise
        except Exception as error:  # noqa: BLE001 -- recorded, and the rest are extracted
            _failed(reporter, item, error, failed)
        else:
            extracted.append(item.key)
    reporter.status(extraction_summary(len(targets), 0, [(str(p.recording), failed[p.key])
                                                         for p in targets if p.key in failed]))
    return ExtractRun(extracted=extracted, failed=failed, up_to_date=up_to_date, jobs={})


def _failed(reporter, item, error, failed):
    reporter.status(f"Unexpected error: {error}")
    reporter.status(f"could not extract {item.recording}")
    failed[item.key] = error


@filter_warnings
def _extract_recording(project, recording, opts):
    """``extract_recording`` with the warning filters a project's extraction runs under."""
    return extract_recording(project, recording, opts)


def job_hours(recording: str | os.PathLike) -> int:
    """The time a cluster job asks for to extract ``recording``, in whole hours: three times the recording's
    length, rounded up, at least one. A recording whose length cannot be read gets the one hour, and its job
    stops early on the problem extraction finds."""
    try:
        seconds = recording_seconds(recording)
    except Exception:  # noqa: BLE001 -- the job reports what is wrong with the recording
        seconds = 0
    return max(1, math.ceil(3 * seconds / 3600))


#: A Kinect v2 depth frame's size in a ``.dat``, and the frame rate it is recorded at.
_KINECT_FRAME_BYTES = 512 * 424 * 2
_KINECT_FPS = 30


def recording_seconds(recording: str | os.PathLike) -> float:
    """About how long ``recording`` is, in seconds, read cheaply: a ``.dat`` from its size, an ``.avi`` or
    ``.mkv`` from its frame count or duration, a ``.tar.gz`` from its size as three times compressed."""
    path = str(recording)
    if path.endswith((".tar.gz", ".tgz")):
        return 3 * os.stat(path).st_size / _KINECT_FRAME_BYTES / _KINECT_FPS
    if path.endswith(".dat"):
        return os.stat(path).st_size / _KINECT_FRAME_BYTES / _KINECT_FPS
    import av

    with av.open(path) as container:
        if container.duration:
            return container.duration / 1_000_000
        stream = container.streams.video[0]
        return (stream.frames or 0) / float(stream.base_rate or _KINECT_FPS)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_submission(path: Path, record: Mapping) -> None:
    """Write a submission record whole, one ``name: value`` line each, through a temporary name, so a reader never
    sees half of one."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(dump_block_yaml(dict(record)))
    os.replace(partial, path)


def _refused(key: str, partition: str, answer: str) -> SubmitError:
    """W16: sbatch refused ``key``'s job."""
    check = (f"check that the partition {partition} exists and that this computer can submit jobs "
             "(`sbatch --version`), then")
    return SubmitError(f"The cluster refused the job for {key}. sbatch said: {answer}", answer=answer,
                       fix=f"{check} press Extract again.", command_line_fix=f"{check} run `moseq2 extract` again.")


def _submit(project, targets, detection, computing, uuids, up_to_date, reporter) -> ExtractRun:
    """Submit one job per target and record it beside the recording's outputs (module docstring, step 5)."""
    options = _options(project, targets, detection, JOB_CPUS)
    folder = str(Path(project.folder).absolute())
    jobs = {}
    everything = {p.key for p in targets} | set(load(project).sessions)
    submitted = _now()
    for item in targets:
        session_folder = session_outputs(project, item.key, keys=everything).folder
        session_folder.mkdir(parents=True, exist_ok=True)
        # what an earlier job of this recording left would otherwise speak for the new one
        for earlier in (PROGRESS_FILE, SUBMITTED_FILE):
            (session_folder / earlier).unlink(missing_ok=True)
        job_file = options[item.key].to_yaml(job_settings_file(session_folder, "extract"), job={
            "project": folder, "session": item.key, "uuid": uuids[item.key], "recording": item.key,
            "cpus": JOB_CPUS})
        wall_time = computing.wall_time
        if wall_time == "auto":
            wall_time = f"{job_hours(item.recording)}:00:00"
        command = job_command("run-job", job_file) + "; "
        log = session_folder / JOB_LOG
        resources = dict(memory=computing.memory, partition=computing.partition, wall_time=wall_time,
                         prefix=computing.prefix)
        try:
            job_id = submit(command, cpus=JOB_CPUS, name=JOB_NAME, log=log, **resources)
        except SubmitError as error:
            raise _refused(item.key, computing.partition, error.answer) from error
        _write_submission(session_folder / SUBMITTED_FILE, {
            "job_id": job_id, "submitted": submitted,
            "sbatch": sbatch_command(command, ncpus=JOB_CPUS, name=JOB_NAME, log=log, parsable=True, **resources),
            "settings_file": Path(job_file).name, "log": JOB_LOG})
        jobs[item.key] = job_file
        reporter.status(f"{item.key}: submitted as job {job_id}.")
    return ExtractRun(extracted=[], failed={}, up_to_date=up_to_date, jobs=jobs)


# ------------------------------------------------------------------------------- jobs on a cluster


@dataclass(frozen=True)
class RecordingRun:
    """Where one recording's cluster job stands (``cluster_runs``)."""

    key: str
    job_id: str
    #: ``queued``, ``running``, ``done``, ``failed`` or ``cancelled``.
    state: str
    #: Chunks extracted, and the recording's chunks, as its progress file last said; None before it says.
    done: int | None = None
    total: int | None = None
    #: Why it failed: what happened and what to do, as an expected error.
    problem: ExpectedError | None = None
    #: When the progress file behind ``done`` was written.
    updated: str | None = None
    #: When the job was submitted, the same for every job one run submitted.
    submitted: str | None = None

    @property
    def live(self) -> bool:
        """Whether the job is queued or running."""
        return self.state in ("queued", "running")


def _submissions(project: Project) -> dict:
    """Each listed session's folder and submission record, by session key, for those that have one."""
    if not Path(project.bookkeeping_file).is_file():
        return {}
    listed = load(project).sessions
    found = {}
    for key in sorted(listed):
        path = session_outputs(project, key, keys=set(listed)).folder / SUBMITTED_FILE
        try:
            record = read_yaml(str(path)) if path.is_file() else None
        except Exception:  # noqa: BLE001 -- a record half written by hand is no record
            record = None
        if isinstance(record, Mapping) and record.get("job_id") is not None:
            found[key] = (path.parent, dict(record))
    return found


def _when(text) -> datetime | None:
    try:
        when = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo is not None else when.astimezone()


def _seconds_since(text) -> float | None:
    when = _when(text)
    return None if when is None else (datetime.now().astimezone() - when).total_seconds()


def _sbatch_value(record: Mapping, flag: str) -> str | None:
    """The value ``flag`` (``--mem``, ``-t``) has in the record's ``sbatch`` line."""
    try:
        words = shlex.split(str(record.get("sbatch") or ""))
    except ValueError:
        return None
    words = words[:words.index("--wrap")] if "--wrap" in words else words
    for number, word in enumerate(words):
        if word.startswith(f"{flag}="):
            return word.split("=", 1)[1]
        if word == flag and number + 1 < len(words):
            return words[number + 1]
    return None


def _hours_text(value: float) -> str:
    return f"{value:g} hour" + ("" if value == 1 else "s")


def _wall_time(value: float) -> str:
    seconds = round(value * 3600)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _job_problem(folder: Path, record: Mapping, job: JobState | None) -> ExpectedError:
    """Why a job failed, from SLURM's state for it: out of memory (W12), out of time (W13), anything else (W14)."""
    if job is not None and job.slurm_state == "OUT_OF_MEMORY":
        try:
            memory = gigabytes(_sbatch_value(record, "--mem") or "")
        except ValueError:
            memory = None
        if memory:
            more = memory + 8
            return ExpectedError(f"The job ran out of its {memory:g} GB of memory.",
                                 fix=f"raise Memory per job to {more:g} GB and press Extract again.",
                                 command_line_fix=f"run `moseq2 extract --memory {more:g}GB`.")
    if job is not None and job.slurm_state == "TIMEOUT":
        try:
            allowed = hours(_sbatch_value(record, "-t") or "")
        except ValueError:
            allowed = None
        if allowed:
            return ExpectedError(f"The job ran out of its {_hours_text(allowed)}.",
                                 fix=f"raise Time per job to {_hours_text(2 * allowed)} and press Extract again.",
                                 command_line_fix=f"run `moseq2 extract --wall-time {_wall_time(2 * allowed)}`.")
    log = folder / str(record.get("log") or JOB_LOG)
    fix = f"read the job's log, {log}, then press Extract again (`moseq2 extract` on the command line)."
    if job is not None and job.slurm_state:
        reason = f" ({job.reason})" if job.reason else ""
        return ExpectedError(f"The job stopped with SLURM state {job.slurm_state}{reason}.", fix=fix)
    return ExpectedError("The job left the queue without finishing, and the cluster keeps no record of why.",
                         fix=fix)


def _own_problem(progress: Mapping) -> ExpectedError:
    """The problem a job stopped on, as its progress file names it."""
    return ExpectedError(str(progress["message"]), fix=progress.get("fix") or None)


def _counts(progress: Mapping | None, before: RecordingRun | None, job_id: str) -> tuple:
    """(done, total, updated) from ``progress``, unless the run seen before, of the same job, was read from a newer
    file: a progress file read through a cache may be older than the last one seen, or missing."""
    same = before is not None and before.job_id == job_id
    if progress is None:
        return (before.done, before.total, before.updated) if same else (None, None, None)
    if same and before.updated:
        newer, older = _when(before.updated), _when(progress.get("updated"))
        if newer is not None and (older is None or older < newer):
            return before.done, before.total, before.updated
    return progress.get("done"), progress.get("total"), progress.get("updated")


def cluster_runs(project: Project, *, previous: Mapping[str, RecordingRun] | None = None) -> dict:
    """Where each of ``project``'s submitted recordings stands, by session key (``RecordingRun``).

    Read from each recording's submission record and progress file, and from one ``squeue`` and one ``sacct``
    (``moseq2_common.slurm.job_states``) for the jobs whose progress file has not said how they ended. A
    progress file that says the job finished, or failed on a problem it names, decides; else SLURM's state does,
    out of memory and out of time becoming problems that say what to raise (W12, W13), any other failure one that
    names SLURM's state and the job's log (W14). Where SLURM no longer knows the job (no accounting), a job whose
    progress file has not been rewritten for ``STALLED_AFTER`` seconds has stopped (W14), and one still writing
    runs. ``previous``, the runs a call before returned, keeps every count from moving backwards when a
    progress file reads older than one seen before, or not at all; a run that had ended stays as it ended.
    No SLURM command runs when no recording has a submission record.
    """
    previous = previous or {}
    submissions = _submissions(project)
    progress, ask = {}, []
    for key, (folder, record) in submissions.items():
        read = read_progress(folder / PROGRESS_FILE)
        progress[key] = read if read is not None and str(read.get("job")) == str(record["job_id"]) else None
        before = previous.get(key)
        settled = before is not None and before.job_id == str(record["job_id"]) and not before.live
        ended = progress[key] is not None and (progress[key]["state"] == "finished"
                                               or (progress[key]["state"] == "failed" and progress[key].get("message")))
        if not (settled or ended or record.get("cancelled")):
            ask.append(str(record["job_id"]))
    states = job_states(ask) if ask else {}
    runs = {}
    for key, (folder, record) in submissions.items():
        job_id = str(record["job_id"])
        before = previous.get(key)
        if before is not None and before.job_id == job_id and not before.live:
            runs[key] = before
            continue
        read = progress[key]
        done, total, updated = _counts(read, before, job_id)
        stamp = dict(key=key, job_id=job_id, done=done, total=total, updated=updated,
                     submitted=record.get("submitted"))
        job = states.get(job_id)
        if read is not None and read["state"] == "finished":
            runs[key] = RecordingRun(state="done", **{**stamp, "done": total if total is not None else done})
        elif read is not None and read["state"] == "failed" and read.get("message"):
            runs[key] = RecordingRun(state="failed", problem=_own_problem(read), **stamp)
        elif record.get("cancelled"):
            runs[key] = RecordingRun(state="cancelled", **stamp)
        elif job is None or job.state == "unknown":
            since = _seconds_since(updated or record.get("submitted"))
            if (read is not None and read["state"] == "failed") or (since is not None and since > STALLED_AFTER):
                runs[key] = RecordingRun(state="failed", problem=_job_problem(folder, record, None), **stamp)
            else:
                runs[key] = RecordingRun(state="running", **stamp)
        elif job.state == "failed" or (read is not None and read["state"] == "failed"):
            runs[key] = RecordingRun(state="failed", problem=_job_problem(folder, record, job), **stamp)
        elif job.state == "done":
            runs[key] = RecordingRun(state="done", **{**stamp, "done": total if total is not None else done})
        else:
            runs[key] = RecordingRun(state=job.state, **stamp)
    return runs


def _how_it_stands(run: RecordingRun) -> str:
    """A queued or running job, as the plan's reason says it."""
    if run.state == "queued":
        return f"waiting in the queue as job {run.job_id}"
    if run.total:
        return f"running as job {run.job_id} ({round(100 * (run.done or 0) / run.total)}% done)"
    return f"running as job {run.job_id}"


def stop(project: Project, keys: Sequence[str] | None = None) -> list[str]:
    """Cancel ``project``'s cluster jobs that are queued or running (of the recordings ``keys`` only, when given)
    with one ``scancel``; the ids of the jobs cancelled. Each cancelled recording's submission record notes when
    (``cancelled``), so the next run extracts it again."""
    runs = cluster_runs(project)
    live = [run for key, run in sorted(runs.items()) if run.live and (keys is None or key in keys)]
    ids = sorted((run.job_id for run in live), key=lambda job_id: (len(job_id), job_id))
    if not ids:
        return []
    try:
        cancel(ids)
    except SubmitError as error:
        check = "check that this computer can submit jobs (`sbatch --version`), then"
        raise SubmitError(f"The jobs could not be cancelled: {error.answer}", answer=error.answer,
                          fix=f"{check} press Stop again.",
                          command_line_fix=f"{check} run `moseq2 extract --stop` again.") from error
    when = _now()
    submissions = _submissions(project)
    for run in live:
        folder, record = submissions[run.key]
        _write_submission(folder / SUBMITTED_FILE, {**record, "cancelled": when})
    return ids


def state_line(run: RecordingRun) -> str:
    """What the command line says when a recording's job reaches ``run``'s state."""
    if run.state == "queued":
        return f"{run.key}: waiting in the queue as job {run.job_id}."
    if run.state == "running":
        return f"{run.key}: running as job {run.job_id}."
    if run.state == "done":
        return f"{run.key}: extracted."
    if run.state == "cancelled":
        return f"{run.key}: its job was cancelled."
    return f"{run.key}: {run.problem}"


#: What a run says of a recording whose job was cancelled.
CANCELLED = "Its job was cancelled before it finished."


def follow(project: Project, keys: Sequence[str], *, reporter: Reporter | None = None,
           every: float = 3.0) -> ExtractRun:
    """Follow the cluster jobs of the recordings ``keys`` until every one has ended, checking every ``every``
    seconds (``cluster_runs``), with a status line each time one changes state and the run's summary last.

    What the jobs did, as a run: those done in ``extracted``, those that failed or were cancelled in ``failed``
    with their problem. A recording with no submission record is left out.
    """
    with reporting(reporter) as reporter:
        runs: dict = {}
        said: dict = {}
        while True:
            runs = cluster_runs(project, previous=runs)
            mine = {key: runs[key] for key in keys if key in runs}
            for key, run in mine.items():
                line = state_line(run)
                if said.get(key) != line:
                    reporter.status(line)
                    if run.problem is not None and (run.problem.command_line_fix or run.problem.fix):
                        reporter.status(f"  To fix it: {run.problem.command_line_fix or run.problem.fix}")
                    said[key] = line
            if not any(run.live for run in mine.values()):
                break
            checkpoint()
            time.sleep(every)
        failed = {key: run.problem or ExpectedError(CANCELLED) for key, run in mine.items()
                  if run.state in ("failed", "cancelled")}
        reporter.status(extraction_summary(len(mine), 0, [(key, failed[key]) for key in sorted(failed)]))
        return ExtractRun(extracted=[key for key, run in mine.items() if run.state == "done"], failed=failed,
                          up_to_date=[], jobs={})


def run_job(job_file: str | os.PathLike, *, reporter: Reporter | None = None) -> steps.ExtractResult:
    """Extract the recording a cluster job's settings file names, as the job does (``moseq2 run-job``).

    The file's ``extract`` section and ``camera_type`` are the recording's settings; its ``job:`` block names
    the project, the recording (relative to the project) and the cores the job asked for, which reading and
    compressing use. The outputs go into the recording's session folder with the uuid
    ``moseq2-project.yaml`` lists for it. What it does is reported to ``reporter`` (the job's log) and, through
    a ``ProgressFileReporter``, into ``PROGRESS_FILE`` beside the settings file, its chunks counted and how it
    ended said, an expected error's message and fix included.
    """
    with reporting(reporter) as reporter:
        progress = ProgressFileReporter(Path(job_file).parent / PROGRESS_FILE, bar=EXTRACTION_BAR, also=reporter)
        with progress, reporting(progress):
            return _run_job(job_file)


def _run_job(job_file):
    data = read_config(job_file)
    check_top_level(job_file, data)
    job = data.get("job") or {}
    project = Project.at(job["project"])
    cpus = int(job.get("cpus") or JOB_CPUS)
    opts = ExtractOptions.resolve(config=read_settings(job_file, ExtractOptions, stage_classes=EXTRACT_STAGE),
                                  explicit={"threads": cpus, "compress_threads": cpus})
    return extract_recording(project, str(Path(job["project"]) / job["recording"]), opts)


# ------------------------------------------------------------------------------- one recording


def find_roi(project: Project, input_file: str, opts: RoiOptions, *,
             output_dir: str | None = None, reporter: Reporter | None = None) -> steps.FindRoiResult:
    """Find the arena and background of ``input_file`` in ``project``; write the background and arena images.

    They go into ``output_dir`` when given, else into the recording's session folder in the project's
    ``aggregate_results/``, where ``extract`` writes its outputs. The record lists the recording, then the
    project's ``config.yaml`` and ``session_config.yaml``.
    """
    record = start_record("find-roi", opts, [input_file, *settings_files(project)], project=project)
    if output_dir is None:
        output_dir = str(session_outputs(project, session_key(project, input_file)).folder)
    return steps.find_roi(input_file, output_dir, opts, record=record, reporter=reporter)


def _session_uuid(project, input_file):
    """The uuid moseq2-project.yaml gave the recording's session; None before it lists the session."""
    listed = load(project).sessions.get(session_key(project, input_file))
    return listed.uuid if listed is not None else None


def extract_recording(project: Project, input_file: str, opts: ExtractOptions, *,
                      output_dir: str | None = None, reporter: Reporter | None = None) -> steps.ExtractResult:
    """Extract ``input_file`` in ``project`` with ``opts``, as given.

    Its outputs go into ``output_dir`` when given (relative to the recording's folder), else into its
    session's folder in the project's ``aggregate_results/`` (``session_outputs``). A session
    ``moseq2-project.yaml`` lists keeps its uuid. The record lists the project's ``config.yaml`` and
    ``session_config.yaml``, then what the extraction reads.
    """
    record = start_record("extract", opts, settings_files(project), project=project)
    if output_dir is None:
        output_dir = str(session_outputs(project, session_key(project, input_file)).folder)
    return steps.extract(input_file, output_dir, opts, uuid=_session_uuid(project, input_file), record=record,
                         reporter=reporter)


# ------------------------------------------------------------------------------- the config file


def generate_config(output_file: str, camera_type: str = "auto", *, overwrite: bool) -> Path:
    """Write the project config file ``output_file`` (``moseq2_extract.steps.generate_config``).

    A folder holding a 1.5.0 project stops with ``OldProjectError`` before anything is written.
    """
    check_not_old(dirname(abspath(output_file)))
    return steps.generate_config(output_file, camera_type, overwrite=overwrite)


def download_flip_file(config_file: str, output_dir: str, selection: int | None = None) -> Path:
    """Download a flip classifier into ``output_dir`` and name it in ``config_file``'s ``extract`` section.

    ``selection`` is the classifier's number in ``moseq2_extract.steps.FLIP_CLASSIFIERS``; without one the
    file layer raises ``ChoiceNeeded`` before anything is downloaded. A config file without sections, or a
    folder holding a 1.5.0 project, stops before anything is downloaded. A config file that cannot be
    updated is reported, and the downloaded classifier kept.
    """
    check_not_old(dirname(abspath(config_file)))
    if exists(config_file):
        check_top_level(config_file, read_config(config_file))
    downloaded = steps.download_flip_file(output_dir, selection)
    # the config file names the classifier joined to output_dir as given (the notebook passes './')
    _name_flip_classifier(config_file, join(output_dir, downloaded.name))
    return downloaded


@filter_warnings
def _name_flip_classifier(config_file, flip_file):
    """Name ``flip_file`` as the flip classifier in ``config_file``'s ``extract`` section."""
    try:
        config_data = read_yaml(config_file)
        config_data.setdefault("extract", {})["flip_classifier"] = flip_file

        write_config(config_data, config_file)
    except Exception as e:
        print("Could not update configuration file flip classifier path")
        print("Unexpected error:", e)
