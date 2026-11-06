"""The extraction pipeline steps, one function each.

``find-roi``, ``extract``, ``batch-extract`` and ``aggregate-results`` on the
command line (``cli.py``) and the Extract notebook's functions (``gui.py``)
resolve their settings into one options object and call these. A step needs no
``config.yaml``: from a script, pass the paths and an options object, such as
``extract("session_1/depth.dat", "proc", ExtractOptions(num_frames=900))``.
``batch_extract`` is the exception, since each recording's settings come from
``config.yaml``'s ``extract`` section and its entry in ``session_config.yaml``.

Each step hands its work function (``get_roi_wrapper``, ``extract_wrapper``,
``aggregate_extract_results_wrapper`` in ``moseq2_extract.helpers.wrappers``)
the options as a dict. Each output carries the record of the run that wrote it
(``moseq2_common.records``): extraction's is the ``/provenance`` attribute of the
session's ``results.h5``, holding the options after detection; ``find-roi``'s
and a SLURM batch's script's are a ``.yaml`` beside them (``sidecar_path``).

Inside a project, a recording's outputs go into its session's folder in
``aggregate_results/`` (``moseq2_extract.recordings.session_outputs``) and keep
the uuid ``moseq2-project.yaml`` gave the session; outside one, ``extract``
writes into ``proc/`` beside the recording. ``aggregate_results`` lists the
extracted sessions in ``moseq2-project.yaml``.

Before any work, ``find-roi`` and ``extract`` stop on a recording that does not
exist, is empty or cannot be read (``moseq2_common.records.check_inputs``), and
extraction on a recording whose frames and timestamps disagree in number.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os.path import abspath
from pathlib import Path
from uuid import uuid4

from moseq2_common.bookkeeping import SESSION_FILE, Session, load, relative_path, update
from moseq2_common.records import check_inputs, start_record, write_record
from moseq2_common.project import Project, find_project as find_project_of, session_key
from moseq2_extract.helpers.extract import run_local_extract, run_slurm_extract
from moseq2_extract.helpers.wrappers import aggregate_extract_results_wrapper, extract_wrapper, get_roi_wrapper
from moseq2_extract.options import AggregateOptions, BatchExtractOptions, ExtractOptions, RoiOptions
from moseq2_extract.recordings import (extracted_in, find_recordings, project_keys, session_outputs,
                                       standalone_outputs)

#: ExtractOptions fields extract_wrapper takes as arguments rather than in its dict.
_WRAPPER_ARGUMENTS = ("num_frames", "skip_completed")


@dataclass(frozen=True)
class FindRoiResult:
    #: The arena masks written (``roi_XX.tiff``), one per ``bg_roi_index`` found.
    roi_files: list


@dataclass(frozen=True)
class ExtractResult:
    #: The extraction's ``results.h5``; None when the recording was skipped as already extracted.
    h5_file: Path | None


@dataclass(frozen=True)
class BatchExtractResult:
    #: The recordings the search found, before any were picked.
    found: list
    #: The recordings extracted locally, or skipped there as already extracted.
    extracted: list
    #: (recording, exception) for each local extraction that failed, in the order tried.
    failures: list
    #: The script of sbatch commands, when the cluster type is slurm.
    script: Path | None


@dataclass(frozen=True)
class AggregateResult:
    #: The session keys listed in the bookkeeping file by this run.
    sessions: list


#: What to do about a recording that is missing, empty or unreadable.
RECORDING_ADVICE = ("Check that it is the recording you meant and that it was copied completely, or copy it again "
                    "from where it was recorded.")


def _settings(opts, leave_out=()):
    """The dict a wrapper takes: the options, but ``leave_out``."""
    return {name: value for name, value in opts.as_dict().items() if name not in leave_out}


def find_roi(input_file: str, output_dir: str | None, opts: RoiOptions) -> FindRoiResult:
    """Find the arena and background of ``input_file``; write the background and arena images."""
    check_inputs("find-roi", {input_file: RECORDING_ADVICE})
    record = start_record("find-roi", opts, [input_file])
    settings = _settings(opts)
    get_roi_wrapper(input_file, settings, output_dir)
    # the wrapper records in settings the folder it wrote to
    folder = settings.get("output_dir")
    indices = opts.bg_roi_index if isinstance(opts.bg_roi_index, (list, tuple)) else [opts.bg_roi_index]
    written = [Path(folder) / f"roi_{idx:02d}.tiff" for idx in indices] if folder else []
    result = FindRoiResult(roi_files=[path for path in written if path.is_file()])
    record.ran_with({"bg_roi_depth_range": settings["bg_roi_depth_range"]})
    finished = record.finish({"files": sorted(str(p) for p in Path(folder).glob("*.tiff")) if folder else []})
    for roi_file in result.roi_files:
        write_record(roi_file, finished)
    return result


def _project_inputs(project):
    """The project's settings files, which every step in it reads."""
    if project is None:
        return []
    return [path for path in (project.config_file, project.session_config_file) if Path(path).is_file()]


def _output_folder(input_file, output_dir, project, keys=None):
    """Where ``input_file``'s outputs go: ``output_dir`` when given (a relative one below the recording's
    folder), its session's folder in a project, else ``proc/`` beside it."""
    if output_dir is not None:
        return Path(abspath(input_file)).parent / output_dir
    if project is None:
        return standalone_outputs(input_file).folder
    return session_outputs(project, session_key(project, input_file), keys=keys).folder


def _session_uuid(input_file, project):
    """The uuid moseq2-project.yaml gave the recording's session; None before it lists the session."""
    if project is None:
        return None
    listed = load(project).sessions.get(session_key(project, input_file))
    return listed.uuid if listed is not None else None


def extract(input_file: str, output_dir: str | None, opts: ExtractOptions, *,
            project: Project | None = None) -> ExtractResult:
    """Extract ``input_file``.

    Its outputs go into ``output_dir`` when given (relative to the recording's folder), into its session's folder in ``project``'s
    ``aggregate_results/`` (``session_outputs``), or outside a project into ``proc/`` beside it.
    In a project, a session ``moseq2-project.yaml`` lists keeps its uuid.
    """
    check_inputs("extract", {input_file: RECORDING_ADVICE})
    record = start_record("extract", opts, _project_inputs(project), project=project)
    settings = _settings(opts, leave_out=_WRAPPER_ARGUMENTS)
    folder = _output_folder(input_file, output_dir, project)
    written = extract_wrapper(input_file, str(folder), settings, num_frames=opts.num_frames,
                              skip=opts.skip_completed, uuid=_session_uuid(input_file, project), record=record)
    if written is None:
        return ExtractResult(h5_file=None)
    return ExtractResult(h5_file=Path(written) / "results.h5")


def batch_extract(input_dir: str, opts: BatchExtractOptions, *, config_file: str,
                  output_dir: str | None = None, explicit: Mapping | None = None,
                  select: Callable[[Sequence[str]], Sequence[str]] | None = None,
                  project: Project | None = None) -> BatchExtractResult:
    """Find the recordings under ``input_dir`` and extract each, here or as SLURM jobs.

    The search looks for ``opts.extensions`` down to ``opts.max_depth`` folders
    below ``input_dir`` and, unless ``opts.skip_checks`` is set, leaves out each
    recording whose ``results.h5`` exists in its session folder. The command line
    and the notebook both list recordings this way.

    Each recording is extracted with ``config.yaml``'s ``extract`` settings, the
    arena tool's settings from its entry in ``session_config.yaml``, and
    ``explicit`` (options typed on the command line) over both
    (``session_config.recordings_options``).

    Args:
    input_dir: the folder to search.
    opts: the search, the cluster type and the SLURM job settings; ``num_frames`` and
        ``skip_completed`` apply to every recording.
    config_file: the project's ``config.yaml``.
    output_dir: the folder holding the session folders; None for the project's ``aggregate_results/``.
    explicit: options typed on the command line, applied over each recording's settings.
    select: picks the recordings to extract from those found (the notebook's prompt).
    project: the project the recordings are sessions of; with SLURM, each job's settings file names it.

    With ``opts.cluster_type`` slurm, each recording ``opts.skip_completed`` does not leave out gets a job: its
    session is given its uuid in ``moseq2-project.yaml`` and its settings are written to a job settings file
    in its session folder (``moseq2_common.slurm.job_settings_file``), before ``opts.extract_out_script``
    is written and, with ``opts.run_cmd``, submitted.
    """
    project = project if project is not None else find_project_of(config_file)
    search = find_recordings(input_dir, extensions=opts.extensions, max_depth=opts.max_depth,
                             extracted=None if opts.skip_checks else extracted_in(project, output_dir))
    message = search.report()
    if message:
        print(message)
    if not search.recordings:
        return BatchExtractResult(found=[], extracted=[], failures=[], script=None)
    to_extract = list(select(search.recordings) if select is not None else search.recordings)
    keys = set(project_keys(project)) | {session_key(project, r) for r in to_extract}
    folders = {recording: str(session_outputs(project, session_key(project, recording), keys=keys,
                                              sessions_dir=output_dir).folder)
               for recording in to_extract}

    if opts.cluster_type == "local":
        failures = run_local_extract(to_extract, config_file, opts.skip_completed, num_frames=opts.num_frames,
                                     output_dirs=folders, cli_options=dict(explicit) if explicit else None)
        failed = {path for path, _ in failures}
        return BatchExtractResult(found=list(search.recordings),
                                  extracted=[Path(p) for p in to_extract if p not in failed],
                                  failures=failures, script=None)

    record = start_record("batch-extract", opts, [config_file, *to_extract], project=project)
    if opts.skip_completed:
        to_extract = [r for r in to_extract if not (Path(folders[r]) / SESSION_FILE).is_file()]
    batch = {**opts.as_dict(), "config_file": abspath(config_file), "output_dir": output_dir}
    run_slurm_extract(input_dir, to_extract, batch, num_frames=opts.num_frames,
                      cli_options=dict(explicit) if explicit else None, output_dirs=folders,
                      jobs=_submit(project, to_extract, folders))
    # run_slurm_extract records where it wrote the script
    script = Path(batch["extract_out_script"])
    if script.is_file():
        write_record(script, record.finish({"files": {"script": str(script), "session_folders": sorted(folders.values())}}))
    return BatchExtractResult(found=list(search.recordings), extracted=[], failures=[], script=script)


def _submit(project, recordings, folders):
    """Give each recording's session its uuid in ``moseq2-project.yaml`` before its job is submitted; return
    each recording's ``job:`` block (the project, the session and that uuid).

    A session the file lists keeps its uuid; a new one gets a new uuid. Its extraction, run by the job, takes
    the uuid from the file, and ``aggregate-results`` lists the session once its ``results.h5`` exists.
    """
    folder = str(Path(project.folder).absolute())
    listed = load(project).sessions
    sessions = {}
    for recording in recordings:
        key = session_key(project, recording)
        uuid = listed[key].uuid if key in listed and listed[key].uuid else str(uuid4())
        sessions[recording] = Session(key=key, recording=key, folder=relative_path(project, folders[recording]),
                                      uuid=uuid)

    def add(bookkeeping):
        for session in sessions.values():
            bookkeeping.sessions[session.key] = session

    if sessions:
        update(project, add)
    return {recording: {"project": folder, "session": s.key, "uuid": s.uuid} for recording, s in sessions.items()}


def aggregate_results(project: Project, opts: AggregateOptions, *, input_dir: str | None = None,
                      output_dir: str | None = None) -> AggregateResult:
    """List every extracted recording of ``project`` as a session in its bookkeeping file.

    Each recording found under ``input_dir`` (the project folder by default) whose ``results.h5``
    exists in its session folder (in ``output_dir``, the project's ``aggregate_results/`` by default)
    becomes a session, keyed by the recording, with the uuid its extraction carries. Nothing is copied.
    Sessions listed before are kept, and nothing else in the file changes.
    """
    found = aggregate_extract_results_wrapper(str(input_dir or project.folder), output_dir, opts.mouse_threshold,
                                              project=project)
    sessions = [Session(key=key, recording=key, folder=relative_path(project, folder), uuid=uuid)
                for key, folder, uuid in found]

    def add(bookkeeping):
        for session in sessions:
            bookkeeping.sessions[session.key] = session

    update(project, add)
    print(f"{len(sessions)} session(s) listed in {project.bookkeeping_file}; it lists "
          f"{len(load(project).sessions)} in all.")
    return AggregateResult(sessions=[session.key for session in sessions])
