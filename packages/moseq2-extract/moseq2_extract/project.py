"""The extraction steps in a project: the project layer.

Each function here is one step run in a project. It takes the project
(``moseq2_common.project.Project``) first, then what a person can type for the
step and the step's resolved options object, and returns the step's result. It
resolves what the project decides for the step: where the recording's outputs go
(its session's folder in ``aggregate_results/``, ``session_outputs``), the uuid
``moseq2-project.yaml`` gave the session, and the settings files the step reads
(``settings_files``). It starts the run's record with the project, so the record
names the project folder and describes each input relative to it, and calls the
file layer (``moseq2_extract.steps``) with explicit paths and that record. The
file layer fills the record in and writes it with the outputs.

Batch extraction and aggregation run only in a project, so they are here whole:
``batch_extract`` finds the project's recordings, resolves each one's settings
from ``config.yaml`` and its entry in ``session_config.yaml``
(``session_config.recordings_options``) and extracts each through ``extract``
here (``run_local_extract``), or writes a SLURM job per recording whose settings
file names the project; a SLURM batch's script carries its record in a ``.yaml``
beside it (``sidecar_path``). ``aggregate_results`` lists the extracted sessions
in ``moseq2-project.yaml``.

``generate_config`` and ``download_flip_file`` take the project's config file
rather than a ``Project``, since writing it is what makes a folder a project and
it may have any name. Both stop on a MoSeq2 1.5.0 project
(``moseq2_common.project.check_not_old``) before writing anything.

The file layer is reached through its module (``steps.extract``), never by a
name imported from it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from os.path import abspath, dirname, exists, join
from pathlib import Path
from uuid import uuid4

from tqdm.auto import tqdm

from moseq2_common.bookkeeping import SESSION_FILE, Session, load, relative_path, update
from moseq2_common.options import check_top_level, read_config
from moseq2_common.project import Project, check_not_old, find_project, session_key, settings_files
from moseq2_common.records import InputError, start_record, write_record
from moseq2_common.yaml import read_yaml, write_config
from moseq2_extract import steps
from moseq2_extract.helpers.extract import extraction_summary, run_slurm_extract
from moseq2_extract.helpers.wrappers import aggregate_extract_results_wrapper
from moseq2_extract.options import AggregateOptions, BatchExtractOptions, ExtractOptions, RoiOptions
from moseq2_extract.recordings import SESSIONS_FOLDER, extracted_in, find_recordings, project_keys, session_outputs
from moseq2_extract.session_config import recordings_options
from moseq2_extract.util import filter_warnings

__all__ = ["AggregateResult", "BatchExtractResult", "aggregate_results", "batch_extract", "download_flip_file",
           "extract", "find_roi", "generate_config", "run_local_extract"]


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


def find_roi(project: Project, input_file: str, opts: RoiOptions, *,
             output_dir: str | None = None) -> steps.FindRoiResult:
    """Find the arena and background of ``input_file`` in ``project``; write the background and arena images.

    They go into ``output_dir`` when given, else into the recording's session folder in the project's
    ``aggregate_results/``, where ``extract`` writes its outputs. The record lists the recording, then the
    project's ``config.yaml`` and ``session_config.yaml``.
    """
    record = start_record("find-roi", opts, [input_file, *settings_files(project)], project=project)
    if output_dir is None:
        output_dir = str(session_outputs(project, session_key(project, input_file)).folder)
    return steps.find_roi(input_file, output_dir, opts, record=record)


def _session_uuid(project, input_file):
    """The uuid moseq2-project.yaml gave the recording's session; None before it lists the session."""
    listed = load(project).sessions.get(session_key(project, input_file))
    return listed.uuid if listed is not None else None


def extract(project: Project, input_file: str, opts: ExtractOptions, *,
            output_dir: str | None = None) -> steps.ExtractResult:
    """Extract ``input_file`` in ``project``.

    Its outputs go into ``output_dir`` when given (relative to the recording's folder), else into its
    session's folder in the project's ``aggregate_results/`` (``session_outputs``). A session
    ``moseq2-project.yaml`` lists keeps its uuid. The record lists the project's ``config.yaml`` and
    ``session_config.yaml``, then what the extraction reads.
    """
    record = start_record("extract", opts, settings_files(project), project=project)
    if output_dir is None:
        output_dir = str(session_outputs(project, session_key(project, input_file)).folder)
    return steps.extract(input_file, output_dir, opts, uuid=_session_uuid(project, input_file), record=record)


def batch_extract(project: Project, input_dir: str, opts: BatchExtractOptions, *, config_file: str,
                  output_dir: str | None = None, explicit: Mapping | None = None,
                  select: Callable[[Sequence[str]], Sequence[str]] | None = None) -> BatchExtractResult:
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
    project: the project the recordings are sessions of; with SLURM, each job's settings file names it.
    input_dir: the folder to search.
    opts: the search, the cluster type and the SLURM job settings; ``num_frames`` and
        ``skip_completed`` apply to every recording.
    config_file: the project's ``config.yaml``.
    output_dir: the folder holding the session folders; None for the project's ``aggregate_results/``.
    explicit: options typed on the command line, applied over each recording's settings.
    select: picks the recordings to extract from those found (the notebook's prompt).

    With ``opts.cluster_type`` slurm, each recording ``opts.skip_completed`` does not leave out gets a job: its
    session is given its uuid in ``moseq2-project.yaml`` and its settings are written to a job settings file
    in its session folder (``moseq2_common.slurm.job_settings_file``), before ``opts.extract_out_script``
    is written and, with ``opts.run_cmd``, submitted.
    """
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


def run_local_extract(
    to_extract,
    config_file,
    skip_extracted=False,
    num_frames=None,
    output_dirs=None,
    cli_options=None,
):
    """
    Extract each recording of ``to_extract`` here, one at a time, and end with a summary of what failed.

    Each recording is extracted with the settings ``recordings_options`` resolves: the arena tool's settings
    from its entry in the project's session_config.yaml, every other setting from the config file, and
    ``cli_options`` over both. It is extracted in the project ``config_file`` belongs to (``extract``), or
    on its own when the file belongs to none (``moseq2_extract.steps.extract``). A recording's error is
    caught and printed so the rest still run.

    Args:
    to_extract (list): list of paths to files to extract
    config_file (str): path to configuration file containing pre-configured extract and ROI
    skip_extracted (bool): Whether to skip already extracted session.
    num_frames (int): number of frames to extract from each session. If None, all frames are extracted.
    output_dirs (dict): each recording's output folder; a recording it does not name goes to its session's folder.
    cli_options (dict): parameters typed on the command line, applied over the config file and each session's config.

    Returns:
    failures (list): (path, exception) for each recording that could not be extracted, in the order tried.
    """
    failures = []
    n_skipped = 0
    for ext in tqdm(to_extract, desc="Extracting Sessions"):
        try:
            result = _extract_recording(ext, (output_dirs or {}).get(ext), config_file, num_frames,
                                        skip_extracted, cli_options)
        except Exception as e:
            print("Unexpected error:", e)
            print("could not extract", ext)
            failures.append((ext, e))
        else:
            if result.h5_file is None:
                n_skipped += 1

    print(extraction_summary(len(to_extract), n_skipped, failures))
    return failures


@filter_warnings
def _extract_recording(recording, output_dir, config_file, num_frames, skip, cli_options):
    """Extract ``recording`` with the options ``recordings_options`` resolves for it from ``config_file``, in
    the project the file belongs to, or on its own outside one."""
    resolved = recordings_options(config_file, [recording], cli_options)
    opts = replace(resolved[recording], num_frames=num_frames, skip_completed=skip)
    project = find_project(config_file)
    if project is None:
        return steps.extract(recording, output_dir, opts)
    return extract(project, recording, opts, output_dir=output_dir)


def aggregate_results(project: Project, opts: AggregateOptions, *, input_dir: str | None = None,
                      output_dir: str | None = None) -> AggregateResult:
    """List every extracted recording of ``project`` as a session in its bookkeeping file.

    Each recording found under ``input_dir`` (the project folder by default) whose ``results.h5``
    exists in its session folder (in ``output_dir``, the project's ``aggregate_results/`` by default)
    becomes a session, keyed by the recording, with the uuid its extraction carries. Nothing is copied.
    Sessions listed before are kept, and nothing else in the file changes. When no recording has a finished
    extraction and the project lists no session yet, it stops before writing anything, saying where it looked.
    """
    folder = str(input_dir or project.folder)
    found = aggregate_extract_results_wrapper(folder, output_dir, opts.mouse_threshold, project=project)
    if not found and not load(project).sessions:
        count = len(find_recordings(folder).recordings)
        sessions_dir = output_dir or Path(project.folder) / SESSIONS_FOLDER
        raise InputError(
            f"aggregate-results found {count} recording{'' if count == 1 else 's'} under {abspath(folder)}, and "
            f"none of them has a finished extraction to list: a recording is listed when its session folder in "
            f"{abspath(sessions_dir)} holds its results.h5 (and its frames' mean depth is above mouse_threshold). "
            f"Extract the recordings first (`moseq2-extract batch-extract {abspath(folder)}`, or the notebook's "
            "extraction cell). If the recordings are in another folder, name it with --input-dir; if extraction "
            "wrote the session folders elsewhere, name that folder with --output-dir."
        )
    sessions = [Session(key=key, recording=key, folder=relative_path(project, folder), uuid=uuid)
                for key, folder, uuid in found]

    def add(bookkeeping):
        for session in sessions:
            bookkeeping.sessions[session.key] = session

    update(project, add)
    print(f"{len(sessions)} session(s) listed in {project.bookkeeping_file}; it lists "
          f"{len(load(project).sessions)} in all.")
    return AggregateResult(sessions=[session.key for session in sessions])


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
