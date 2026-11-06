"""
The notebooks' project cell.

``start`` names the project the notebook works in, the folder holding
``config.yaml``, and prints what it holds (``progress``): how many of its
recordings are extracted, its sessions, the outputs its ``moseq2-project.yaml``
names, and each output older than what it was built from: PC scores computed
before one of their sessions was re-extracted, a model fit before the scores
were computed again, an output computed from a file that no longer exists, and
PCA outputs or a model computed before a session was added to the project. An
output is older when an input its record holds is no longer that file
(``moseq2_common.records.unchanged``: an input the pipeline wrote now carries
the record of another run; any other input differs in size or content), when a
recorded input is gone, or, for PCA outputs and fitted models, when a listed
session ``config.yaml`` does not exclude is not among its recorded inputs. No
file time is read, so a project copied elsewhere shows nothing older. This is a
message only: no step reads it, or a record, to decide what to run. Every later
cell takes the project folder and reads the bookkeeping file when it needs a
path, so nothing is kept between cells. A project made by MoSeq2 1.5.0 or
earlier stops here, before anything else runs.
"""

from __future__ import annotations

import os
from pathlib import Path

from moseq2_common.bookkeeping import load, resolve_path, session_h5
from moseq2_common.project import Project, check_not_old, read_exclude
from moseq2_common.records import read_record, run_of, unchanged, written_on
from moseq2_extract.recordings import DEFAULT_MAX_DEPTH

__all__ = ["progress", "start", "stale_outputs", "summary"]


def summary(project: Project) -> str:
    """What the project's bookkeeping file names, one line per step."""
    bookkeeping = load(project)
    lines = [f"Project: {project.folder}"]
    if not Path(project.config_file).is_file():
        lines.append("  config.yaml: not written yet (the next cell writes it)")
    lines.append(f"  Sessions: {len(bookkeeping.sessions)} aggregated")
    pca = [f"{name} {bookkeeping.pca[name]}" for name in ("components", "scores", "changepoints")
           if bookkeeping.pca.get(name)]
    lines.append(f"  PCA: {', '.join(pca) if pca else 'not run yet'}")
    fitted = bookkeeping.models.get("fitted") or []
    current = bookkeeping.models.get("current")
    if fitted:
        lines.append(f"  Models: {len(fitted)} fitted; current {current}")
    else:
        lines.append("  Models: none fitted yet")
    if bookkeeping.applied_models:
        lines.append(f"  Applied models: {', '.join(bookkeeping.applied_models)}")
    return "\n".join(lines)


def _outputs(project, bookkeeping):
    """Each output the bookkeeping file names, as (the path it stores, the file, whether it is built from all
    the project's sessions: the PCA outputs and the fitted models)."""
    pca = [bookkeeping.pca.get(name) for name in ("components", "scores", "changepoints")]
    stored = [(path, True) for path in [*pca, *(bookkeeping.models.get("fitted") or [])]]
    stored += [(path, False) for path in bookkeeping.applied_models]
    return [(path, resolve_path(project, path), all_sessions) for path, all_sessions in stored if path]


def stale_outputs(project: Project) -> list[str]:
    """A line for each output the bookkeeping file names that is older than its inputs: an input that is no
    longer the file the output's record holds (``unchanged``), an input that no longer exists, and, for a
    PCA output or a fitted model, a listed session ``config.yaml`` does not exclude that is not among its
    inputs (aggregated after it was computed). The settings files are not inputs in this sense; a changed
    setting takes effect when a step is run again. An input the pipeline wrote is compared by the run its record
    names, any other by size and content, never by time."""
    bookkeeping = load(project)
    sessions = {str(session_h5(project, s)): key for key, s in bookkeeping.sessions.items()}
    excluded = set(read_exclude(project))
    included = [h5 for h5, key in sessions.items() if key not in excluded]
    settings = {str(Path(p).absolute()) for p in (project.config_file, project.session_config_file,
                                                  project.bookkeeping_file)}
    lines = []
    for stored, output, all_sessions in _outputs(project, bookkeeping):
        try:
            record = read_record(output)
        except (OSError, ValueError):
            continue
        step = record.get("step", "the step that wrote it")
        read = set()
        for entry in record.get("inputs") or []:
            if not isinstance(entry, dict) or not (isinstance(entry.get("fingerprint"), dict) or entry.get("run")):
                continue
            path = resolve_path(project, entry["relative_path"]) if entry.get("relative_path") else Path(entry["path"])
            read.add(str(path))
            if str(path) in settings:
                continue
            name = entry.get("relative_path") or entry["path"]
            if not path.is_file():
                if str(path) in sessions:
                    lines.append(f"{stored} was computed from the extraction of {sessions[str(path)]} ({name}), "
                                 "which no longer exists; extract that session again, or list its recording "
                                 f"under exclude in config.yaml, then run {step} again.")
                else:
                    lines.append(f"{stored} was computed from {name}, which no longer exists; run {step} again.")
                continue
            if unchanged(entry, path):
                continue
            run = run_of(path)
            when = f" ({written_on(run['ended'])})" if run is not None and run.get("ended") else ""
            if str(path) in sessions:
                lines.append(f"{stored} was computed before {sessions[str(path)]} was re-extracted{when}; "
                             f"run {step} again to include it.")
            else:
                lines.append(f"{stored} was computed before {name} was written again{when}; run {step} again "
                             "to use it.")
        if all_sessions and read & set(sessions):
            lines += [f"{stored} was computed before {sessions[h5]} was added to the project; run {step} again "
                      "to include it." for h5 in included if h5 not in read]
    return lines


def progress(project: Project) -> str:
    """What the project holds (``summary``), then each output older than a re-run of its inputs."""
    return "\n".join([summary(project), *stale_outputs(project)])


def start(folder: str | os.PathLike = "./", exts=("dat", "mkv", "avi", "tar.gz"),
          max_depth: int = DEFAULT_MAX_DEPTH) -> Project:
    """
    The notebook's project: the folder holding (or about to hold) ``config.yaml``.

    Prints how many of the project's recordings are extracted, what its
    ``moseq2-project.yaml`` names and which outputs are older than their inputs (``progress``). A project made by MoSeq2 1.5.0 or earlier
    stops with ``OldProjectError``, saying how to finish it or start again.

    Args:
    folder (str): the project folder.
    exts (tuple): the recording extensions to look for.
    max_depth (int): how many folders below the project folder to search for recordings.

    Returns:
    (Project): the project; pass ``project.folder`` to the later cells.
    """
    from moseq2_app.gui.progress import get_extraction_progress

    project = Project.at(Path(folder).absolute())
    check_not_old(project.folder)
    path_dict, num_extracted = get_extraction_progress(str(project.folder), exts=list(exts), max_depth=max_depth)
    if path_dict:
        print(f"Extraction Progress: {num_extracted} out of {len(path_dict)} session(s) extracted")
    print(progress(project))
    return project
