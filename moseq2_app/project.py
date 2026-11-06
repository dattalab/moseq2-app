"""
The notebooks' project cell.

``start`` names the project the notebook works in, the folder holding
``config.yaml``, and prints what it holds (``progress``): how many of its
recordings are extracted, its sessions, the outputs its ``moseq2-project.yaml``
names, and each output older than a re-run of what it was built from (PC scores
computed before one of their sessions was re-extracted, a model fit before the
scores were computed again). An output is older when the fingerprint its record
holds for an input differs from that input's now. This is a message only: no
step reads it, or a record, to decide what to run.
Every later cell takes the project folder and reads the bookkeeping file when it
needs a path, so nothing is kept between cells. A project made by MoSeq2 1.5.0
or earlier stops here, before anything else runs.
"""

from __future__ import annotations

import os
from pathlib import Path

from datetime import datetime

from moseq2_common.bookkeeping import load, resolve_path, session_h5
from moseq2_common.project import Project, check_not_old
from moseq2_common.records import fingerprint, read_record
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
    """Each output the bookkeeping file names, as (the path it stores, the file)."""
    stored = [bookkeeping.pca.get(name) for name in ("components", "scores", "changepoints")]
    stored += list(bookkeeping.models.get("fitted") or []) + list(bookkeeping.applied_models)
    return [(path, resolve_path(project, path)) for path in stored if path]


def stale_outputs(project: Project) -> list[str]:
    """A line for each output the bookkeeping file names that is older than a re-run of one of its inputs:
    an input whose fingerprint now differs from the one the output's record holds. The settings files are
    not inputs in this sense; a changed setting takes effect when a step is run again."""
    bookkeeping = load(project)
    sessions = {str(session_h5(project, s)): key for key, s in bookkeeping.sessions.items()}
    settings = {str(Path(p).absolute()) for p in (project.config_file, project.session_config_file,
                                                  project.bookkeeping_file)}
    now, lines = {}, []
    for stored, output in _outputs(project, bookkeeping):
        try:
            record = read_record(output)
        except (OSError, ValueError):
            continue
        for entry in record.get("inputs") or []:
            recorded = entry.get("fingerprint") if isinstance(entry, dict) else None
            if not isinstance(recorded, dict):
                continue
            path = resolve_path(project, entry["relative_path"]) if entry.get("relative_path") else Path(entry["path"])
            if str(path) in settings or not path.is_file():
                continue
            if str(path) not in now:
                now[str(path)] = fingerprint(path)
            if (now[str(path)]["size"], now[str(path)]["sha256"]) == (recorded.get("size"), recorded.get("sha256")):
                continue
            when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            step = record.get("step", "the step that wrote it")
            if str(path) in sessions:
                lines.append(f"{stored} was computed before {sessions[str(path)]} was re-extracted ({when}); "
                             f"run {step} again to include it.")
            else:
                name = entry.get("relative_path") or entry["path"]
                lines.append(f"{stored} was computed before {name} was written again ({when}); run {step} again "
                             "to use it.")
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
