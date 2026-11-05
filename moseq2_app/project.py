"""
The notebooks' project cell.

``start`` names the project the notebook works in, the folder holding
``config.yaml``, and prints what it holds: how many of its recordings are
extracted, its sessions, and the outputs its ``moseq2-project.yaml`` names.
Every later cell takes the project folder and reads the bookkeeping file when it
needs a path, so nothing is kept between cells. A project made by MoSeq2 1.5.0
or earlier stops here, before anything else runs.
"""

from __future__ import annotations

import os
from pathlib import Path

from moseq2_common.bookkeeping import load
from moseq2_common.project import Project, check_not_old
from moseq2_extract.recordings import DEFAULT_MAX_DEPTH

__all__ = ["start", "summary"]


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


def start(folder: str | os.PathLike = "./", exts=("dat", "mkv", "avi", "tar.gz"),
          max_depth: int = DEFAULT_MAX_DEPTH) -> Project:
    """
    The notebook's project: the folder holding (or about to hold) ``config.yaml``.

    Prints how many of the project's recordings are extracted and what its
    ``moseq2-project.yaml`` names. A project made by MoSeq2 1.5.0 or earlier
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
    print(summary(project))
    return project
