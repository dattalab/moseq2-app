"""Writing a project's files by hand, for tests that set up a few settings or sessions.

A 1.6 ``config.yaml`` holds ``camera_type`` at its top level, and every other
setting in the section of its stage. These helpers take the settings of one
section as a flat dict, the way the tests list them. ``write_bookkeeping``
writes a ``moseq2-project.yaml`` listing given session files and outputs.
"""

from __future__ import annotations

from moseq2_common.yaml import read_yaml, write_yaml

#: The keys config.yaml keeps at its top level (``moseq2_common.options.TOP_LEVEL_KEYS``).
TOP_LEVEL = ("camera_type", "groups", "exclude", "job")


def sectioned(flat: dict, section: str = "extract") -> dict:
    """``flat``'s settings in ``section``, its top-level keys at the top."""
    data = {key: value for key, value in flat.items() if key in TOP_LEVEL}
    data[section] = {key: value for key, value in flat.items() if key not in TOP_LEVEL}
    return data


def write_settings(flat: dict, path, section: str = "extract") -> None:
    """Write a config.yaml holding ``flat`` in ``section``."""
    write_yaml(sectioned(flat, section), str(path))


def read_settings(path, section: str = "extract") -> dict:
    """``section``'s settings in the file at ``path``, with the top-level keys beside them."""
    data = read_yaml(str(path)) or {}
    out = dict(data.get(section) or {})
    out.update({key: value for key, value in data.items() if key in TOP_LEVEL})
    return out


def update_settings(path, section: str = "extract", **changes) -> None:
    """Set ``changes`` in ``section`` of the config.yaml at ``path``; top-level keys at the top."""
    data = read_yaml(str(path)) or {}
    for key, value in changes.items():
        if key in TOP_LEVEL:
            data[key] = value
        else:
            data.setdefault(section, {})[key] = value
    write_yaml(data, str(path))


def write_bookkeeping(project, session_files=(), **pca) -> None:
    """Write ``project``'s ``moseq2-project.yaml``: a session per file in ``session_files`` and the PCA outputs
    given as ``components=``, ``scores=``, ``changepoints=`` (paths relative to the project, or absolute)."""
    from pathlib import Path

    from moseq2_common.bookkeeping import relative_path
    from moseq2_common.project import Project

    project = Project.at(project)
    sessions = {}
    for i, path in enumerate(session_files):
        folder = relative_path(project, Path(project.folder) / path)
        sessions[f"session_{i}/depth.avi"] = {"recording": f"session_{i}/depth.avi", "folder": folder,
                                              "uuid": f"uuid-{i}"}
    write_yaml({"sessions": sessions,
                "pca": {name: relative_path(project, Path(project.folder) / path) for name, path in pca.items()}},
               str(project.bookkeeping_file))
