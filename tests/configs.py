"""Writing and reading ``config.yaml`` in its sectioned shape, for tests that set a few settings by hand.

A 1.6 ``config.yaml`` holds ``camera_type`` (and, until later pull requests,
``session_config_path`` and the PCA output paths) at its top level, and every
other setting in the section of its stage. These helpers take the settings of
one section as a flat dict, the way the tests list them.
"""

from __future__ import annotations

from moseq2_common.yaml import read_yaml, write_yaml

#: The keys config.yaml keeps at its top level (``moseq2_common.options.TOP_LEVEL_KEYS``).
TOP_LEVEL = ("camera_type", "groups", "exclude", "job", "session_config_path", "pca_file", "pca_file_scores",
             "changepoint_file")


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
