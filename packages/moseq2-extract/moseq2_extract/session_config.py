"""
The settings each recording is extracted with, from ``config.yaml`` and ``session_config.yaml``.

``session_config.yaml`` sits in the project folder under that fixed name
(``moseq2_common.project.SESSION_CONFIG_NAME``). Each entry is keyed by its
session, the recording's path relative to the project folder
(``moseq2_common.project.session_key``), and holds the settings the arena tool
(moseq2-app's ``ArenaMaskWidget``) saved for that recording:
``ARENA_TOOL_SETTINGS`` and nothing else. The tool writes an entry when Save is
pressed for that recording (``save_entry``), and only that entry.

A recording is extracted with ``config.yaml``'s ``extract`` section, its entry
over it, and options typed on the command line over both
(``recording_options``). A recording without an entry is extracted with
``config.yaml``. An entry key outside ``ARENA_TOOL_SETTINGS`` is ignored, and a
line naming it is printed (``recording_entry``).

Below, "config.yaml's settings" (``config_data``) are its ``extract`` section and
``camera_type``, as ``extract_settings`` reads them.
"""

import os
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

from ruamel.yaml import YAMLError

from moseq2_common.options import ConfigFileError, arena_fields, read_settings
from moseq2_common.project import find_project, is_job_file, session_key
from moseq2_common.yaml import read_yaml, write_yaml
from moseq2_extract.options import EXTRACT_STAGE, ExtractOptions

#: The settings the arena tool sets for a recording, which extraction takes from
#: the recording's entry in ``session_config.yaml``: arena detection, the mouse's
#: height and crop, the cleaning filters, noise tolerance, the tracking model,
#: flip smoothing, compression, chunk overlap and the camera formats.
#: moseq2-app's ``ArenaMaskWidget.set_session_config_vars`` writes exactly these keys.
ARENA_TOOL_SETTINGS = arena_fields(ExtractOptions)


def extract_settings(config_file):
    """config.yaml's ``extract`` section, checked, with its ``camera_type``."""
    return read_settings(config_file, ExtractOptions, stage_classes=EXTRACT_STAGE)


def read_entries(project):
    """
    The entries of ``project``'s session_config.yaml, by session key.

    Args:
    project (Project): the project.

    Returns:
    (dict): {session key: entry}; empty when the project has no session_config.yaml.

    Raises:
    ConfigFileError: the file cannot be read, is not YAML, or does not hold entries.
    """
    path = Path(project.session_config_file)
    if not path.exists():
        return {}
    advice = ("Correct the file, or delete it and save each recording's settings again in the "
              "notebook's arena tool.")
    try:
        data = read_yaml(str(path))
    except OSError as error:
        raise ConfigFileError(f"The session config {path} could not be read ({error.strerror}). "
                              "Check that it is a file you have permission to read.") from None
    except YAMLError as error:
        raise ConfigFileError(f"The session config {path} is not valid YAML:\n{error}\n{advice}") from None
    if data is None:
        return {}
    if not isinstance(data, Mapping) or not all(isinstance(e, Mapping) or e is None for e in data.values()):
        raise ConfigFileError(f"The session config {path} does not hold one entry of settings per recording, "
                              f"keyed by the recording's path in the project. {advice}")
    return {str(key): dict(entry or {}) for key, entry in data.items()}


def arena_settings(entry, key="", path=""):
    """
    The settings of ``entry`` extraction takes: those in ``ARENA_TOOL_SETTINGS``.

    Any other key is ignored, and one line naming the keys, the entry and the file is printed.

    Args:
    entry (dict): a recording's entry, or None.
    key (str): the entry's session key, for the line.
    path (str): the session config, for the line.

    Returns:
    (dict): the entry's settings in ``ARENA_TOOL_SETTINGS``, in the entry's order.
    """
    entry = entry or {}
    other = [setting for setting in entry if setting not in ARENA_TOOL_SETTINGS]
    if other:
        names = ", ".join(f"`{setting}`" for setting in other)
        print(f"{path}'s entry for {key} sets {names}, which the arena tool does not set, so extraction "
              "ignores them and takes them from config.yaml's extract section. Delete those lines from the "
              "entry to silence this message.")
    return {setting: value for setting, value in entry.items() if setting in ARENA_TOOL_SETTINGS}


def recording_entry(project, recording, entries=None):
    """
    The arena settings ``recording``'s entry in ``project``'s session_config.yaml gives
    (``arena_settings``).

    Args:
    project (Project or None): the project; None outside one.
    recording (str): path to the recording.
    entries (dict): the file's entries, when already read (``read_entries``).

    Returns:
    (dict): {setting: the entry's value}; empty without a project, a session config or an entry.
    """
    if project is None:
        return {}
    if entries is None:
        entries = read_entries(project)
    key = session_key(project, recording)
    if key not in entries:
        return {}
    return arena_settings(entries[key], key, project.session_config_file)


def save_entry(project, key, settings):
    """
    Write ``settings`` as the entry for session ``key`` in ``project``'s session_config.yaml.

    Only the settings in ``ARENA_TOOL_SETTINGS`` are written. Every other entry is left as
    it is. The file is written under a temporary name and renamed into place.

    Args:
    project (Project): the project.
    key (str): the session key (``moseq2_common.project.session_key``).
    settings (Mapping): the arena tool's settings for the recording.
    """
    entries = read_entries(project)
    entries[key] = {setting: value for setting, value in settings.items() if setting in ARENA_TOOL_SETTINGS}
    path = Path(project.session_config_file)
    temporary = path.with_name(f".{path.name}.tmp")
    write_yaml(entries, str(temporary))
    os.replace(temporary, path)


def recording_options(config_data, entry, cli_options=None):
    """
    The ``ExtractOptions`` one recording is extracted with.

    ``ExtractOptions.resolve`` over config.yaml, then the entry's settings in
    ``ARENA_TOOL_SETTINGS``, then the options in ``cli_options`` that ``ExtractOptions`` declares.

    Args:
    config_data (dict): config.yaml's settings.
    entry (dict): the recording's entry in session_config.yaml, or None when it has none.
    cli_options (dict): parameters typed on the command line.

    Returns:
    (ExtractOptions): the options the recording is extracted with.
    """
    declared = {f.name for f in fields(ExtractOptions)}
    typed = {setting: value for setting, value in (cli_options or {}).items() if setting in declared}
    entry = {setting: value for setting, value in (entry or {}).items() if setting in ARENA_TOOL_SETTINGS}
    return ExtractOptions.resolve(config=config_data, entry=entry, explicit=typed)


def recordings_options(config_file, recordings, cli_options=None):
    """
    The ``ExtractOptions`` each recording is extracted with (``recording_options``).

    The entries come from the session_config.yaml of the project ``config_file`` belongs to.
    A job settings file already holds its recording's entry, so it reads none.

    Args:
    config_file (str): path to config.yaml, or a job settings file.
    recordings (list): paths to the recordings.
    cli_options (dict): parameters typed on the command line.

    Returns:
    (dict): recording path -> ``ExtractOptions``.
    """
    config_data = extract_settings(config_file)
    project = None if is_job_file(config_file) else find_project(config_file)
    entries = read_entries(project) if project is not None else {}
    return {
        recording: recording_options(config_data, recording_entry(project, recording, entries), cli_options)
        for recording in recordings
    }


def recording_settings(config_data, entry):
    """
    The settings dict extraction hands its work function for one recording: the options
    ``recording_options`` resolves, as the extraction steps build it. The arena tool computes
    its arena and preview with it.

    Args:
    config_data (dict): config.yaml's settings.
    entry (dict): the recording's entry in session_config.yaml, or None when it has none.

    Returns:
    (dict): the recording's settings.
    """
    return recording_options(config_data, entry).as_dict()


__all__ = ["ARENA_TOOL_SETTINGS", "arena_settings", "extract_settings", "read_entries",
           "recording_entry", "recording_options", "recording_settings", "recordings_options", "save_entry"]
