"""
The settings each recording is extracted with, from ``config.yaml`` and ``session_config.yaml``.

The arena tool (moseq2-app's ``ArenaMaskWidget``) writes ``session_config.yaml``
with one entry per recording. An entry starts as a full copy of ``config.yaml``,
taken when the tool first lists the recording, and holds the values tuned in the
tool once they are saved. Extraction takes from a recording's entry only the
settings the tool sets, ``ARENA_TOOL_SETTINGS``, and every other setting from
``config.yaml``, so a setting changed in ``config.yaml`` later (a flip
classifier, a ``frame_trim``) reaches every recording. Options typed on the
command line win over both. A recording without an entry is extracted with
``config.yaml``. Entries in files written by earlier releases, which hold every
setting, are read by the same rule and not rewritten.

Where an entry holds a value for another setting that differs from
``config.yaml``'s, ``RecordingConfigs.message`` says so, one line per setting.

``recording_configs`` is what every extraction path uses: ``extract_command``
(the notebook's local extraction and ``batch-extract``) and
``run_slurm_extract`` (each SLURM job's ``config.yaml``). It resolves each
recording with ``recording_config``, which the arena tool also uses for the arena
it shows and its preview extraction, with the values the tool holds as the entry.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from os.path import basename, exists

from moseq2_common.yaml import read_yaml
from moseq2_extract.recordings import session_config_key

#: The settings the arena tool sets for a recording, which extraction takes from
#: the recording's entry in ``session_config.yaml``. moseq2-app's
#: ``ArenaMaskWidget.set_session_config_vars`` writes exactly these keys.
ARENA_TOOL_SETTINGS = (
    # arena detection
    "bg_roi_depth_range",
    "manual_set_depth_range",
    "bg_roi_weights",
    "bg_roi_shape",
    "bg_roi_index",
    "bg_roi_dilate",
    "dilate_iterations",
    # the mouse
    "min_height",
    "max_height",
    "crop_size",
    # cleaning
    "spatial_filter_size",
    "temporal_filter_size",
    "tail_filter_iters",
    "tail_filter_shape",
    "tail_filter_size",
    "cable_filter_iters",
    "cable_filter_shape",
    "cable_filter_size",
    "noise_tolerance",
    # tracking model, flip smoothing, output
    "use_tracking_model",
    "tracking_model_mask_threshold",
    "flip_classifier_smoothing",
    "compress",
    "chunk_overlap",
    # camera formats
    "pixel_format",
    "movie_dtype",
    "frame_dtype",
)

#: Settings an entry may hold that extraction never read from it, so a
#: difference in them is not reported: file bookkeeping, and the values
#: ``batch-extract`` and the notebook pass to the extraction on their own.
_NOT_REPORTED = frozenset(
    {"session_config_path", "config_file", "num_frames", "skip_completed", "cluster_type"}
)


def _plain(value):
    """A value as YAML holds it, so a tuple and a list of the same items compare equal."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _show(value):
    """A value written as it appears in a YAML file."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_show(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_show(v)}" for k, v in value.items()) + "}"
    return str(value)


def _command_line_option(setting):
    """The ``batch-extract`` option that sets ``setting``, or None."""
    from moseq2_extract.cli import batch_extract

    for param in batch_extract.params:
        if param.name == setting:
            return max(param.opts, key=len)
    return None


@dataclass
class RecordingConfigs:
    """
    What ``recording_configs`` resolved.

    Attributes:
    configs (dict): recording path -> the settings it is extracted with.
    differences (dict): recording path -> (its entry's name, {setting: the entry's value}) for
        each setting outside ``ARENA_TOOL_SETTINGS`` whose entry value differs from config.yaml's.
    config_file (str): path to config.yaml.
    session_config_path (str): path to session_config.yaml, or "" when there is none.
    config_data (dict): config.yaml as read.
    """

    configs: dict
    differences: dict = field(default_factory=dict)
    config_file: str = ""
    session_config_path: str = ""
    config_data: dict = field(default_factory=dict)

    def message(self, recordings=None):
        """
        Say which settings in session_config.yaml differ from config.yaml and that config.yaml's are used.

        One line per setting, listing every recording whose entry holds a different value,
        grouped by that value.

        Args:
        recordings (list): the recordings to report; None reports every one resolved.

        Returns:
        (str or None): the message, or None when nothing differs.
        """
        if recordings is None:
            recordings = list(self.configs)
        by_setting = {}
        for recording in recordings:
            if recording not in self.differences:
                continue
            name, settings = self.differences[recording]
            for setting, value in settings.items():
                by_setting.setdefault(setting, {}).setdefault(_show(value), []).append(name)
        if not by_setting:
            return None

        config_name = basename(self.config_file)
        session_name = basename(self.session_config_path)
        lines = [
            f"{self.session_config_path} holds values for these recordings that differ from "
            f"{self.config_file}. Extraction takes them from {config_name}: {session_name} "
            "supplies only the settings the arena tool sets (arena detection, mouse height, "
            "crop size, the cleaning filters, noise tolerance, the tracking model, flip "
            "smoothing, compression, chunk overlap and the camera formats)."
        ]
        for setting in sorted(by_setting):
            held = " and ".join(
                f"{value} for {', '.join(names)}" for value, names in by_setting[setting].items()
            )
            option = _command_line_option(setting)
            how = f"To change {setting}, set it in {config_name}"
            if option:
                how += f", or type {option} on batch-extract's command line"
            lines.append(
                f"  {setting}: {config_name}'s {_show(self.config_data[setting])} is used. "
                f"{session_name} holds {held}. {how}."
            )
        return "\n".join(lines)


def recording_configs(config_file, recordings, cli_options=None):
    """
    The settings each recording is extracted with.

    From the recording's entry in session_config.yaml, the settings in ``ARENA_TOOL_SETTINGS``,
    and any setting config.yaml does not hold; from config.yaml, every other setting; from
    ``cli_options``, the options typed on the command line, over both. A recording without an
    entry, or a project without session_config.yaml, takes config.yaml.

    Args:
    config_file (str): path to config.yaml; its session_config_path names session_config.yaml.
    recordings (list): paths to the recordings.
    cli_options (dict): parameters typed on the command line.

    Returns:
    (RecordingConfigs): each recording's settings, and the differences to report.
    """
    config_data = read_yaml(config_file) or {}
    session_config_path = config_data.get("session_config_path") or ""
    session_configs = (read_yaml(session_config_path) or {}) if exists(session_config_path) else {}
    typed = cli_options or {}

    resolved = RecordingConfigs(
        configs={},
        config_file=config_file,
        session_config_path=session_config_path,
        config_data=config_data,
    )
    for recording in recordings:
        key = session_config_key(session_configs, recording)
        entry = session_configs[key] if key is not None else None
        config, differing = recording_config(config_data, entry, typed)
        if differing:
            resolved.differences[recording] = (key, differing)
        resolved.configs[recording] = config
    return resolved


def recording_config(config_data, entry, cli_options=None):
    """
    The settings one recording is extracted with, from config.yaml and its entry.

    From ``entry``, the settings in ``ARENA_TOOL_SETTINGS`` and any setting ``config_data``
    does not hold; from ``config_data``, every other setting; from ``cli_options``, the
    options typed on the command line, over both. ``recording_configs`` resolves every
    recording of an extraction with it, and the arena tool its arena and preview extraction.

    Args:
    config_data (dict): config.yaml as read.
    entry (dict): the recording's entry in session_config.yaml, or None when it has none.
    cli_options (dict): parameters typed on the command line.

    Returns:
    config (dict): the settings the recording is extracted with.
    differing (dict): {setting: the entry's value} for each setting outside
        ``ARENA_TOOL_SETTINGS`` whose entry value differs from config.yaml's and is reported.
    """
    typed = cli_options or {}
    config = deepcopy(config_data)
    differing = {}
    for setting, value in (entry or {}).items():
        if setting in ARENA_TOOL_SETTINGS or setting not in config_data:
            config[setting] = deepcopy(value)
        elif (
            setting not in _NOT_REPORTED
            and setting not in typed
            and _plain(value) != _plain(config_data[setting])
        ):
            differing[setting] = value
    config.update(deepcopy(typed))
    return config, differing
