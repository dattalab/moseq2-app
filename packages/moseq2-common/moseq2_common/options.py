"""A step's options, declared once and resolved in one order.

Each step's settings are one frozen dataclass whose fields are made with
``option``: the field's default, its command-line flags, its click type and its
help text live together. From that class:

- ``Options.resolve`` builds the settings a run uses: the class defaults, then
  ``config.yaml``'s section for the step's stage, then the session's entry in
  ``session_config.yaml``, then the values passed explicitly (options the user
  typed, arguments a script passes).
- ``read_settings`` reads a stage's section of ``config.yaml`` for ``resolve``,
  stopping on a key no step of that stage declares.
- ``click_options`` writes the command's ``click.option`` decorators, in the
  order the command's help lists them.
- ``explicit_values`` returns the options the user typed on the command line,
  including one typed at its default value.

``config.yaml`` has one section per stage, named by each class's ``SECTION``
(``extract``, ``pca``, ``model``, ``crowd_movies``), and ``camera_type`` at its
top level. A field declared with ``setting=False`` is a one-run switch: a flag on
the command line and an argument of the notebook's function, never read from
``config.yaml``, a session entry or a job file.

A field is *tuple-valued* when it is declared ``multiple=True`` or with a tuple
``type`` (``type=(float, float)``); its default is a tuple, and ``resolve``
hands it a tuple whatever shape ``config.yaml`` holds.
"""

from __future__ import annotations

import dataclasses
import difflib
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import click
from click.core import ParameterSource
from ruamel.yaml import YAMLError

from moseq2_common.yaml import read_yaml

__all__ = [
    "SECTIONS",
    "TOP_LEVEL_KEYS",
    "ConfigFileError",
    "Options",
    "UnknownSettingError",
    "arena_fields",
    "click_options",
    "command_settings",
    "explicit_values",
    "option",
    "read_config",
    "read_settings",
    "setting_fields",
]

_KEY = "moseq2_option"

#: The sections of config.yaml, one per stage.
SECTIONS = ("extract", "pca", "model", "crowd_movies")

#: The keys config.yaml holds at its top level, outside the sections.
#: ``session_config_path`` leaves with PR 4.2, the three output paths with PR 4.3
#: (``moseq2_pca.cli.shim_write_back_paths``); ``job`` is a job settings file's.
TOP_LEVEL_KEYS = ("camera_type", "groups", "exclude", "job",
                  "session_config_path", "pca_file", "pca_file_scores", "changepoint_file")

#: Where each section's settings are listed, for the unknown-setting message.
_SECTION_HELP = {
    "extract": "`moseq2-extract extract --help` and `moseq2-extract batch-extract --help`",
    "pca": "`moseq2-pca train-pca --help`, `moseq2-pca apply-pca --help` and "
           "`moseq2-pca compute-changepoints --help`",
    "model": "`moseq2-model learn-model --help`, `moseq2-model kappa-scan --help` and "
             "`moseq2-model apply-model --help`",
    "crowd_movies": "`moseq2-viz make-crowd-movies --help`",
}

O = TypeVar("O", bound="Options")
F = TypeVar("F", bound=Callable[..., Any])


def option(default, *flags: str, help: str, type=None, multiple: bool = False,
           is_flag: bool = False, arena: bool = False, setting: bool = True) -> dataclasses.Field:
    """Declare one setting: its default, flags, click type and help text.

    ``type`` is passed to click unchanged (a Python type, a ``click.ParamType``
    or a tuple of types). ``arena`` marks a setting the arena tool detects for
    each recording. ``setting=False`` marks a one-run switch, which ``resolve``
    never takes from ``config.yaml`` or a session entry and ``generate-config``
    never writes.
    """
    spec = {"flags": flags, "help": help, "type": type, "multiple": multiple,
            "is_flag": is_flag, "arena": arena, "setting": setting}
    return dataclasses.field(default=default, metadata={_KEY: spec})


def _spec(field: dataclasses.Field) -> dict:
    return field.metadata[_KEY]


def _is_tuple_valued(spec: Mapping) -> bool:
    return spec["multiple"] or isinstance(spec["type"], tuple)


def _as_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    """Base class of every step's options."""

    #: The config.yaml section the class reads; None for a class with no section.
    SECTION: ClassVar[str | None] = None

    @classmethod
    def resolve(cls: type[O], config: Mapping | None = None, entry: Mapping | None = None,
                explicit: Mapping | None = None) -> O:
        """Class defaults, then ``config``, then ``entry``, then ``explicit``.

        From ``config`` and ``entry`` only the settings the class declares are
        read (never a ``setting=False`` switch), and a null never replaces the
        value before it. A list or a scalar for a tuple-valued field becomes a
        tuple; every other value is taken as it is. ``explicit`` applies as
        given, and a key the class does not declare in it raises ``TypeError``.
        """
        fields = {f.name: f for f in dataclasses.fields(cls)}
        values = {}
        for layer in (config, entry):
            for name, value in (layer or {}).items():
                if name not in fields or value is None or not _spec(fields[name])["setting"]:
                    continue
                if _is_tuple_valued(_spec(fields[name])):
                    value = _as_tuple(value)
                values[name] = value
        undeclared = sorted(set(explicit or {}) - set(fields))
        if undeclared:
            raise TypeError(
                f"{cls.__name__} has no option {', '.join(map(repr, undeclared))}. "
                f"Its options are: {', '.join(fields)}. Pass only these names."
            )
        values.update(explicit or {})
        return cls(**values)

    def as_dict(self) -> dict:
        """Every field by name, values as held (tuples stay tuples)."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def settings(self) -> dict:
        """The fields a settings file holds: every field but the ``setting=False`` switches."""
        return {name: value for name, value in self.as_dict().items() if name in setting_fields(type(self))}


def setting_fields(cls: type[Options]) -> tuple[str, ...]:
    """The names of the fields ``config.yaml`` may set (not the ``setting=False`` switches)."""
    return tuple(f.name for f in dataclasses.fields(cls) if _spec(f)["setting"])


class ConfigFileError(ValueError):
    """A config file named by the user is missing, empty or unreadable."""


class UnknownSettingError(ValueError):
    """A section of config.yaml sets a key no step of its stage declares, or a one-run switch."""


def read_config(path: str | os.PathLike) -> dict:
    """Return the settings in the config file at ``path``.

    Raises ``ConfigFileError`` naming the file when it is missing, unreadable,
    empty, not YAML, or holds something other than ``name: value`` settings.
    """
    advice = (
        "Check that the path is the project's config.yaml, or write a new one with "
        f"`moseq2-extract generate-config --output-file {path}`."
    )
    try:
        data = read_yaml(path)
    except FileNotFoundError:
        raise ConfigFileError(f"The config file {path} does not exist. {advice}") from None
    except OSError as error:
        raise ConfigFileError(
            f"The config file {path} could not be read ({error.strerror}). "
            "Check that it is a file you have permission to read."
        ) from None
    except YAMLError as error:
        raise ConfigFileError(
            f"The config file {path} is not valid YAML:\n{error}\n"
            "Correct the line named above, or replace the file with one written by "
            f"`moseq2-extract generate-config --output-file {path}`."
        ) from None
    if data is None:
        raise ConfigFileError(f"The config file {path} is empty. {advice}")
    if not isinstance(data, Mapping):
        holds = "a list" if isinstance(data, list) else "a single value"
        raise ConfigFileError(
            f"The config file {path} holds {holds}, not settings "
            f"written as `name: value` lines. {advice}"
        )
    return dict(data)


def _closest(key: str, names) -> list[str]:
    return difflib.get_close_matches(key, sorted(names), n=3, cutoff=0.6)


def _did_you_mean(names: list[str]) -> str:
    if not names:
        return ""
    return "Did you mean " + " or ".join(f"`{name}`" for name in names) + "? "


def check_top_level(config_file, data: Mapping) -> None:
    """Stop on a config.yaml without sections (1.5.0's) or with a setting outside them."""
    from moseq2_common.project import OldProjectError, old_project_message

    present = [name for name in SECTIONS if name in data]
    stray = [key for key in data if key not in SECTIONS and key not in TOP_LEVEL_KEYS]
    if not present and stray:
        raise OldProjectError(old_project_message(Path(config_file), "its settings have no sections"))
    for key in stray:
        sections = [name for name in SECTIONS if isinstance(data.get(name), Mapping) and key in data[name]]
        where = f"`{sections[0]}: {key}`" if sections else "the section of its stage"
        raise UnknownSettingError(
            f"{config_file} sets `{key}` at its top level, outside the sections. The top level holds "
            f"only camera_type, groups and exclude; every other setting goes in the section of its "
            f"stage (extract, pca, model or crowd_movies), as {where}. Move or delete that line and "
            "run again."
        )
    for name in present:
        if data[name] is not None and not isinstance(data[name], Mapping):
            holds = "a list" if isinstance(data[name], list) else "a single value"
            raise ConfigFileError(
                f"{config_file}'s `{name}` section holds {holds}, not settings written as "
                f"`name: value` lines indented below `{name}:`. Correct the section, or replace the "
                f"file with one written by `moseq2-extract generate-config --output-file {config_file}`."
            )


def _switch_message(config_file, key: str, field: dataclasses.Field) -> str:
    flags = [flag for flag in _spec(field)["flags"] for flag in flag.split("/") if flag.startswith("--")]
    flag = flags[0] if flags else f"--{key.replace('_', '-')}"
    return (f"`{key}` is not read from {config_file}: it applies to one run. Delete the line and pass "
            f"`{flag}` on the command line, or `{key}=` to the notebook function.")


def check_section(config_file, section: str, settings: Mapping, stage_classes: Sequence[type[Options]]) -> None:
    """Stop on a key in ``section`` that no class of ``stage_classes`` declares, or a one-run switch."""
    fields = {}
    for cls in stage_classes:
        for field in dataclasses.fields(cls):
            fields.setdefault(field.name, field)
    settings_names = {name for name, field in fields.items() if _spec(field)["setting"]}
    for key in settings:
        if key == "camera_type":
            raise UnknownSettingError(
                f"{config_file} sets `camera_type` in its `{section}` section. camera_type is the one "
                "setting at the top level of config.yaml, read by every stage: move the line to the top "
                "level (no indent) and run again."
            )
        if key in settings_names:
            continue
        if key in fields:
            raise UnknownSettingError(_switch_message(config_file, key, fields[key]))
        raise UnknownSettingError(
            f"{config_file} sets `{key}` in its `{section}` section, and no {section} step has that "
            f"setting. {_did_you_mean(_closest(key, settings_names))}Correct or delete that line and run "
            f"again; the {section} settings are listed by {_SECTION_HELP.get(section, 'the commands')}."
        )


def read_settings(config_file: str | os.PathLike | None, cls: type[Options], *,
                  stage_classes: Sequence[type[Options]]) -> dict:
    """The settings ``cls`` reads from ``config_file``: its section, checked, and ``camera_type``.

    The section named by ``cls.SECTION`` is checked against every field of
    ``stage_classes`` (the classes of every step of that stage), and the
    top-level ``camera_type`` is added when ``cls`` declares it. A class with no
    section, or no ``config_file``, reads nothing.

    Raises ``ConfigFileError`` for a file that cannot be read,
    ``moseq2_common.project.OldProjectError`` for a file without sections, and
    ``UnknownSettingError`` for a key no step of the stage declares.
    """
    if cls.SECTION is None or not config_file:
        return {}
    data = read_config(config_file)
    check_top_level(config_file, data)
    settings = dict(data.get(cls.SECTION) or {})
    check_section(config_file, cls.SECTION, settings, stage_classes)
    names = {f.name for f in dataclasses.fields(cls)}
    if "camera_type" in names and data.get("camera_type") is not None:
        settings["camera_type"] = data["camera_type"]
    return settings


def command_settings(config_file, cls: type[Options], *, stage_classes: Sequence[type[Options]] = ()) -> dict:
    """A command's settings: ``read_settings`` from its project's config, its errors as click's.

    The settings come from ``--config-file`` or, without one, from the
    ``config.yaml`` of the folder the command runs in. Without either, one line
    says the command runs outside a project.
    """
    from moseq2_common.project import OldProjectError, find_project, standalone_note

    try:
        project = find_project(config_file)
    except ConfigFileError as error:
        raise click.ClickException(str(error)) from None
    if project is None:
        click.echo(standalone_note(Path.cwd()))
    source = config_file or (project.config_file if project is not None else None)
    try:
        return read_settings(source, cls, stage_classes=stage_classes or (cls,))
    except (ConfigFileError, UnknownSettingError, OldProjectError) as error:
        raise click.ClickException(str(error)) from None


def arena_fields(cls: type[Options]) -> tuple[str, ...]:
    """The names of the fields declared with ``arena=True``."""
    return tuple(f.name for f in dataclasses.fields(cls) if _spec(f)["arena"])


def click_options(cls: type[Options], *, order: Sequence[str],
                  others: Mapping[str, Callable[[F], F]] = {}) -> Callable[[F], F]:
    """The click options of ``cls`` and ``others``, in the help order ``order``.

    Each name in ``order`` is a field of ``cls``, written as its
    ``click.option``, or a key of ``others``, whose decorator is used as it is.
    ``order`` must name every field and every key of ``others`` exactly once.
    """
    fields = {f.name: f for f in dataclasses.fields(cls)}
    both = sorted(set(fields) & set(others))
    counts = {name: list(order).count(name) for name in order}
    twice = sorted(name for name, n in counts.items() if n > 1)
    unknown = sorted(name for name in counts if name not in fields and name not in others)
    missing = [name for name in [*fields, *others] if name not in counts]
    problems = [
        f"{label}: {', '.join(names)}"
        for label, names in (("declared both as a field and in others", both),
                             ("named more than once", twice),
                             ("neither a field nor in others", unknown),
                             ("missing from order", missing))
        if names
    ]
    if problems:
        raise ValueError(
            f"The help order given for {cls.__name__} does not name each option once ("
            + "; ".join(problems) + ")."
        )

    decorators = []
    for name in order:
        if name in others:
            decorators.append(others[name])
            continue
        field = fields[name]
        spec = _spec(field)
        decorators.append(click.option(
            name, *spec["flags"], default=field.default, type=spec["type"],
            multiple=spec["multiple"], is_flag=spec["is_flag"], help=spec["help"],
        ))

    def apply(function: F) -> F:
        # click lists options in the order their decorators are written, top
        # first, so they are applied bottom first.
        for decorator in reversed(decorators):
            function = decorator(function)
        return function

    return apply


def explicit_values(ctx: click.Context, cls: type[Options]) -> dict:
    """The fields of ``cls`` the user typed on the command line, with their values."""
    return {
        f.name: ctx.params[f.name]
        for f in dataclasses.fields(cls)
        if ctx.get_parameter_source(f.name) is ParameterSource.COMMANDLINE
    }
