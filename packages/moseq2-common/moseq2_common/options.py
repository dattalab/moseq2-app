"""A step's options, declared once and resolved in one order.

Each step's settings are one frozen dataclass whose fields are made with
``option``: the field's default, its command-line flags, its click type and its
help text live together. From that class:

- ``Options.resolve`` builds the settings a run uses: the class defaults, then
  ``config.yaml``'s section for the step's stage, then the session's entry in
  ``session_config.yaml``, then the values passed explicitly (options the user
  typed, arguments a script passes).
- ``read_settings`` reads a stage's section of ``config.yaml`` for ``resolve``,
  printing one message naming the keys no step of that stage reads, which it
  ignores.
- ``click_options`` writes the command's ``click.option`` decorators, in the
  order the command's help lists them.
- ``explicit_values`` returns the options the user typed on the command line,
  including one typed at its default value.
- ``command_project`` is the one project a command runs on: ``--project-file``'s
  folder (``project_file_option``), else ``--config-file``'s project, else the
  working folder's. ``command_settings`` reads that project's settings and
  ``output_folder`` is where its outputs go by default.
- ``ProjectGroup`` is the click group of every command-line program: each of
  its commands first stops on a project made by MoSeq2 1.5.0 or earlier
  (``moseq2_common.project.check_not_old``).
- ``described_argument`` is a required command-line argument that, when left
  out, says what it is and where it usually is in a project, instead of click's
  bare "Missing argument". ``project_argument`` is one that, left out in a
  project, is the project's file (``ProjectDefault``: its PC scores, its current
  model, a folder in it); outside a project it is required.
- ``exclude_options`` are ``--exclude KEY`` and ``--no-exclude``, the sessions
  a step after extraction leaves out for one run in place of ``config.yaml``'s
  ``exclude``; ``typed_exclude`` reads them.

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

from moseq2_common.yaml import read_yaml, write_config

__all__ = [
    "SECTIONS",
    "TOP_LEVEL_KEYS",
    "ConfigFileError",
    "Options",
    "ProjectCommand",
    "ProjectGroup",
    "arena_fields",
    "click_options",
    "command_settings",
    "explicit_values",
    "command_project",
    "described_argument",
    "exclude_options",
    "option",
    "output_folder",
    "project_argument",
    "project_file_option",
    "ProjectDefault",
    "PROJECT_CHANGEPOINTS",
    "PROJECT_FOLDER",
    "PROJECT_MODEL",
    "PROJECT_SCORES",
    "in_project",
    "read_config",
    "read_settings",
    "setting_fields",
    "typed_exclude",
]

_KEY = "moseq2_option"

#: The sections of config.yaml, one per stage.
SECTIONS = ("extract", "pca", "model", "crowd_movies")

#: The keys config.yaml holds at its top level, outside the sections; ``job`` is a job settings file's.
TOP_LEVEL_KEYS = ("camera_type", "groups", "exclude", "job")

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

    def to_yaml(self, path: str | os.PathLike, *, job: Mapping) -> Path:
        """Write these options to ``path`` as a SLURM job's settings file, and return its path.

        The file is in ``config.yaml``'s shape: ``settings()`` in the class's
        section (``camera_type`` at the top level, where config.yaml holds it),
        and ``job`` as the top-level ``job:`` block, which names the project
        (``moseq2_common.project.find_project`` reads it) and the session or
        model. It is written whole, through a temporary file renamed into place.
        """
        settings = self.settings()
        data = {}
        if "camera_type" in settings:
            data["camera_type"] = settings.pop("camera_type")
        data[self.SECTION] = settings
        data["job"] = dict(job)
        path = Path(path)
        temporary = path.with_name(f".{path.name}.tmp")
        write_config(data, str(temporary))
        os.replace(temporary, path)
        return path


def setting_fields(cls: type[Options]) -> tuple[str, ...]:
    """The names of the fields ``config.yaml`` may set (not the ``setting=False`` switches)."""
    return tuple(f.name for f in dataclasses.fields(cls) if _spec(f)["setting"])


class ConfigFileError(ValueError):
    """A config file named by the user is missing, empty or unreadable."""


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


def _closest(key: str, names) -> str | None:
    """The name in ``names`` closest to ``key``, or None when none is close."""
    matches = difflib.get_close_matches(key, sorted(names), n=1, cutoff=0.6)
    return matches[0] if matches else None


def check_top_level(config_file, data: Mapping) -> None:
    """Stop on a config.yaml without sections (1.5.0's) or with a section that does not hold settings."""
    from moseq2_common.project import OldProjectError, old_project_message

    present = [name for name in SECTIONS if name in data]
    if not present and _stray_keys(data):
        raise OldProjectError(old_project_message(Path(config_file), "its settings have no sections"))
    for name in present:
        if data[name] is not None and not isinstance(data[name], Mapping):
            holds = "a list" if isinstance(data[name], list) else "a single value"
            raise ConfigFileError(
                f"{config_file}'s `{name}` section holds {holds}, not settings written as "
                f"`name: value` lines indented below `{name}:`. Correct the section, or replace the "
                f"file with one written by `moseq2-extract generate-config --output-file {config_file}`."
            )


def _stray_keys(data: Mapping) -> list[str]:
    """The keys at config.yaml's top level that are neither a section nor a top-level key."""
    return [key for key in data if key not in SECTIONS and key not in TOP_LEVEL_KEYS]


def _ticked(keys) -> str:
    return ", ".join(f"`{key}`" for key in keys)


def _switch_message(config_file, section: str, key: str, field: dataclasses.Field) -> str:
    flags = [flag for flag in _spec(field)["flags"] for flag in flag.split("/") if flag.startswith("--")]
    flag = flags[0] if flags else f"--{key.replace('_', '-')}"
    return (f"`{key}` in the `{section}` section of {config_file} is ignored: it applies to one run and is "
            f"never read from config.yaml. Pass `{flag}` on the command line, or `{key}=` to the notebook "
            "function, and delete the line to silence this message.")


def _ignored_settings(config_file, section: str, settings: Mapping,
                     stage_classes: Sequence[type[Options]]) -> tuple[list[str], list[str]]:
    """The keys of ``section`` no step reads, and the messages that name them.

    The keys no class of ``stage_classes`` declares share one message, each
    with the declared setting closest to it; a one-run switch
    (``setting=False``) and ``camera_type``, which is read from the top level
    only, each get a message of their own.
    """
    fields = {}
    for cls in stage_classes:
        for field in dataclasses.fields(cls):
            fields.setdefault(field.name, field)
    settings_names = {name for name, field in fields.items() if _spec(field)["setting"]}
    ignored, unknown, messages = [], [], []
    for key in settings:
        if key in settings_names and key != "camera_type":
            continue
        ignored.append(key)
        if key == "camera_type":
            messages.append(
                f"`camera_type` in the `{section}` section of {config_file} is ignored: camera_type is read "
                "only from the top level of config.yaml, by every stage. Move the line to the top level "
                "(no indent), or delete it, to silence this message."
            )
        elif key in fields:
            messages.append(_switch_message(config_file, section, key, fields[key]))
        else:
            closest = _closest(key, settings_names - {"camera_type"})
            unknown.append(f"`{key}`" + (f" (did you mean `{closest}`?)" if closest else ""))
    if unknown:
        messages.insert(0, (
            f"{config_file} has settings in its `{section}` section that no {section} step reads, so they "
            f"are ignored: {', '.join(unknown)}. Correct or delete those lines to silence this message."
        ))
    return ignored, messages


def _stray_message(config_file, stray: list[str]) -> str:
    return (f"{config_file} has settings at its top level, outside the sections, that no step reads, so "
            f"they are ignored: {_ticked(stray)}. The top level holds only camera_type, groups and exclude; "
            "move each of those lines into the section of its stage (extract, pca, model or crowd_movies), "
            "or delete it, to silence this message.")


def read_settings(config_file: str | os.PathLike | None, cls: type[Options], *,
                  stage_classes: Sequence[type[Options]]) -> dict:
    """The settings ``cls`` reads from ``config_file``: its section, checked, and ``camera_type``.

    The section named by ``cls.SECTION`` is checked against every field of
    ``stage_classes`` (the classes of every step of that stage), and the
    top-level ``camera_type`` is added when ``cls`` declares it. A key no step
    of the stage reads (``_ignored_settings``) and a setting at the top level
    outside the sections are left out, and a message naming them is printed.
    A class with no section, or no ``config_file``, reads nothing.

    Raises ``ConfigFileError`` for a file that cannot be read or a section that
    does not hold settings, and ``moseq2_common.project.OldProjectError`` for a
    file without sections.
    """
    if cls.SECTION is None or not config_file:
        return {}
    data = read_config(config_file)
    check_top_level(config_file, data)
    stray = _stray_keys(data)
    if stray:
        click.echo(_stray_message(config_file, stray))
    settings = dict(data.get(cls.SECTION) or {})
    ignored, messages = _ignored_settings(config_file, cls.SECTION, settings, stage_classes)
    for message in messages:
        click.echo(message)
    for key in ignored:
        del settings[key]
    names = {f.name for f in dataclasses.fields(cls)}
    if "camera_type" in names and data.get("camera_type") is not None:
        settings["camera_type"] = data["camera_type"]
    return settings


def command_settings(ctx: click.Context, cls: type[Options], *,
                     stage_classes: Sequence[type[Options]] = ()) -> dict:
    """A command's settings: ``read_settings`` from its project's config, its errors as click's.

    The settings come from ``--config-file`` or, without one, from the
    ``config.yaml`` of the project the command runs on (``command_project``),
    when it has one. Without a project, one line says the command runs outside
    a project.
    """
    from moseq2_common.project import OldProjectError, standalone_note

    project = command_project(ctx)
    if project is None:
        click.echo(standalone_note(Path.cwd()))
    source = ctx.params.get("config_file")
    if not source and project is not None and Path(project.config_file).is_file():
        source = project.config_file
    try:
        return read_settings(source, cls, stage_classes=stage_classes or (cls,))
    except (ConfigFileError, OldProjectError) as error:
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


def project_file_option(help: str):
    """``--project-file``: a project's ``moseq2-project.yaml``, by default the one in the working folder."""
    return click.option("--project-file", type=click.Path(dir_okay=False), default=None, help=help)


#: Where ``command_project`` keeps the project it resolved, in the command's ``ctx.meta``.
_PROJECT = "moseq2_common.project"


def _command_name(ctx: click.Context) -> str:
    """The command as a person types it: ``moseq2-viz make-df``, named by the program it belongs to."""
    root = ctx.find_root().command if ctx.parent is not None else ctx.command
    callback = root.callback or ctx.command.callback
    package = callback.__module__.split(".")[0] if callback else ""
    return f"{package.replace('_', '-')} {ctx.info_name}".strip()


class DescribedArgument(click.Argument):
    """A required argument that, when left out, says what it is and where it usually is (``missing``)."""

    def __init__(self, *args, missing: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.missing = missing

    def process_value(self, ctx: click.Context, value: Any) -> Any:
        try:
            return super().process_value(ctx, value)
        except click.MissingParameter:
            raise click.UsageError(
                f"`{_command_name(ctx)}` needs {(self.metavar or self.name).upper()}, {self.missing}", ctx=ctx
            ) from None


def described_argument(name: str, *, missing: str, **kwargs) -> Callable[[F], F]:
    """A required ``click.argument`` whose absence is explained by ``missing``, a sentence that completes
    "`moseq2-model learn-model` needs INPUT_FILE, ": what the argument is and where it usually is in a project.
    """
    return click.argument(name, cls=DescribedArgument, missing=missing, **kwargs)


@dataclasses.dataclass(frozen=True)
class ProjectDefault:
    """What a ``project_argument`` left out is in a project.

    ``what`` names it in messages and help ("the project's PC scores"); ``find(project, params)`` returns its
    path, or None when the project holds none yet (``params`` are the command's other values, such as the
    model an applied model is named after); ``make`` is the sentence saying how to make it.
    """

    what: str
    find: Callable[[Any, Mapping], Any]
    make: str = ""


def _recorded_pca(output: str):
    def find(project, params):
        from moseq2_common.bookkeeping import load, resolve_path

        stored = load(project).pca.get(output)
        return resolve_path(project, stored) if stored else None
    return find


def _current_model(project, params):
    from moseq2_common.bookkeeping import load, resolve_path

    stored = load(project).models.get("current")
    return resolve_path(project, stored) if stored else None


#: The PC scores, changepoints and current model the project's ``moseq2-project.yaml`` names, and its folder.
PROJECT_SCORES = ProjectDefault(
    "the project's PC scores", _recorded_pca("scores"),
    "Compute them with `moseq2-pca apply-pca` in the project folder (or the notebook's PCA cells).")
PROJECT_CHANGEPOINTS = ProjectDefault(
    "the project's changepoints", _recorded_pca("changepoints"),
    "Compute them with `moseq2-pca compute-changepoints` in the project folder (or the notebook's changepoints "
    "cell).")
PROJECT_MODEL = ProjectDefault(
    "the project's current model", _current_model,
    "Fit one with `moseq2-model learn-model`, or pick one from a kappa scan with `moseq2-viz get-best-model`, in "
    "the project folder (or the notebook's modeling cells).")
PROJECT_FOLDER = ProjectDefault("the project folder", lambda project, params: Path(project.folder))


def in_project(*parts: str, make: str = "") -> ProjectDefault:
    """A path in the project folder, such as ``models/model.p``."""
    return ProjectDefault(f"{'/'.join(parts)} in the project folder",
                          lambda project, params: Path(project.folder, *parts), make)


#: click's marker for a value not given, in the versions that have one.
_UNSET = getattr(getattr(click, "_utils", None), "UNSET", None)


def _left_out(value) -> bool:
    return value is None or (_UNSET is not None and value is _UNSET)


class ProjectArgument(DescribedArgument):
    """A command-line argument that may be left out in a project, where it is ``project_default``.

    Parsing only collects the values typed; ``fill_project_arguments`` settles which argument each is, fills
    the ones left out from the project, and converts them all, when the command runs. An argument with no
    ``project_default`` is required, and is given before any left out after it: in
    ``clip-scores [PCA_FILE] CLIP_SAMPLES``, one value is ``CLIP_SAMPLES``.
    """

    def __init__(self, *args, missing: str, project_default: ProjectDefault | None = None, **kwargs):
        # required (shown without brackets in the usage line) only when it has no project default
        super().__init__(*args, missing=missing, required=project_default is None, **kwargs)
        self.project_default = project_default

    def process_value(self, ctx: click.Context, value: Any) -> Any:
        # the value as typed: fill_project_arguments settles its place, checks it is there and converts it
        return None if _left_out(value) else value

    @property
    def shown_name(self) -> str:
        return (self.metavar or self.name).upper()

    def missing_message(self, ctx: click.Context) -> str:
        message = f"`{_command_name(ctx)}` needs {self.shown_name}, {self.missing}"
        if self.project_default is not None:
            message += (f" It can be left out when run from a project folder, to use "
                        f"{self.project_default.what}.")
        return message


def project_argument(name: str, *, missing: str, default: ProjectDefault | None = None,
                     **kwargs) -> Callable[[F], F]:
    """A ``described_argument`` that, left out in a project, is ``default`` (``ProjectDefault``).

    Outside a project it is required, and its message says it can be left out when run from a project folder.
    The command must be a ``ProjectCommand``, which fills it when it runs. Every positional argument of a
    command that has one is a ``project_argument`` (with no ``default`` when it has none).
    """
    return click.argument(name, cls=ProjectArgument, missing=missing, project_default=default, **kwargs)


def _project_arguments(command: click.Command) -> list[ProjectArgument]:
    return [param for param in command.params if isinstance(param, ProjectArgument)]


def fill_project_arguments(ctx: click.Context) -> None:
    """Settle the values typed for a command's ``project_argument``s, fill the ones left out from its project,
    and convert them.

    The values typed go, in order, to the arguments with no project default and to as many of the others,
    leftmost first, as the values allow. A required argument with no value stops the command
    (``ProjectArgument.missing_message``), and so does one left out outside a project, or in a project that
    does not hold its file yet, or holds it at a path that does not exist.
    """
    arguments = _project_arguments(ctx.command)
    if not arguments:
        return
    typed = [ctx.params.get(argument.name) for argument in arguments]
    typed = [value for value in typed if not _left_out(value)]
    required = [argument for argument in arguments if argument.project_default is None]
    filled = max(0, len(typed) - len(required))
    given = []
    for argument in arguments:
        if argument.project_default is None:
            given.append(argument)
        elif filled:
            given.append(argument)
            filled -= 1
    values = dict(zip((argument.name for argument in given), typed))
    for argument in arguments:  # the first one missing, in order, before any default is worked out
        if argument.name not in values and (argument.project_default is None or command_project(ctx) is None):
            raise click.UsageError(argument.missing_message(ctx), ctx=ctx)
    converted = {}
    for argument in arguments:
        if argument.name in values:
            converted[argument.name] = argument.type_cast_value(ctx, values[argument.name])
    for argument in arguments:
        if argument.name not in values:
            converted[argument.name] = _from_project(ctx, argument, {**ctx.params, **converted})
    ctx.params.update(converted)


def _from_project(ctx: click.Context, argument: ProjectArgument, params: Mapping) -> Any:
    """``argument``'s value when left out: its project default, converted; the command stops when there is
    none to use."""
    from moseq2_common.bookkeeping import BOOKKEEPING_NAME

    project = command_project(ctx)
    if project is None:
        raise click.UsageError(argument.missing_message(ctx), ctx=ctx)
    default = argument.project_default
    name = argument.shown_name
    path = default.find(project, params)
    make = f"{default.make} " if default.make else ""
    if path is None:
        raise click.ClickException(
            f"`{_command_name(ctx)}` was run without {name}, so it uses {default.what}, and "
            f"{Path(project.folder) / BOOKKEEPING_NAME} names none yet. {make}Or give {name} on the command line."
        )
    try:
        return argument.type_cast_value(ctx, str(path))
    except click.BadParameter:
        raise click.ClickException(
            f"`{_command_name(ctx)}` was run without {name}, so it uses {default.what}, {path}, which does not "
            f"exist. {make}Or give {name} on the command line."
        ) from None


def project_defaults_help(command: click.Command) -> str | None:
    """The help paragraph naming what each of ``command``'s arguments is when left out in a project."""
    defaults = [f"{argument.shown_name} ({argument.project_default.what})"
                for argument in _project_arguments(command) if argument.project_default is not None]
    if not defaults:
        return None
    return f"In a project these can be left out: {', '.join(defaults)}."


EXCLUDE_HELP = ("A session to leave out of this run, named by its recording's path in the project, as "
                "moseq2-project.yaml lists it (for example cohort1/mouse_1/depth.avi); repeat it for each "
                "session. The sessions given replace config.yaml's exclude for this run (default: config.yaml's "
                "exclude)")
NO_EXCLUDE_HELP = "Leave no session out of this run, whatever config.yaml's exclude lists"


def exclude_options() -> dict[str, Callable[[F], F]]:
    """``--exclude KEY`` (repeatable) and ``--no-exclude``, for ``click_options``' ``others``; ``typed_exclude``
    reads them."""
    return {
        "exclude": click.option("--exclude", multiple=True, metavar="KEY", show_default=False, help=EXCLUDE_HELP),
        "no_exclude": click.option("--no-exclude", is_flag=True, default=False, show_default=False,
                                   help=NO_EXCLUDE_HELP),
    }


def typed_exclude(ctx: click.Context) -> list[str] | None:
    """The sessions to leave out typed for this run: ``--exclude``'s keys, ``[]`` for ``--no-exclude``, or None
    (``config.yaml``'s ``exclude``) when neither was typed. Typing both stops the command."""
    keys = list(ctx.params.get("exclude") or ())
    if ctx.params.get("no_exclude"):
        if keys:
            raise click.UsageError(
                f"Both --no-exclude and --exclude ({', '.join(keys)}) were given. --exclude names the sessions to "
                "leave out of this run, and --no-exclude leaves none out: pick one.", ctx=ctx)
        return []
    return keys or None


def command_project(ctx: click.Context, *, required: bool = False):
    """The one project a command runs on, or None outside a project.

    ``--project-file``'s folder, else the project ``--config-file`` belongs to
    (``find_project``: the file's folder, or the project a job settings file
    names), else the working folder when it holds a ``config.yaml``. The project
    gives the command its settings (``command_settings``), its sessions, the
    bookkeeping file it records in, and the default place of its outputs
    (``output_folder``). It is resolved once per command.

    A ``--project-file`` that does not exist stops the command. With
    ``required``, so does finding no project, naming the folder it looked in and
    both ways out (the project folder, ``--project-file``), and a project
    without a ``moseq2-project.yaml`` to read, saying to aggregate its sessions.
    """
    from moseq2_common.bookkeeping import BOOKKEEPING_NAME
    from moseq2_common.project import Project, find_project

    if _PROJECT not in ctx.meta:
        project_file = ctx.params.get("project_file")
        if project_file:
            if not Path(project_file).is_file():
                raise click.ClickException(
                    f"`{_command_name(ctx)}` needs a project's {BOOKKEEPING_NAME}, and {project_file}, given with "
                    "--project-file, does not exist. Check the path, or run the command from the project folder "
                    "without --project-file."
                )
            ctx.meta[_PROJECT] = Project.at(Path(project_file).absolute().parent)
        else:
            ctx.meta[_PROJECT] = find_project(ctx.params.get("config_file"))
    project = ctx.meta[_PROJECT]
    if required and project is None:
        raise click.ClickException(
            f"`{_command_name(ctx)}` works on the sessions a project's {BOOKKEEPING_NAME} lists, and found no "
            f"project: there is no config.yaml in {Path.cwd()}, and no --project-file was given. Run it from the "
            f"project folder, or name the project with `--project-file /path/to/project/{BOOKKEEPING_NAME}`."
        )
    if required and not Path(project.bookkeeping_file).is_file():
        raise click.ClickException(
            f"`{_command_name(ctx)}` works on the sessions a project's {BOOKKEEPING_NAME} lists, and there is no "
            f"{BOOKKEEPING_NAME} in {project.folder} yet, so the project has no sessions. Run "
            f"`moseq2-extract aggregate-results` in {project.folder} (or the notebook's aggregation cell) to list "
            "its extractions. If the project is another folder, run the command from there, or name it with "
            f"`--project-file /path/to/project/{BOOKKEEPING_NAME}`."
        )
    return project


def output_folder(ctx: click.Context) -> Path:
    """Where a command's outputs go when no path is given: its project's folder, else the folder it runs in."""
    project = command_project(ctx)
    return Path(project.folder) if project is not None else Path.cwd()


class ProjectCommand(click.Command):
    """A command that stops on a project made by MoSeq2 1.5.0 or earlier before it runs.

    Its options are parsed first, so ``--help`` answers anywhere. The folders
    checked are the working folder, the folder of a ``--project-file`` given,
    and the project ``--config-file`` (or the ``config-file`` argument) belongs
    to (``find_project``). Its ``project_argument``s are then settled and filled
    from the project (``fill_project_arguments``), and its help names what each is
    when left out. A step's input check (``moseq2_common.records.StepInputError``)
    prints as a message, not a traceback.
    """

    def format_help_text(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        super().format_help_text(ctx, formatter)
        defaults = project_defaults_help(self)
        if defaults:
            formatter.write_paragraph()
            with formatter.indentation():
                formatter.write_text(defaults)

    def invoke(self, ctx: click.Context) -> Any:
        from moseq2_common.project import OldProjectError, check_not_old, find_project

        folders = [Path.cwd()]
        if ctx.params.get("project_file"):
            folders.append(Path(ctx.params["project_file"]).absolute().parent)
        if ctx.params.get("config_file"):
            project = find_project(ctx.params["config_file"])
            folders.append(project.folder if project is not None
                           else Path(ctx.params["config_file"]).absolute().parent)
        try:
            for folder in folders:
                check_not_old(folder)
        except OldProjectError as error:
            raise click.ClickException(str(error)) from None
        from moseq2_common.records import StepInputError

        try:
            fill_project_arguments(ctx)
            return super().invoke(ctx)
        except StepInputError as error:  # a step's input check: a message, not a traceback
            raise click.ClickException(str(error)) from None


class ProjectGroup(click.Group):
    """A command-line program whose commands are ``ProjectCommand``s."""

    command_class = ProjectCommand
