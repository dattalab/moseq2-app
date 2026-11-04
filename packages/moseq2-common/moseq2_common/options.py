"""A step's options, declared once and resolved in one order.

Each step's settings are one frozen dataclass whose fields are made with
``option``: the field's default, its command-line flags, its click type and its
help text live together. From that class:

- ``Options.resolve`` builds the settings a run uses: the class defaults, then
  ``config.yaml``, then the session's entry in ``session_config.yaml``, then the
  values passed explicitly (options the user typed, arguments a script passes).
- ``click_options`` writes the command's ``click.option`` decorators, in the
  order the command's help lists them.
- ``explicit_values`` returns the options the user typed on the command line,
  including one typed at its default value.

The commands share ``read_config_option``, which reads ``--config-file``, and
two shims that keep 1.5.0's command line: ``shim_click_defaults`` and
``shim_flags_without_default``.

A field is *tuple-valued* when it is declared ``multiple=True`` or with a tuple
``type`` (``type=(float, float)``); its default is a tuple, and ``resolve``
hands it a tuple whatever shape ``config.yaml`` holds.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import click
from click.core import ParameterSource
from ruamel.yaml import YAMLError

from moseq2_common.yaml import read_yaml

__all__ = [
    "ConfigFileError",
    "Options",
    "arena_fields",
    "click_options",
    "explicit_values",
    "option",
    "read_config",
    "read_config_option",
    "shim_click_defaults",
    "shim_flags_without_default",
]

_KEY = "moseq2_option"

O = TypeVar("O", bound="Options")
F = TypeVar("F", bound=Callable[..., Any])


def option(default, *flags: str, help: str, type=None, multiple: bool = False,
           is_flag: bool = False, arena: bool = False) -> dataclasses.Field:
    """Declare one setting: its default, flags, click type and help text.

    ``type`` is passed to click unchanged (a Python type, a ``click.ParamType``
    or a tuple of types). ``arena`` marks a setting the arena tool detects for
    each recording.
    """
    spec = {"flags": flags, "help": help, "type": type, "multiple": multiple,
            "is_flag": is_flag, "arena": arena}
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

    @classmethod
    def resolve(cls: type[O], config: Mapping | None = None, entry: Mapping | None = None,
                explicit: Mapping | None = None) -> O:
        """Class defaults, then ``config``, then ``entry``, then ``explicit``.

        From ``config`` and ``entry`` only the keys the class declares are read,
        and a null never replaces the value before it. A list or a scalar for a
        tuple-valued field becomes a tuple; every other value is taken as it is.
        ``explicit`` applies as given, and a key the class does not declare in
        it raises ``TypeError``.
        """
        fields = {f.name: f for f in dataclasses.fields(cls)}
        values = {}
        for layer in (config, entry):
            for name, value in (layer or {}).items():
                if name not in fields or value is None:
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


def read_config_option(config_file) -> dict | None:
    """``config.yaml``'s settings, or None without ``--config-file``; a file that cannot be read stops the command."""
    if not config_file:
        return None
    try:
        return read_config(config_file)
    except ConfigFileError as error:
        raise click.ClickException(str(error)) from None


def shim_click_defaults(ctx: click.Context, cls: type[O], config: Mapping | None = None) -> O:
    """Resolve ``cls`` from click's defaults, ``config`` and the typed options (removed by PR 4.1).

    click converts each default through its option's type (extract's ``--fps``
    30 becomes 30.0, pca's ``--gaussfilter-space`` (1.5, 1) becomes (1.5, 1.0),
    and a ``multiple`` option's None becomes ()), and 1.5.0's command line ran
    on those values where the notebook ran on the declared ones. A null in
    ``config`` keeps the default. PR 4.1 gives both front ends one resolution.
    """
    defaults = {name: ctx.params[name] for name in cls().as_dict()}
    settings = {name: value for name, value in (config or {}).items() if value is not None}
    return cls.resolve(config={**defaults, **settings}, explicit=explicit_values(ctx, cls))


def shim_flags_without_default(*names: str) -> Callable[[F], F]:
    """Declare the flags ``names`` with no default, as 1.5.0 did (removed by PR 4.1).

    Their options classes hold False, and a flag not typed is False either way;
    this keeps each option's declared default what 1.5.0's command line reports.
    Applied above ``click_options``, so the options are already on the function.
    """
    def apply(function: F) -> F:
        function.__click_params__ = [
            click.Option([param.name, *param.opts], is_flag=True, type=param.type, help=param.help)
            if param.name in names else param
            for param in function.__click_params__
        ]
        return function
    return apply
