"""YAML reading and writing, in one place.

``ruamel.yaml`` removed the module-level ``safe_load``/``safe_dump`` functions in
0.18 in favour of instantiating a parser. Routing every call in the MoSeq2
packages through here keeps that detail, and the format every file is written
in, in one file rather than at each of the call sites.

The parser stays ``ruamel``, deliberately. ``ruamel``'s safe loader is YAML 1.2;
PyYAML is YAML 1.1 and reads unquoted ``yes``/``no``/``on``/``off`` as booleans
and ``06:00:00`` as a sexagesimal integer. moseq2 config and index files contain
both shapes -- moseq2-model's ``wall_time: 06:00:00`` is interpolated straight
into an ``sbatch -t`` argument by ``util.create_command_strings``, so it must stay
a string -- and switching parsers would quietly change how existing user files
are read.

Two things the safe writer cannot do on its own are handled here. numpy values
(a kappa read from a model, a syllable id from ``np.unique``) are written as the
plain Python value they hold; the safe representer refuses every numpy type.
And nothing is written until the whole document has serialized, so a value that
cannot be written leaves the file it was meant to replace as it was.
"""

import io
import reprlib

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.representer import RepresenterError

__all__ = ["UnwritableValueError", "dump_yaml", "read_yaml", "write_yaml"]

_ADVICE = (
    "If you set this value, change it to text, a number, true/false, a list or a "
    "dictionary and run the step again; if MoSeq2 set it, please report this message "
    "at https://github.com/dattalab/moseq2-app/issues."
)


class UnwritableValueError(RepresenterError):
    """A value in the data has no YAML form, so nothing was written.

    A ``RepresenterError``, so code that caught the safe writer's own error
    catches this one too.

    Attributes:
        key_path: the keys and list indices leading from the top of the data to
            the value (for a key, to the mapping that holds it).
        value: the value (or key) that cannot be written.
    """

    def __init__(self, message, key_path, value):
        super().__init__(message)
        self.key_path = tuple(key_path)
        self.value = value


def _parser():
    """A YAML parser configured the way ``safe_dump`` was.

    A new one per call, because a ``YAML`` instance carries a serializer across
    calls: if a value cannot be represented, the failed call leaves that
    serializer open and every later dump writes nothing at all, silently.
    Callers across the packages move their output over a file the user already
    has, so a dumper that quietly produces nothing destroys data.

    ``pure=True`` selects the Python implementation, matching what the
    module-level functions used.

    The two assignments restore emitter behaviour that ``YAML()`` does not
    share with ``safe_dump``. ``allow_unicode`` escapes non-ASCII rather than
    writing it literally, which also keeps a write from depending on the
    locale encoding of the file handle. ``width`` is 80 in both, but the two
    versions measure the column at which a flow sequence may break two
    characters apart, so 82 here reproduces where ``safe_dump`` put its line
    breaks -- both for the two-element ``path: [x.h5, x.yaml]`` entries that
    fill an index file and for sequences long enough to wrap.

    One emitter difference is left, and no configuration reaches it: where a
    flow sequence does wrap, this version writes a space before the break. The
    breaks land in the same places, so the files moseq2 writes -- indexes,
    session results, configs, the PCA parameter yaml, crowd-movie info -- come
    out byte for byte as they did.
    """
    parser = YAML(typ="safe", pure=True)
    parser.allow_unicode = False
    parser.width = 82
    return parser


def _plain(value):
    """``value`` with every numpy scalar and array replaced by its plain value.

    Follows ``keypoint_moseq.io._plain``, with two differences. Dict keys are
    converted too, since syllable ids from ``np.unique`` end up as keys. And
    only exact ``dict``, ``list`` and ``tuple`` are walked, so a container the
    safe writer refuses (a ``defaultdict``) still raises, and an
    ``OrderedDict`` still writes as ``!!omap``. A tuple becomes a list, which
    is how the safe writer writes one anyway, except as a dict key, where it
    stays a tuple so it can still be a key. The input is never changed.
    """
    kind = type(value)
    if kind is dict:
        return {_plain_key(k): _plain(v) for k, v in value.items()}
    if kind is list or kind is tuple:
        return [_plain(item) for item in value]
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tolist()
    return value


def _plain_key(key):
    if type(key) is tuple:
        return tuple(_plain_key(item) for item in key)
    return _plain(key)


def _writable(value):
    try:
        _parser().dump(value, io.StringIO())
    except RepresenterError:
        return False
    return True


def _find_unwritable(data, key_path=()):
    """``(key_path, value, is_key)`` for the first thing the safe writer refuses.

    Depth first, in iteration order. Runs only after a dump has failed, so its
    cost does not matter.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if not _writable(key):
                return key_path, key, True
            found = _find_unwritable(value, key_path + (key,))
            if found:
                return found
    elif isinstance(data, (list, tuple)):
        for index, value in enumerate(data):
            found = _find_unwritable(value, key_path + (index,))
            if found:
                return found
    return None if _writable(data) else (key_path, data, False)


def _type_name(value):
    kind = type(value)
    if kind.__module__ == "builtins":
        return kind.__qualname__
    return f"{kind.__module__}.{kind.__qualname__}"


def _serialize(data, target, kind):
    """The whole YAML document for ``data``, or ``UnwritableValueError``.

    ``kind`` is ``"file"``, ``"stream"`` or ``"string"``; with ``target``, it
    decides how the error names what was being written.
    """
    plain = _plain(data)
    buffer = io.StringIO()
    try:
        _parser().dump(plain, buffer)
    except RepresenterError as err:
        found = _find_unwritable(plain)
        if found is None:
            raise
        key_path, value, is_key = found
        where = "".join(f"[{k!r}]" for k in key_path) or "the top level"
        location = f"a key in the mapping at {where}" if is_key else f"the value at {where}"
        core = (
            f"{location} is {reprlib.repr(value)}, of type {_type_name(value)}, "
            "which a YAML file cannot hold."
        )
        if kind == "file":
            message = f"Cannot save {target}: {core} {target} was left as it was. {_ADVICE}"
        elif kind == "stream":
            message = f"Cannot write YAML to {target}: {core} Nothing was written to it. {_ADVICE}"
        else:
            message = f"Cannot write this data as YAML: {core} {_ADVICE}"
        raise UnwritableValueError(message, key_path, value) from err
    return buffer.getvalue()


def read_yaml(path_or_stream):
    """Parse a YAML file, given a path or an open stream."""
    if hasattr(path_or_stream, "read"):
        return _parser().load(path_or_stream)
    with open(path_or_stream, "r") as handle:
        return _parser().load(handle)


def write_yaml(data, path_or_stream):
    """Write ``data`` as YAML, given a path or an open stream.

    The whole document is serialized before anything is written, so a value
    that cannot be written raises ``UnwritableValueError`` with the file (or
    stream) exactly as it was. A path is then opened in text mode, so each
    platform keeps the line endings the packages have always written, and
    written in place rather than replaced, so a symlinked file stays a symlink
    and keeps its permissions.
    """
    if hasattr(path_or_stream, "write"):
        name = getattr(path_or_stream, "name", "the output stream")
        path_or_stream.write(_serialize(data, name, "stream"))
        return
    text = _serialize(data, path_or_stream, "file")
    with open(path_or_stream, "w") as handle:
        handle.write(text)


def dump_yaml(data):
    """Serialize ``data`` to a YAML string."""
    return _serialize(data, None, "string")
