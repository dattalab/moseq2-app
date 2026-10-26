"""YAML reading and writing, in one place.

``ruamel.yaml`` removed the module-level ``safe_load``/``safe_dump`` functions in
0.18 in favour of instantiating a parser. Routing every call through here keeps
that detail in one file rather than at each of the call sites.

The parser stays ``ruamel``, deliberately. ``ruamel``'s safe loader is YAML 1.2;
PyYAML is YAML 1.1 and reads unquoted ``yes``/``no``/``on``/``off`` as booleans
and ``06:00:00`` as a sexagesimal integer. moseq2 config and index files contain
both shapes, so switching parsers would quietly change how existing user files
are read.
"""

import io

from ruamel.yaml import YAML


def _parser():
    """A YAML parser configured the way ``safe_dump`` was.

    A new one per call, because a ``YAML`` instance carries a serializer across
    calls: if a value cannot be represented, the failed call leaves that
    serializer open and every later dump writes nothing at all, silently.
    Three of the four callers here move their output over a file the user
    already has, so a dumper that quietly produces nothing destroys data.

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
    breaks land in the same places, so the files moseq2 writes -- the session
    index, the project and per-session configs -- come out byte for byte as
    they did.
    """
    parser = YAML(typ="safe", pure=True)
    parser.allow_unicode = False
    parser.width = 82
    return parser


def read_yaml(path_or_stream):
    """Parse a YAML file, given a path or an open stream."""
    if hasattr(path_or_stream, "read"):
        return _parser().load(path_or_stream)
    with open(path_or_stream, "r") as handle:
        return _parser().load(handle)


def write_yaml(data, path_or_stream):
    """Write ``data`` as YAML, given a path or an open stream."""
    if hasattr(path_or_stream, "write"):
        _parser().dump(data, path_or_stream)
        return
    with open(path_or_stream, "w") as handle:
        _parser().dump(data, handle)


def dump_yaml(data):
    """Serialize ``data`` to a YAML string."""
    stream = io.StringIO()
    _parser().dump(data, stream)
    return stream.getvalue()
