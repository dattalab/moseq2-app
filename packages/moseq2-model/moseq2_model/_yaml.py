"""YAML reading and writing, in one place.

``ruamel.yaml`` removed the module-level ``safe_load``/``safe_dump`` functions in
0.18 in favour of instantiating a parser. Routing every call through here keeps
that detail in one file rather than at each of the call sites.

The parser stays ``ruamel``, deliberately. ``ruamel``'s safe loader is YAML 1.2;
PyYAML is YAML 1.1 and reads unquoted ``yes``/``no``/``on``/``off`` as booleans
and ``06:00:00`` as a sexagesimal integer. moseq2 config files contain both
shapes -- ``wall_time: 06:00:00`` is interpolated straight into an ``sbatch -t``
argument by ``util.get_parameter_strings`` -- so switching parsers would quietly
change how existing user configs are read.
"""

import io

from ruamel.yaml import YAML

# ``pure=True`` selects the Python implementation, matching what the module-level
# ``safe_load`` used and keeping behaviour identical whether or not the C
# extension is available.
_yaml = YAML(typ="safe", pure=True)


def read_yaml(path_or_stream):
    """Parse a YAML file, given a path or an open stream."""
    if hasattr(path_or_stream, "read"):
        return _yaml.load(path_or_stream)
    with open(path_or_stream, "r") as handle:
        return _yaml.load(handle)


def write_yaml(data, path_or_stream):
    """Write ``data`` as YAML, given a path or an open stream."""
    if hasattr(path_or_stream, "write"):
        _yaml.dump(data, path_or_stream)
        return
    with open(path_or_stream, "w") as handle:
        _yaml.dump(data, handle)


def dump_yaml(data):
    """Serialize ``data`` to a YAML string."""
    stream = io.StringIO()
    _yaml.dump(data, stream)
    return stream.getvalue()
