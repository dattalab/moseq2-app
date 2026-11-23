"""The ``moseq2`` command's commands, one module per step of the pipeline.

Each module declares ``ORDER`` (its place in ``moseq2 --help``: setup 10, arena
20, extract 30, diagnostics 40, pca 50, records 90) and tuples of click
commands: ``STEPS`` (``moseq2 <step>``, in the project in the working folder),
``STANDALONE`` (``moseq2 standalone <step>``, on files named), ``RECORDS`` and
``HIDDEN`` (run by MoSeq2 itself, such as a cluster job's ``run-job``).
``moseq2_app.cli`` imports every module here (``modules``) and builds the
command from them, so a step's commands are added by adding its module.
"""

from __future__ import annotations

import importlib
import pkgutil


def modules() -> list:
    """Every module of this package, in ``ORDER``."""
    found = [importlib.import_module(f"{__name__}.{info.name}") for info in pkgutil.iter_modules(__path__)]
    return sorted(found, key=lambda module: getattr(module, "ORDER", 100))
