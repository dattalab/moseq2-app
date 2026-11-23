"""Saving a step's figures to the project, whatever shows them.

Every chart a step's GUI draws is also a file in the project: the project layer
draws it with matplotlib from the same data the GUI is sent, and
``save_figure`` writes it, at matplotlib's default size and resolution, as
``<stem>.png`` and ``<stem>.pdf`` with the run's
record beside them (``<stem>.yaml``, ``moseq2_common.records.sidecar_path``).
The three files are written under temporary names and renamed into place
together (``atomic_outputs``), so an interrupted save leaves the previous figure
and record as they were.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from moseq2_common.records import atomic_outputs, record_text

__all__ = ["save_figure"]


def save_figure(figure, stem: str | os.PathLike, record: Mapping) -> list[Path]:
    """Write ``figure`` (a matplotlib figure) as ``<stem>.png`` and ``<stem>.pdf``, ``record`` beside them as
    ``<stem>.yaml``; returns the two figure files. ``stem``'s folder is made when missing."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with atomic_outputs(stem, (".png", ".pdf", ".yaml")) as temporary:
        figure.savefig(f"{temporary}.png")
        figure.savefig(f"{temporary}.pdf")
        Path(f"{temporary}.yaml").write_text(record_text(record))
    return [stem.with_name(f"{stem.name}.png"), stem.with_name(f"{stem.name}.pdf")]
