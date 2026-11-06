"""Reading an extraction's frame times, in one place.

Extraction writes each frame's time to ``/timestamps`` (``/metadata/timestamps``
in files older than moseq2-extract 0.1.3), in milliseconds, and records that in
the dataset's ``unit`` attribute. A file extracted before the attribute existed
records no unit; its timestamps are read as milliseconds.

``read_timestamps`` returns the times in the unit the caller asks for: PCA
works in seconds, the scalar table and the QC cell in milliseconds.

``frame_steps`` and ``dropped_frames`` find the frames the camera dropped, for
PCA, which puts an empty row where each one was, and for the QC cell, which
reports how many there were.

A session's frame rate is data, like its timestamps: extraction writes it to
``/metadata/fps``, from the recording's timestamps (``rate_from_timestamps``)
or the rate the video declares, and ``frame_rate`` reads it back. An
extraction written before ``/metadata/fps`` existed gives its rate from its
timestamps.
"""

from __future__ import annotations

import os

import h5py
import numpy as np

__all__ = [
    "TIMESTAMP_UNITS",
    "timestamps_path",
    "timestamp_unit",
    "read_timestamps",
    "frame_steps",
    "dropped_frames",
    "FRAME_RATE_PATH",
    "rate_from_timestamps",
    "frame_rate",
]

#: Where an extraction keeps its recording's frame rate, in frames per second.
FRAME_RATE_PATH = "/metadata/fps"

#: How many of each value of a timestamps dataset's ``unit`` attribute make a
#: second.
TIMESTAMP_UNITS = {"milliseconds": 1000.0, "ms": 1000.0, "seconds": 1.0, "s": 1.0}


def timestamps_path(f: h5py.File) -> str | None:
    """Where an open extraction file keeps its timestamps, or None when it has none."""
    for path in ("/timestamps", "/metadata/timestamps"):
        if path in f:
            return path
    return None


def timestamp_unit(dataset: h5py.Dataset) -> str | None:
    """
    The unit a timestamps dataset records in its ``unit`` attribute, or None
    when it records none (a file extracted before the attribute was written).

    Args:
    dataset (h5py.Dataset): an extraction's ``/timestamps`` dataset.

    Returns:
    unit (str or None): the recorded unit.
    """
    unit = dataset.attrs.get("unit")
    if isinstance(unit, bytes):
        unit = unit.decode()
    if unit is not None and unit not in TIMESTAMP_UNITS:
        raise ValueError(
            f"{dataset.file.filename}: the timestamps at {dataset.name} record their unit as "
            f"{unit!r}, which is not one of {', '.join(TIMESTAMP_UNITS)}. moseq2-extract writes "
            f"milliseconds; check that this file is the extraction's own results h5 and is not "
            f"damaged, or extract the session again."
        )
    return unit


def _read(f: h5py.File, unit: str) -> tuple[np.ndarray, bool]:
    path = timestamps_path(f)
    if path is None:
        raise KeyError(
            f"{f.filename} has no timestamps at /timestamps or /metadata/timestamps; check "
            "that this file is the extraction's own results h5 and is not damaged, or extract "
            "the session again."
        )
    stored_unit = timestamp_unit(f[path])
    stored = f[path][()]
    from_unit = stored_unit if stored_unit is not None else "milliseconds"
    if TIMESTAMP_UNITS[from_unit] == TIMESTAMP_UNITS[unit]:
        return stored, stored_unit is not None
    return stored / TIMESTAMP_UNITS[from_unit] * TIMESTAMP_UNITS[unit], stored_unit is not None


def read_timestamps(h5file: str | os.PathLike | h5py.File, unit: str) -> tuple[np.ndarray, bool]:
    """
    An extraction's frame times in ``unit``.

    Args:
    h5file (str, os.PathLike or h5py.File): the extraction's results h5, or the
        file open, which is left open.
    unit (str): a key of ``TIMESTAMP_UNITS``.

    Returns:
    timestamps (np.ndarray): the time of every frame, in ``unit``; the stored
        array itself when it is already in ``unit``.
    unit_recorded (bool): whether the file records its timestamps' unit.
    """
    if isinstance(h5file, h5py.File):
        return _read(h5file, unit)
    with h5py.File(h5file, "r") as f:
        return _read(f, unit)


def frame_steps(timestamps, period=None) -> np.ndarray:
    """
    How many frame periods each gap between consecutive timestamps spans, to
    the nearest whole number: 1 where no frame was dropped, n + 1 where n were.

    Args:
    timestamps (numpy.array): each frame's time, in any one unit.
    period (float): the frame period, in the timestamps' unit; by default the
        median gap, which the drops themselves do not move.

    Returns:
    steps (numpy.array): one value per gap, so one fewer than the timestamps.
    """
    timestamps = np.asarray(timestamps)
    gaps = np.diff(timestamps)
    if len(gaps) == 0:
        return gaps
    if period is None:
        period = np.median(gaps)
    if not (np.isfinite(period) and period > 0):
        raise ValueError(
            f"Cannot count dropped frames: the frame period from these timestamps is "
            f"{period:g}, and a positive period is needed. The first timestamp differences "
            f"are [{', '.join(f'{d:g}' for d in gaps[:5])}]. Check the timestamps the "
            f"session was extracted from (the recording's depth_ts.txt or timestamps.csv) "
            f"and extract the session again."
        )
    return np.round(gaps / period)


def dropped_frames(timestamps) -> int:
    """
    How many frames the camera dropped, by the rule PCA fills them with: a gap
    of n frame periods (the median gap) holds n - 1 dropped frames.

    Args:
    timestamps (numpy.array): each frame's time, in any one unit.

    Returns:
    dropped (int): the number of dropped frames.
    """
    steps = frame_steps(timestamps)
    return int(np.sum(steps[steps > 1] - 1))


def rate_from_timestamps(timestamps, unit: str) -> float | None:
    """
    The frames per second timestamps imply, from the median gap between them
    (which dropped frames do not move), to the nearest whole frame per second.

    Args:
    timestamps (numpy.array): each frame's time.
    unit (str): the timestamps' unit, a key of ``TIMESTAMP_UNITS``.

    Returns:
    rate (float or None): frames per second, or None when the timestamps give
        no positive frame period (fewer than two, or not increasing).
    """
    gaps = np.diff(np.asarray(timestamps, dtype="float64"))
    if len(gaps) == 0:
        return None
    period = np.median(gaps) / TIMESTAMP_UNITS[unit]
    if not (np.isfinite(period) and period > 0):
        return None
    rate = float(np.round(1 / period))
    return rate if rate > 0 else None


def frame_rate(h5: h5py.File) -> float:
    """
    A session's frame rate, in frames per second, from its open extraction file:
    ``/metadata/fps``, or for an extraction written before that existed, the
    rate its timestamps imply.

    Args:
    h5 (h5py.File): the session's extraction results h5, open.

    Returns:
    fps (float): frames per second.
    """
    if FRAME_RATE_PATH in h5:
        return float(h5[FRAME_RATE_PATH][()])
    rate = None
    if timestamps_path(h5) is not None:
        rate = rate_from_timestamps(read_timestamps(h5, "seconds")[0], "seconds")
    if rate is None:
        has = "timestamps that give no frame period" if timestamps_path(h5) is not None else "no timestamps"
        raise ValueError(
            f"{h5.filename} records no frame rate: it has no {FRAME_RATE_PATH} and {has}. "
            f"Check that this file is the extraction's own results h5 and is not damaged, "
            f"or extract the session again, which records the recording's frame rate."
        )
    return rate
