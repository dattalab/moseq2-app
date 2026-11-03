"""Every package reads an extraction's frame times through one reader in moseq2-common.

Extraction writes ``/timestamps`` in milliseconds and records that in the
dataset's ``unit`` attribute; a file extracted before the attribute existed
records no unit and holds milliseconds. PCA wants seconds, the scalar table
and the QC cell milliseconds. Before ``moseq2_common.timestamps``, pca and viz
each kept their own unit table (one per second, the other in milliseconds),
their own lookup of ``/timestamps`` or ``/metadata/timestamps`` and their own
error for an unknown unit.

``read_timestamps(h5file, unit)`` returns the frame times in ``unit`` and
whether the file records its unit. The stored array comes back unconverted
when it is already in ``unit``, and otherwise by the same arithmetic 1.5.0
used, so no result moves.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from moseq2_common.timestamps import read_timestamps, timestamp_unit, timestamps_path

MILLISECONDS = np.arange(10) * 1000 / 30 + 25044816.2766


def write(tmp_path, values, unit=None, path="timestamps"):
    h5 = tmp_path / "results_00.h5"
    with h5py.File(h5, "w") as f:
        f[path] = values
        if unit is not None:
            f[path].attrs["unit"] = unit
    return str(h5)


def test_milliseconds_read_in_seconds(tmp_path):
    times, recorded = read_timestamps(write(tmp_path, MILLISECONDS, "milliseconds"), "seconds")
    np.testing.assert_allclose(times, MILLISECONDS / 1000)
    assert recorded is True


def test_seconds_read_in_milliseconds(tmp_path):
    times, recorded = read_timestamps(write(tmp_path, MILLISECONDS / 1000, "s"), "milliseconds")
    np.testing.assert_allclose(times, MILLISECONDS)
    assert recorded is True


def test_a_file_recording_no_unit_reads_as_milliseconds(tmp_path):
    h5 = write(tmp_path, MILLISECONDS)
    in_ms, recorded = read_timestamps(h5, "milliseconds")
    in_s, _ = read_timestamps(h5, "seconds")
    np.testing.assert_array_equal(in_ms, MILLISECONDS)
    np.testing.assert_allclose(in_s, MILLISECONDS / 1000)
    assert recorded is False


@pytest.mark.parametrize(
    "stored, stored_unit, unit",
    [(MILLISECONDS, "milliseconds", "milliseconds"), (MILLISECONDS, "ms", "milliseconds"),
     (MILLISECONDS, None, "milliseconds"), (MILLISECONDS / 1000, "seconds", "seconds"),
     (np.arange(10, dtype="int64") * 33, "milliseconds", "milliseconds")],
    ids=["milliseconds", "ms", "no-unit", "seconds", "integers"],
)
def test_the_stored_values_come_back_unchanged_when_no_conversion_is_needed(
        tmp_path, stored, stored_unit, unit):
    times, _ = read_timestamps(write(tmp_path, stored, stored_unit), unit)
    assert np.array_equal(times, stored)
    assert times.dtype == stored.dtype


def test_conversion_matches_1_5_0s_arithmetic(tmp_path):
    # pca divided milliseconds by 1000.0; viz multiplied seconds by 1000.
    stored_ms = MILLISECONDS + np.linspace(0, 1, 10) / 7
    times, _ = read_timestamps(write(tmp_path, stored_ms, "milliseconds"), "seconds")
    assert np.array_equal(times, stored_ms / 1000.0)
    times, _ = read_timestamps(write(tmp_path, stored_ms, None), "seconds")
    assert np.array_equal(times, stored_ms / 1000.0)

    stored_s = stored_ms / 1000
    times, _ = read_timestamps(write(tmp_path, stored_s, "seconds"), "milliseconds")
    assert np.array_equal(times, stored_s * 1000)


def test_an_unknown_unit_names_the_file_and_the_unit(tmp_path):
    h5 = write(tmp_path, MILLISECONDS, "fortnights")
    with pytest.raises(ValueError) as e:
        read_timestamps(h5, "seconds")
    message = str(e.value)
    assert h5 in message and "/timestamps" in message and "'fortnights'" in message
    assert "milliseconds" in message
    with h5py.File(h5, "r") as f, pytest.raises(ValueError):
        timestamp_unit(f["timestamps"])


def test_the_old_layout_metadata_timestamps_is_found(tmp_path):
    h5 = write(tmp_path, MILLISECONDS / 1000, "seconds", path="metadata/timestamps")
    with h5py.File(h5, "r") as f:
        assert timestamps_path(f) == "/metadata/timestamps"
        assert timestamp_unit(f["/metadata/timestamps"]) == "seconds"
    times, recorded = read_timestamps(h5, "milliseconds")
    np.testing.assert_allclose(times, MILLISECONDS)
    assert recorded is True


def test_a_file_without_timestamps_names_the_file(tmp_path):
    h5 = write(tmp_path, MILLISECONDS, path="frames")
    with h5py.File(h5, "r") as f:
        assert timestamps_path(f) is None
    with pytest.raises(KeyError) as e:
        read_timestamps(h5, "seconds")
    assert h5 in str(e.value)
