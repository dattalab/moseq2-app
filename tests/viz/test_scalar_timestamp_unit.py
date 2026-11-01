"""The scalar table's ``timestamps`` column is in milliseconds.

``get_timestamps_from_h5`` reads an extraction's frame times for the scalar
table and the analysis built on it. Extraction writes them in milliseconds
and records that in the dataset's ``unit`` attribute; a file that records
another unit is converted, and a file written before the attribute existed is
read as it is.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from moseq2_viz.util import get_timestamps_from_h5

MILLISECONDS = np.arange(10) * 1000 / 30


def write(tmp_path, values, unit=None, path="timestamps"):
    h5 = tmp_path / "results_00.h5"
    with h5py.File(h5, "w") as f:
        f[path] = values
        if unit is not None:
            f[path].attrs["unit"] = unit
    return str(h5)


@pytest.mark.parametrize(
    "stored, unit",
    [(MILLISECONDS, "milliseconds"), (MILLISECONDS / 1000, "seconds"), (MILLISECONDS, None)],
    ids=["milliseconds", "seconds", "no-attribute"],
)
def test_timestamps_are_returned_in_milliseconds(tmp_path, stored, unit):
    np.testing.assert_allclose(get_timestamps_from_h5(write(tmp_path, stored, unit)), MILLISECONDS)


def test_older_files_keep_their_timestamps_under_metadata(tmp_path):
    h5 = write(tmp_path, MILLISECONDS / 1000, "seconds", path="metadata/timestamps")
    np.testing.assert_allclose(get_timestamps_from_h5(h5), MILLISECONDS)


def test_an_unknown_unit_stops_with_the_file_and_the_unit(tmp_path):
    h5 = write(tmp_path, MILLISECONDS, "fortnights")
    with pytest.raises(ValueError) as e:
        get_timestamps_from_h5(h5)
    assert h5 in str(e.value) and "'fortnights'" in str(e.value)
