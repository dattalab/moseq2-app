"""An Azure recording extracts against its own floor, and the mouse on it is found.

An Azure Kinect's settings read depth as ``gray16be`` into ``>u2``. The
background, the floor depth the arena is searched at and every height above
the floor follow from those samples, so a recording read at the wrong byte
order extracts without an error and without a mouse: the floor of a 596 mm
recording is found at 21250 mm and the arena is 156 pixels.

The recording is ``azure_arena``'s: a floor 596 mm from the camera, the
arena's rim, and a mouse walking a circle on the floor.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

import moseq2_extract.cli as extract_cli

from azure_arena import FLOOR_DEPTH, NFRAMES, mouse, write_azure_arena_mkv


@pytest.fixture(scope="module")
def azure_extraction(tmp_path_factory):
    session = tmp_path_factory.mktemp("azure")
    write_azure_arena_mkv(session / "depth.mkv")
    (session / "metadata.json").write_text("{}")
    result = CliRunner().invoke(
        extract_cli.cli,
        ["extract", str(session / "depth.mkv"), "--camera-type", "azure",
         "--output-dir", str(session / "proc")],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    return session / "proc" / "results_00.h5"


def test_the_floor_is_found_at_the_depth_the_camera_saw(azure_extraction):
    with h5py.File(azure_extraction, "r") as f:
        true_depth = f["metadata/extraction/true_depth"][()]
    assert abs(true_depth - FLOOR_DEPTH) <= 5


def test_the_mouse_is_found_where_it_walked(azure_extraction):
    """The centroid is in the coordinates of the arena's bounding box.

    The background is the recording's first frame alone, mouse included, so
    the first few frames, where the mouse still overlaps where it started,
    may go without one.
    """
    with h5py.File(azure_extraction, "r") as f:
        rows, cols = np.nonzero(f["metadata/extraction/roi"][()])
        x = f["scalars/centroid_x_px"][()] + cols.min()
        y = f["scalars/centroid_y_px"][()] + rows.min()
    drawn = np.array([mouse(i)[1] for i in range(NFRAMES)])
    found = ~np.isnan(x)
    assert found.sum() >= 50
    assert np.median(np.hypot(x - drawn[:, 0], y - drawn[:, 1])[found]) < 1
