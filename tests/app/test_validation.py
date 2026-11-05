"""The counts behind the extraction QC report.

``validate_extractions`` turns each of these counts into a percentage of a
recording's frames and raises a flag at five percent. The suite walks that
cell in ``extract_modeling/test_07_validate.py``; the tests here hold the
counts themselves to what they claim to count.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from moseq2_app.roi.validation import (
    count_frames_with_small_areas,
    count_stationary_frames,
    make_session_status_dicts,
)
from moseq2_pca.util import insert_nans


def test_a_recording_that_never_stops_has_no_stationary_frames():
    """The count discounts the first frame, whose speed is defined as zero.

    When the first frame has no mouse its speed is NaN instead, and there is
    nothing to discount.
    """
    moving = pd.DataFrame({"velocity_2d_mm": [float("nan"), 1.0, 2.0, 3.0]})
    assert count_stationary_frames(moving) == 0


def test_a_recording_that_starts_without_a_mouse_counts_every_still_frame():
    """The first frame is left out whatever its speed: zero, or NaN when it has no mouse.

    The frames after it are counted as they are, so two still frames are two.
    """
    for first in (0.0, float("nan")):
        recording = pd.DataFrame({"velocity_2d_mm": [first, 0.0, 0.0, 1.0]})
        assert count_stationary_frames(recording) == 2, first


def test_small_areas_are_counted_against_the_recordings_own_mean():
    """A frame is small when its area is two standard deviations under the mean.

    Measuring the same mouse in units five times larger flags the same frames.
    """
    areas = pd.DataFrame({"area_px": [100.0] * 19 + [70.0]})
    assert count_frames_with_small_areas(areas) == 1
    assert count_frames_with_small_areas(areas * 5) == 1


def test_a_frame_is_small_from_two_standard_deviations_under_the_mean():
    """Areas with mean 100 and standard deviation 4: 94 is 1.5 below, 90 is 2.5 below.

    Fourteen frames at 100 and one each at 94, 106, 90 and 110 give exactly
    that mean and standard deviation, so only 90 is counted.
    """
    areas = pd.DataFrame({"area_px": [100.0] * 14 + [94.0, 106.0, 90.0, 110.0]})
    assert areas["area_px"].mean() == 100 and areas["area_px"].std() == 4
    assert count_frames_with_small_areas(areas) == 1


def _extraction(tmp_path, name, timestamps, unit):
    h5 = tmp_path / f"{name}.h5"
    with h5py.File(h5, "w") as f:
        f["timestamps"] = timestamps
        if unit is not None:
            f["timestamps"].attrs["unit"] = unit
    (tmp_path / f"{name}.yaml").write_text(f"uuid: {name}\nmetadata: {{}}\n")
    (tmp_path / f"{name}.mp4").touch()
    return str(tmp_path / f"{name}.mp4")


def _pca_count(timestamps):
    """How many empty rows PCA puts in a session's scores for its dropped frames."""
    filled, _, _ = insert_nans(timestamps / 1000, np.zeros(len(timestamps)))
    return int(np.isnan(filled).sum())


@pytest.mark.parametrize(
    "recording, timestamps",
    [
        # a 15 fps camera that dropped nothing: the mean rate is half of 30 fps
        ("slow_camera", np.arange(600) * (1000 / 15)),
        # a Kinect, 33.02 ms a frame, that dropped 40 of 600 frames in two runs
        ("kinect_drops", np.delete(np.arange(600) * 33.02, np.r_[100:120, 400:420])),
    ],
)
def test_qc_and_pca_count_the_same_dropped_frames(tmp_path, recording, timestamps):
    """The QC cell reports the share of a recording's frames that PCA fills as dropped.

    PCA puts an empty row in the scores for each frame a gap between timestamps
    skips, with the frame period taken from the median gap. QC counts the same
    frames, as a share of the frames the recording would have had, and raises
    its flag at five percent. A camera running at another rate drops nothing.
    """
    dropped = _pca_count(timestamps)
    share = dropped / (len(timestamps) + dropped)
    status = make_session_status_dicts(
        {recording: _extraction(tmp_path, recording, timestamps, "milliseconds")}
    )
    expected = share if share >= 0.05 else False
    assert status[recording]["dropped_frames"] == pytest.approx(expected), (dropped, share)


def test_the_qc_cell_names_the_file_whose_timestamps_have_no_frame_period(tmp_path):
    """Timestamps that are all equal give no frame period, so no count; the error names the file."""
    mp4 = _extraction(tmp_path, "stuck", np.zeros(10), "milliseconds")
    with pytest.raises(ValueError, match=r"stuck\.h5: Cannot count dropped frames"):
        make_session_status_dicts({"stuck": mp4})
