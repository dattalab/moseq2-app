"""The counts behind the extraction QC report.

``validate_extractions`` turns each of these counts into a percentage of a
recording's frames and raises a flag at five percent. The suite walks that
cell in ``extract_modeling/test_07_validate.py``; the tests here hold the
counts themselves to what they claim to count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moseq2_app.roi.validation import (
    check_timestamp_error_percentage,
    count_frames_with_small_areas,
    count_stationary_frames,
)


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


def test_integer_timestamps_are_rescaled_like_float_ones():
    """Millisecond timestamps are rescaled to seconds before the rate is measured.

    Extraction writes them as floats, but an h5 written any other way can hold
    integers, and those should give the same answer.
    """
    milliseconds = np.arange(10) * 33
    assert check_timestamp_error_percentage(milliseconds) == check_timestamp_error_percentage(
        milliseconds.astype(float)
    )


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
