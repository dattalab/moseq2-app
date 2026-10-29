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
