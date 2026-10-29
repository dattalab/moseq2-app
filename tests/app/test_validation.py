"""The counts behind the extraction QC report.

``validate_extractions`` turns each of these counts into a percentage of a
recording's frames and raises a flag at five percent. The suite walks that
cell in ``extract_modeling/test_07_validate.py``; the tests here hold the
counts themselves to what they claim to count.
"""

from __future__ import annotations

import pandas as pd

from moseq2_app.roi.validation import count_stationary_frames


def test_a_recording_that_never_stops_has_no_stationary_frames():
    """The count discounts the first frame, whose speed is defined as zero.

    When the first frame has no mouse its speed is NaN instead, and there is
    nothing to discount.
    """
    moving = pd.DataFrame({"velocity_2d_mm": [float("nan"), 1.0, 2.0, 3.0]})
    assert count_stationary_frames(moving) == 0
