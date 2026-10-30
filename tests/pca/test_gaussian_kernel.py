"""The Gaussian kernel the changepoint curve and PCA's temporal filters smooth with.

``gauss_smooth`` convolves a signal with ``gaussian_kernel1d`` in place, so
the kernel has to be centred on zero for a smoothed frame to stay at its own
time. The changepoint call hands ``sigma`` (3.5 by default) over as the
window's half-width, which is rounded down to whole frames; a whole-number
half-width is used as given.
"""

from __future__ import annotations

import numpy as np
import pytest

from moseq2_pca.util import gauss_smooth, gaussian_kernel1d

IMPULSE = 50


def impulse():
    x = np.zeros(101)
    x[IMPULSE] = 1
    return x


def test_the_kernel_is_symmetric_about_zero():
    """Every tap left of centre has a matching tap to its right."""
    kernel = gaussian_kernel1d(sig=2.0)
    assert len(kernel) % 2 == 1
    np.testing.assert_allclose(kernel, kernel[::-1])


@pytest.mark.parametrize(
    "kwargs",
    [{"sig": 2}, {"win_length": 3.5}],
    ids=["whole-number window", "changepoint default window"],
)
def test_smoothing_keeps_a_frame_at_its_own_time(kwargs):
    """An impulse comes out centred on the frame it went in at."""
    smoothed = gauss_smooth(impulse(), **kwargs)
    assert np.argmax(smoothed) == IMPULSE
    assert np.dot(np.arange(smoothed.size), smoothed) == pytest.approx(IMPULSE)


def test_the_changepoint_default_window_is_seven_frames():
    """A half-width of 3.5 frames rounds down to 3: seven taps, -3 to 3.

    The changepoint yardstick was calibrated on a seven-frame window.
    Rounding up or to nearest would give nine and move every default
    changepoint.
    """
    assert len(gaussian_kernel1d(n=3.5, sig=1.5)) == 7
