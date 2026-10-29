"""Converting pixel coordinates to millimetres.

``compute_scalars`` converts each frame's centroid with ``convert_pxs_to_mm``
to give the ``*_mm`` scalars. The pipeline passes floating-point centroids,
but the function is public and takes whatever array it is given.
"""

from __future__ import annotations

import numpy as np

from moseq2_extract.util import convert_pxs_to_mm


def test_integer_pixels_give_the_same_millimetres_as_float_ones():
    """Millimetres are not rounded to whole numbers because the pixels were."""
    pixels = np.array([[257, 213], [258, 214]], dtype=np.int64)
    millimetres = convert_pxs_to_mm(pixels)

    assert np.issubdtype(millimetres.dtype, np.floating)
    np.testing.assert_allclose(millimetres, convert_pxs_to_mm(pixels.astype(float)))
