"""Converting pixel coordinates to millimetres.

``compute_scalars`` converts each frame's centroid with ``convert_pxs_to_mm``
to give the ``*_mm`` scalars. The conversion is the pinhole relation: a pixel
``n`` columns from the image centre sits ``depth * n / f`` mm off the
optical axis, with the focal length ``f = width / (2 tan(FOV / 2))`` taken
from the nominal field of view of the camera that writes that frame size:
the Kinect v2 (512 x 424, 70.6 x 60 degrees) and the Azure Kinect and
Orbbec Femto narrow (75 x 65) and wide (120 x 120) modes. A frame size no
supported camera writes is converted as a Kinect v2 frame. The conversion
lives in ``moseq2_common.scalars``, which viz's legacy conversion uses too.
"""

from __future__ import annotations

import numpy as np
import pytest

from moseq2_extract.extract.proc import compute_scalars
from moseq2_common.scalars import convert_pxs_to_mm

DEPTH = 500.0

#: The settings extract_chunk takes from its caller, which has no defaults for them.
CHUNK_SETTINGS = dict(
    use_tracking_model=False, spatial_filter_size=(3,), temporal_filter_size=None, tail_filter_iters=1,
    cable_filter_iters=0, min_height=10, max_height=100, use_cc=False, tracking_model_segment=True,
    flip_classifier=None, flip_classifier_smoothing=51, frame_dtype="uint8", progress_bar=True,
    crop_size=(80, 80), centroid_hampel_span=5, centroid_hampel_sig=3, angle_hampel_span=5,
    angle_hampel_sig=3, model_smoothing_clips=(-300, -150), tracking_model_init="raw",
    compute_raw_scalars=False,
)


def test_integer_pixels_give_the_same_millimetres_as_float_ones():
    """Millimetres are not rounded to whole numbers because the pixels were."""
    pixels = np.array([[257, 213], [258, 214]], dtype=np.int64)
    millimetres = convert_pxs_to_mm(pixels)

    assert np.issubdtype(millimetres.dtype, np.floating)
    np.testing.assert_allclose(millimetres, convert_pxs_to_mm(pixels.astype(float)))


def test_the_image_edge_maps_to_the_edge_of_the_field_of_view():
    """The corner pixel sits depth x tan(FOV / 2) off the axis; the centre sits on it."""
    millimetres = convert_pxs_to_mm(np.array([[512.0, 424.0], [256.0, 212.0]]), true_depth=DEPTH)
    np.testing.assert_allclose(millimetres[0], DEPTH * np.tan(np.deg2rad([35.3, 30.0])))
    np.testing.assert_array_equal(millimetres[1], [0.0, 0.0])


def mm_per_pixel(resolution):
    """The x and y scale ``compute_scalars`` uses for one recording's frame size."""
    rng = np.random.default_rng(0)
    track = {
        "centroid": rng.uniform(100, 200, (5, 2)),
        "orientation": np.zeros(5),
        "axis_length": np.tile([20.0, 10.0], (5, 1)),
    }
    scalars = compute_scalars(np.full((5, 8, 8), 30.0), track, true_depth=DEPTH, resolution=resolution)
    return scalars["length_mm"] / scalars["length_px"], scalars["width_mm"] / scalars["width_px"]


@pytest.mark.parametrize(
    "resolution, field_of_view",
    [((512, 424), (70.6, 60)), ((640, 576), (75, 65)), ((512, 512), (120, 120))],
    ids=["Kinect v2", "Azure narrow", "Azure wide"],
)
def test_each_frame_size_uses_its_cameras_field_of_view(resolution, field_of_view):
    x_scale, y_scale = mm_per_pixel(resolution)
    expected = DEPTH * 2 * np.tan(np.deg2rad(np.array(field_of_view) / 2)) / np.array(resolution)
    np.testing.assert_allclose(x_scale, expected[0])
    np.testing.assert_allclose(y_scale, expected[1])


def test_an_unknown_frame_size_is_converted_as_a_kinect_v2_frame():
    np.testing.assert_allclose(mm_per_pixel((640, 480)), mm_per_pixel((512, 424)))


@pytest.mark.parametrize(
    "compute_raw_scalars", [False, True], ids=["filtered frames", "raw frames (--compute-raw-scalars)"]
)
def test_extraction_measures_a_frame_with_the_camera_that_wrote_it(compute_raw_scalars):
    """``extract_chunk`` passes the raw frame size on to ``compute_scalars``.

    640 x 576 is an Azure Kinect or Femto frame in narrow mode. The size is
    taken before the arena crop, so the mouse's millimetres use that
    camera's field of view, 75 x 65 degrees, not the Kinect v2's 70.6 x 60.
    The mouse here lies along the x axis, so its length is measured in x
    pixels and its width in y pixels. Scalars are measured on the filtered
    frames, or on the raw ones with ``--compute-raw-scalars``, and both use
    the same camera.
    """
    from moseq2_extract.extract.extract import extract_chunk
    from moseq2_extract.util import get_strels

    floor = 670.0
    nframes, height, width = 10, 576, 640
    yy, xx = np.mgrid[:height, :width]
    bground = np.full((height, width), floor, dtype=np.float32)
    chunk = np.repeat(bground[None], nframes, 0).copy()
    chunk[:, ((xx - 320) / 30.0) ** 2 + ((yy - 288) / 12.0) ** 2 < 1] = floor - 30
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[100:476, 120:520] = 1
    strels = get_strels(dict(
        bg_roi_shape="ellipse", bg_roi_dilate=(10, 10), bg_roi_erode=(10, 10),
        tail_filter_shape="ellipse", tail_filter_size=(9, 9),
        cable_filter_shape="rectangle", cable_filter_size=(5, 5),
    ))

    scalars = extract_chunk(
        chunk, roi=roi, bground=bground, true_depth=floor, **{
            **CHUNK_SETTINGS, "min_height": 10, "max_height": 100, "centroid_hampel_span": 0,
            "angle_hampel_span": 0, "compute_raw_scalars": compute_raw_scalars,
        }, **strels,
    )["scalars"]

    azure_narrow = floor * 2 * np.tan(np.deg2rad(np.array([75.0, 65.0]) / 2)) / np.array([width, height])
    np.testing.assert_allclose(
        [np.nanmedian(scalars["length_mm"] / scalars["length_px"]),
         np.nanmedian(scalars["width_mm"] / scalars["width_px"])],
        azure_narrow, rtol=1e-9,
    )
