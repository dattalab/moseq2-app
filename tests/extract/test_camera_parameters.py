"""Filling in the settings that depend on the camera.

``detect_and_set_camera_parameters`` runs before every extraction and in the
arena tool. It works out the camera from ``camera_type``, or from the file
extension when that is ``auto``, and fills in ``bg_roi_weights``,
``pixel_format`` and ``movie_dtype`` for it. A setting still at the generic
default (the Kinect v2's values, which a generated config holds) takes the
camera's value; a setting the user changed is kept.
"""

from __future__ import annotations

import pytest

from moseq2_extract.util import detect_and_set_camera_parameters

GENERIC = {"bg_roi_weights": [1, 0.1, 1], "pixel_format": "gray16le", "movie_dtype": "<u2"}
AZURE = {"bg_roi_weights": (10, 0.1, 1), "pixel_format": "gray16be", "movie_dtype": ">u2"}


def detect(camera_type, input_file="session/depth.mkv", **settings):
    config = {"camera_type": camera_type, **GENERIC, **settings}
    return detect_and_set_camera_parameters(config, input_file)


@pytest.mark.parametrize("camera_type", ["auto", "azure"])
def test_settings_at_the_generic_default_take_the_cameras_values(camera_type):
    config = detect(camera_type)
    assert {key: config[key] for key in AZURE} == AZURE


@pytest.mark.parametrize("camera_type", ["auto", "azure"])
def test_roi_weights_the_user_set_are_kept(camera_type):
    config = detect(camera_type, bg_roi_weights=[5, 0.2, 1])
    assert list(config["bg_roi_weights"]) == [5, 0.2, 1]
    assert config["pixel_format"] == AZURE["pixel_format"]


@pytest.mark.parametrize("camera_type", ["auto", "azure"])
def test_pixel_format_and_dtype_the_user_set_are_kept(camera_type):
    config = detect(camera_type, pixel_format="gray12le", movie_dtype="uint16")
    assert (config["pixel_format"], config["movie_dtype"]) == ("gray12le", "uint16")


@pytest.mark.parametrize("camera_type", ["k2", "kinect"])
def test_k2_is_the_kinect(camera_type, recwarn):
    """The notebook and moseq2-pca call the Kinect v2 ``k2``."""
    config = detect(camera_type, input_file="session/depth.dat", bg_roi_weights=(1, 0.1, 1))
    assert tuple(config["bg_roi_weights"]) == (1, 0.1, 1)
    assert not [w for w in recwarn if "make sure the following parameters" in str(w.message)]
