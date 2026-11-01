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


# Notebooks from 1.4.5 and earlier left ``camera_type: k2`` (the PCA cell) or
# ``camera_type: auto`` (the config cell) in an Azure or Femto project's config.
# Only those cameras write ``.mkv``, so an ``.mkv`` recording gets their settings.

KINECT = {"bg_roi_weights": (1, 0.1, 1), "pixel_format": "gray16le", "movie_dtype": "<u2"}


def camera_lines(out):
    return [line for line in out.splitlines() if "camera_type" in line]


@pytest.mark.parametrize("camera_type", ["k2", "kinect"])
def test_an_mkv_recording_gets_the_azure_settings_when_the_config_says_kinect(camera_type, capsys):
    config = detect(camera_type, input_file="project/session_1/depth.mkv")
    assert {key: config[key] for key in AZURE} == AZURE
    assert camera_lines(capsys.readouterr().out) == [
        f"project/session_1/depth.mkv is an .mkv recording, which only Azure Kinect and "
        f"Orbbec Femto cameras write, but config.yaml says camera_type: {camera_type} "
        f"(the Kinect v2). Used the Azure settings for it: bg_roi_weights [10, 0.1, 1], "
        f"pixel_format gray16be, movie_dtype >u2. Set camera_type: azure in config.yaml."
    ]


def test_an_mkv_recording_keeps_settings_the_user_set_when_the_config_says_k2(capsys):
    config = detect("k2", input_file="project/session_1/depth.mkv", bg_roi_weights=[5, 0.2, 1])
    assert list(config["bg_roi_weights"]) == [5, 0.2, 1]
    assert (config["pixel_format"], config["movie_dtype"]) == ("gray16be", ">u2")
    assert "bg_roi_weights [5, 0.2, 1]" in capsys.readouterr().out


def test_auto_with_an_mkv_recording_is_azure_without_a_line(capsys):
    config = detect("auto", input_file="project/session_1/depth.mkv")
    assert {key: config[key] for key in AZURE} == AZURE
    assert camera_lines(capsys.readouterr().out) == []


@pytest.mark.parametrize("camera_type", ["k2", "kinect"])
def test_a_dat_recording_stays_kinect_without_a_line(camera_type, capsys):
    config = detect(camera_type, input_file="project/session_1/depth.dat")
    assert {key: tuple(v) if isinstance(v, list) else v for key, v in config.items() if key in KINECT} == KINECT
    assert camera_lines(capsys.readouterr().out) == []
