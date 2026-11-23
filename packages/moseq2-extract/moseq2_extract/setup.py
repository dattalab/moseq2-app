"""What a project's ``config.yaml`` holds when it is written: its sections and each camera's preset.

``moseq2_extract.steps.generate_config`` writes a config file from these, and
``moseq2_extract.project.generate_config`` writes a project's after checking it
is no MoSeq2 1.5.0 project; ``generate-config`` on the command line and the
Extract notebook's ``generate_config_command`` call the project layer.

``config.yaml`` holds ``camera_type`` at its top level and one section per
stage (``extract``, ``pca``, ``model``, ``crowd_movies``), each holding every
setting of its stage's steps at its default (``_stages``, ``_defaults``), with
the camera's preset values (``CAMERA_PRESETS``) in place of the defaults they
replace; then ``computing``, with a block for each installed step that does long
work, saying where it runs (``_computing``); and an empty ``exclude`` list,
where the sessions every step after extraction leaves out are named. ``project_camera`` tells the camera from the
recordings in a folder.
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

from moseq2_common.options import setting_fields
from moseq2_extract.recordings import find_recordings

#: Each camera's settings in place of the defaults, by section. A Kinect v2 takes the defaults.
CAMERA_PRESETS: dict[str, dict[str, dict]] = {
    "azure": {
        "extract": {
            "bg_roi_depth_range": [550, 650],
            "spatial_filter_size": [5],
            "tail_filter_size": [15, 15],
            "crop_size": [120, 120],
        },
        "pca": {
            "gaussfilter_space": [2.25, 1.5],
            "tailfilter_size": [13, 13],
        },
    },
    "kinect": {},
}

#: The ``camera_type`` config.yaml records for each preset: extraction's own detection
#: (``auto``) suits every Kinect v2 recording, and ``azure`` sets the Azure formats.
_CAMERA_TYPE = {"azure": "azure", "kinect": "auto"}

#: Settings left out of a generated file: ``dask_cache_path``'s default is the folder
#: each PCA step writes to, decided when the step runs.
_NOT_GENERATED = ("dask_cache_path",)


#: Each section's options classes: the module and name that declare them, in the order the file lists them.
_STAGES = {
    "extract": ("moseq2_extract.options", "EXTRACT_STAGE"),
    "pca": ("moseq2_pca.options", "PCA_STAGE"),
    "model": ("moseq2_model.options", "MODEL_STAGE"),
    "crowd_movies": ("moseq2_viz.options", "CROWD_MOVIE_STAGE"),
}


def _stages():
    """Each section's classes; ``()`` for a section whose package is not installed.

    moseq2-extract installs without moseq2-pca, moseq2-model and moseq2-viz; a
    section of a package that is not installed is written empty, and its steps
    take their defaults when the package is installed later.
    """
    stages = {}
    for section, (module, name) in _STAGES.items():
        try:
            stages[section] = getattr(importlib.import_module(module), name)
        except ModuleNotFoundError as error:
            if error.name != module.split(".")[0]:
                raise
            stages[section] = ()
    return stages


def _computing() -> dict:
    """Each installed step's ``computing`` block at its defaults, keyed by the step's command.

    The steps are the ``COMPUTING`` classes of each section's options module, in section order.
    """
    blocks = {}
    for module, _ in _STAGES.values():
        try:
            classes = getattr(importlib.import_module(module), "COMPUTING", ())
        except ModuleNotFoundError as error:
            if error.name != module.split(".")[0]:
                raise
            continue
        for cls in classes:
            blocks[cls.STEP] = _defaults((cls,))
    return blocks


def _defaults(classes) -> dict:
    """Every setting of ``classes`` at its declared default, in declaration order."""
    values = {}
    for cls in classes:
        settable = set(setting_fields(cls))
        for field in dataclasses.fields(cls):
            if field.name not in settable or field.name in _NOT_GENERATED or field.name == "camera_type":
                continue
            default = list(field.default) if isinstance(field.default, tuple) else field.default
            values.setdefault(field.name, default)
    return values


def project_camera(folder: str | Path) -> str:
    """``azure`` when the recordings found in ``folder`` are all ``.mkv`` files, else ``kinect``."""
    recordings = find_recordings(str(folder)).recordings
    if recordings and all(str(recording).endswith(".mkv") for recording in recordings):
        return "azure"
    return "kinect"
