"""Setting up a project: its ``config.yaml`` and its flip classifier.

``generate-config`` and ``download-flip-file`` on the command line and the
Extract notebook's ``generate_config_command`` and ``download_flip_command``
call these. They are the setup commands: they write ``config.yaml`` on a
person's behalf, where no pipeline step does. Both stop on a project made by
MoSeq2 1.5.0 or earlier (``moseq2_common.project.check_not_old``) before writing.

``config.yaml`` holds ``camera_type`` at its top level and one section per
stage (``extract``, ``pca``, ``model``, ``crowd_movies``), each holding every
setting of its stage's steps at its default, with the camera's preset values
(``CAMERA_PRESETS``) in place of the defaults they replace, and an empty
``exclude`` list, where the sessions every step after extraction leaves out are
named.
"""

from __future__ import annotations

import dataclasses
import importlib
from os.path import abspath, dirname, exists
from pathlib import Path

from moseq2_common.options import check_top_level, read_config, setting_fields
from moseq2_common.project import check_not_old
from moseq2_common.yaml import write_config
from moseq2_extract.helpers.wrappers import flip_file_wrapper
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
#: moseq2-pca was imported in, which a file would pin to wherever it was generated.
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


def generate_config(output_file: str, camera_type: str, *, overwrite: bool) -> Path:
    """Write ``config.yaml``: every setting at its default, by section, with the camera's preset.

    ``camera_type`` is ``kinect`` (or ``k2``), ``azure``, or ``auto``, which
    takes ``azure`` when every recording in ``output_file``'s folder is an
    ``.mkv`` and ``kinect`` otherwise. An existing file is replaced only when
    ``overwrite`` is True; otherwise it is kept as it is. A folder holding a
    1.5.0 project stops with ``OldProjectError``.
    """
    check_not_old(dirname(abspath(output_file)))
    if exists(output_file) and not overwrite:
        return Path(output_file)
    camera = {"k2": "kinect"}.get(camera_type, camera_type)
    if camera == "auto":
        camera = project_camera(dirname(abspath(output_file)))
    if camera not in CAMERA_PRESETS:
        raise ValueError(f"generate-config takes the camera types auto, kinect (or k2) and azure, not "
                         f"{camera_type!r}. Pass one of those as --camera-type.")
    preset = CAMERA_PRESETS[camera]
    data = {"camera_type": _CAMERA_TYPE[camera]}
    for section, classes in _stages().items():
        data[section] = {**_defaults(classes), **preset.get(section, {})}
    data["exclude"] = []
    write_config(data, output_file)
    return Path(output_file)


def download_flip_file(config_file: str, output_dir: str, selection: int | None) -> Path:
    """Download a flip classifier into ``output_dir`` and name it in ``config_file``'s ``extract`` section.

    ``selection`` is the classifier's number in the list ``flip_file_wrapper``
    prints; None asks for it. A config file without sections, or a folder
    holding a 1.5.0 project, stops before anything is downloaded.
    """
    check_not_old(dirname(abspath(config_file)))
    if exists(config_file):
        check_top_level(config_file, read_config(config_file))
    return Path(flip_file_wrapper(config_file, output_dir, selected_flip=selection))
