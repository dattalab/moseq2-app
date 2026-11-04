"""Setting up a project for extraction: its ``config.yaml`` and its flip classifier.

``generate-config`` and ``download-flip-file`` on the command line and the
Extract notebook's ``generate_config_command`` and ``download_flip_command``
call these.
"""

from __future__ import annotations

from os.path import exists
from pathlib import Path

from moseq2_common.yaml import write_yaml
from moseq2_extract.helpers.wrappers import flip_file_wrapper

#: What an Azure Kinect project's config.yaml holds in place of the Kinect v2 defaults.
AZURE_SETTINGS = {
    "bg_roi_depth_range": [550, 650],
    "spatial_filter_size": [5],
    "tail_filter_size": [15, 15],
    "crop_size": [120, 120],
    "camera_type": "azure",
}


def generate_config(output_file: str, camera_type: str, *, overwrite: bool) -> Path:
    """Write ``config.yaml``: every option of ``moseq2-extract extract`` at its default.

    ``camera_type`` ``azure`` writes ``AZURE_SETTINGS`` over the Kinect v2
    defaults. An existing file is replaced only when ``overwrite`` is True;
    otherwise it is kept as it is.
    """
    from moseq2_extract.cli import extract

    if exists(output_file) and not overwrite:
        return Path(output_file)
    params = {param.name: param.default for param in extract.params if not param.required}
    if camera_type == "azure":
        params.update(AZURE_SETTINGS)
    write_yaml(params, output_file)
    return Path(output_file)


def download_flip_file(config_file: str, output_dir: str, selection: int | None) -> Path:
    """Download a flip classifier into ``output_dir`` and name it in ``config_file``.

    ``selection`` is the classifier's number in the list ``flip_file_wrapper``
    prints; None asks for it.
    """
    return Path(flip_file_wrapper(config_file, output_dir, selected_flip=selection))
