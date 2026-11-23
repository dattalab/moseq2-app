"""
The 1.x notebook functions that write a project's config file and download a flip classifier.

Each calls the step in the project layer, ``moseq2_extract.project``. The
questions a person answers here (whether to replace a config file, which flip
classifier to download) are asked by these functions, never by the layer.
Extraction itself is the notebook's Extract step, ``moseq2_app.notebook.extract``.
"""

from os.path import exists

from moseq2_common.interface import answered
from moseq2_extract import project as project_steps
from moseq2_extract.util import filter_warnings


@filter_warnings
def generate_config_command(output_file, camera_type="auto"):
    """
    Generate configuration file (config.yaml) to use throughout pipeline.

    Args:
    output_file (str): path to saved config file.
    camera_type (str): ``auto``, ``kinect`` (``k2``) or ``azure``, as ``moseq2-extract generate-config
        --camera-type`` takes it.

    Returns:
    (str): status message.
    """

    # Check if the file already exists, and prompt user if they would like to overwrite pre-existing file
    if exists(output_file):
        ow = input(
            "This file already exists, would you like to overwrite it? [y -> yes, n -> no] "
        )
        if ow.lower() != "y":
            return "Configuration file has been retained"
    else:
        print("Creating configuration file.")
    project_steps.generate_config(output_file, camera_type, overwrite=True)

    return "Configuration file has been successfully generated."


def download_flip_command(output_dir, config_file="", selection=1):
    """
    Download flip classifier and saves its path to config file (config.yaml)

    Args:
    output_dir (str): path to output directory to save flip classifier
    config_file (str): path to config file (config.yaml)
    selection (int): index of which flip file to download (default is Adult male C57 classifer); None lists
        them and asks.

    Returns:
    """

    answered(project_steps.download_flip_file, config_file, output_dir, selection=selection)
