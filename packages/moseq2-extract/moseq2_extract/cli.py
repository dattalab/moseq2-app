"""
The ``moseq2-extract`` command: finding an arena, writing a config file, downloading a flip classifier.

Extraction itself is ``moseq2 extract`` in a project and ``moseq2 standalone
extract`` on recordings named (``moseq2_app.cli``). Each command's settings are
generated from its options class in ``moseq2_extract.options``; ``find-roi``
reads the ``extract`` section of its project's ``config.yaml`` (``--config-file``,
or the working folder's), lays the arena tool's entry for the recording in the
project's ``session_config.yaml`` over it, and calls the project layer,
``moseq2_extract.project``, writing into the recording's session folder in the
project's ``aggregate_results/``; outside a project it calls the file layer,
``moseq2_extract.steps``, writing into ``proc/`` beside the recording.
``download-flip-file`` asks which classifier to download
(``moseq2_common.interface.answered``).
"""

import os

import click

from moseq2_common.interface import answered
from moseq2_common.options import (
    ConfigFileError,
    ProjectGroup,
    click_options,
    command_project,
    command_settings,
    described_argument,
    explicit_values,
)
from moseq2_common.project import is_job_file
from moseq2_common.records import print_record_command
from moseq2_extract import project as project_steps, steps
from moseq2_extract.options import EXTRACT_STAGE, RoiOptions
from moseq2_extract.session_config import recording_entry


@click.group(cls=ProjectGroup, context_settings={"show_default": True})
@click.version_option()
def cli():
    pass


def _roi_others(argument):
    """The options of find-roi that are not settings, and its argument."""
    return {
        **argument,
        "config_file": click.option("--config-file", type=click.Path(), default=None),
        "output_dir": click.option("--output-dir", default=None, show_default=False,
                                   help="Folder for the background and arena images (default: the recording's "
                                        "session folder in the project's aggregate_results, or proc beside the "
                                        "recording outside a project)"),
    }


def _recording(command, what, example="depth.avi"):
    """A command's ``INPUT_FILE``: a depth recording, ``what`` saying what the command does with it."""
    return described_argument(
        "input-file", type=click.Path(exists=True, resolve_path=False),
        missing=f"the depth recording {what}: a .dat, .avi, .mkv or .tar.gz file, which in a project sits in its "
                f"session's folder. Give its path, for example `moseq2-extract {command} session_1/{example}` "
                "from the project folder.",
    )


# The help order of find-roi, as 1.5.0 lists its options.
_ROI_ORDER = (
    "progress_bar", "config_file", "use_plane_bground", "output_dir", "noise_tolerance",
    "erode_iterations", "bg_roi_erode", "dilate_iterations", "bg_sort_roi_by_position_max_rois",
    "bg_sort_roi_by_position", "bg_roi_fill_holes", "bg_roi_gradient_kernel", "bg_roi_gradient_threshold",
    "bg_roi_gradient_filter", "bg_roi_depth_range", "manual_set_depth_range", "camera_type", "bg_roi_weights",
    "bg_roi_index", "bg_roi_shape", "bg_roi_dilate",
)
FIND_ROI_ORDER = ("input_file", *_ROI_ORDER)


def _entry(project, config_file, recording):
    """The arena tool's settings for ``recording``, from its entry in the project's session_config.yaml;
    none from a job settings file, which already holds them."""
    if is_job_file(config_file):
        return {}
    try:
        return recording_entry(project, recording)
    except ConfigFileError as error:
        raise click.ClickException(str(error)) from None


@cli.command(
    name="find-roi",
    help="Finds the ROI (the arena) and background to subtract from frames when extracting.",
)
@click_options(RoiOptions, order=FIND_ROI_ORDER, others=_roi_others({
    "input_file": _recording("find-roi", "to find the arena in"),
}))
@click.pass_context
def find_roi(ctx, input_file, output_dir, config_file, **_):
    config = command_settings(ctx, RoiOptions, stage_classes=EXTRACT_STAGE)
    project = command_project(ctx)
    # a --bg-roi-depth-range typed sets --manual-set-depth-range (RoiOptions.resolve)
    opts = RoiOptions.resolve(config=config, entry=_entry(project, config_file, input_file),
                              explicit=explicit_values(ctx, RoiOptions))
    if project is not None:
        project_steps.find_roi(project, input_file, opts, output_dir=output_dir)
    else:
        steps.find_roi(input_file, output_dir, opts)


@cli.command(
    name="download-flip-file",
    help="Downloads Flip-correction model that helps with orienting the mouse during extraction.",
)
@click.argument(
    "config-file",
    type=click.Path(resolve_path=False),
    default="config.yaml",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    show_default=False,
    help="Output directory for downloaded flip file (default: the folder the command runs in)",
)
def download_flip_file(config_file, output_dir):
    if not os.path.isfile(config_file):
        raise click.ClickException(
            f"download-flip-file sets the flip classifier in a project's config.yaml, and there is no "
            f"{os.path.abspath(config_file)}. Run it from the project folder, or name the project's config.yaml "
            "as its argument: `moseq2-extract download-flip-file /path/to/project/config.yaml`. To start a "
            "project here, write its config.yaml first with `moseq2-extract generate-config`."
        )
    answered(project_steps.download_flip_file, config_file, output_dir or os.getcwd(), selection=None)


@cli.command(
    name="generate-config",
    help="Generates a configuration file (config.yaml) that holds editable options for extraction parameters.",
)
@click.option("--output-file", "-o", type=click.Path(), default="config.yaml")
@click.option(
    "--camera-type",
    default="auto",
    type=click.Choice(["auto", "kinect", "azure", "k2"]),
    help="The camera the project's recordings were made with, whose preset settings the file gets: "
    "kinect (a Kinect v2; k2 is the same), or azure (an Azure Kinect or Orbbec Femto). auto takes "
    "azure when every recording in the output file's folder is an .mkv, which only those cameras "
    "write, and kinect otherwise.",
)
def generate_config(output_file, camera_type):

    project_steps.generate_config(output_file, camera_type, overwrite=True)

    print("Successfully generated config file in base directory.")


cli.add_command(print_record_command())


if __name__ == "__main__":
    cli()
