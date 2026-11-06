"""
CLI for extracting the depth data.

Each command's settings are generated from its options class in
``moseq2_extract.options``; the command reads the ``extract`` section of its
project's ``config.yaml`` (``--config-file``, or the working folder's), resolves
the settings and calls its step in ``moseq2_extract.steps`` (or
``moseq2_extract.setup``). ``find-roi`` and ``extract`` lay the arena tool's
entry for the recording in the project's ``session_config.yaml`` over
``config.yaml``, and write into the recording's session folder in the project's
``aggregate_results/`` (``proc/`` beside the recording outside a project).
"""

import os

import click
from click.core import ParameterSource

from moseq2_common.options import (
    ConfigFileError,
    ProjectGroup,
    click_options,
    command_project,
    command_settings,
    explicit_values,
)
from moseq2_common.project import find_project, is_job_file
from moseq2_common.records import print_record_command
from moseq2_extract import setup, steps
from moseq2_extract.helpers.wrappers import (
    convert_raw_to_avi_wrapper,
    copy_slice_wrapper,
)
from moseq2_extract.options import (
    EXTRACT_STAGE,
    AggregateOptions,
    BatchExtractOptions,
    ConvertOptions,
    CopySliceOptions,
    ExtractOptions,
    RoiOptions,
)
from moseq2_extract.session_config import recording_entry


@click.group(cls=ProjectGroup, context_settings={"show_default": True})
@click.version_option()
def cli():
    pass


def _roi_others(argument):
    """The options of find-roi, extract and batch-extract that are not settings, and their argument."""
    return {
        **argument,
        "config_file": click.option("--config-file", type=click.Path(), default=None),
        "output_dir": click.option("--output-dir", default=None, show_default=False,
                                   help="Folder for the background and arena images (default: the recording's "
                                        "session folder in the project's aggregate_results, or proc beside the "
                                        "recording outside a project)"),
    }


_OUTPUT_FILE = click.option("-o", "--output-file", type=click.Path(), default=None, help="Path to output file")

# The help order of each command, as 1.5.0 lists its options.
_ROI_ORDER = (
    "progress_bar", "config_file", "use_plane_bground", "output_dir", "noise_tolerance",
    "erode_iterations", "bg_roi_erode", "dilate_iterations", "bg_sort_roi_by_position_max_rois",
    "bg_sort_roi_by_position", "bg_roi_fill_holes", "bg_roi_gradient_kernel", "bg_roi_gradient_threshold",
    "bg_roi_gradient_filter", "bg_roi_depth_range", "manual_set_depth_range", "camera_type", "bg_roi_weights",
    "bg_roi_index", "bg_roi_shape", "bg_roi_dilate",
)
_EXTRACTION_ORDER = (
    "mapping", "threads", "delete", "chunk_size", "output_file", "skip_completed", "compress_threads",
    "compress_chunk_size", "compress", "frame_trim", "model_smoothing_clips", "angle_hampel_sig",
    "angle_hampel_span", "centroid_hampel_sig", "centroid_hampel_span", "pixel_format", "movie_dtype",
    "frame_dtype", "write_movie", "chunk_overlap", "temporal_filter_size", "spatial_filter_size",
    "tail_filter_shape", "tail_filter_size", "tail_filter_iters", "cable_filter_size", "cable_filter_shape",
    "cable_filter_iters", "tracking_model_init", "tracking_model_segment", "tracking_model_ll_clip",
    "use_tracking_model", "use_cc",
    "widen_radius", "graduate_walls", "flip_classifier_smoothing", "flip_classifier", "compute_raw_scalars",
    "detected_true_depth", "max_height", "min_height", "num_frames", "crop_size",
)
FIND_ROI_ORDER = ("input_file", *_ROI_ORDER)
EXTRACT_ORDER = ("input_file", "cluster_type", *_ROI_ORDER, *_EXTRACTION_ORDER)
BATCH_EXTRACT_ORDER = (
    "input_folder", *_ROI_ORDER, *_EXTRACTION_ORDER, "extensions", "skip_checks", "max_depth",
    "extract_out_script", "cluster_type", "prefix", "ncpus", "memory", "wall_time", "partition", "get_cmd",
    "run_cmd",
)
_AVI_ORDER = ("input_file", "mapping", "threads", "delete", "fps", "chunk_size", "output_file")


def _project_config(config_file):
    """``--config-file``, or the config.yaml of the project the command runs in; None outside a project."""
    if config_file:
        return config_file
    project = find_project()
    return str(project.config_file) if project is not None else None


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
    "input_file": click.argument("input-file", type=click.Path(exists=True)),
}))
@click.pass_context
def find_roi(ctx, input_file, output_dir, config_file, **_):
    config = command_settings(ctx, RoiOptions, stage_classes=EXTRACT_STAGE)
    project = command_project(ctx)
    # a --bg-roi-depth-range typed sets --manual-set-depth-range (RoiOptions.resolve)
    opts = RoiOptions.resolve(config=config, entry=_entry(project, config_file, input_file),
                              explicit=explicit_values(ctx, RoiOptions))
    steps.find_roi(input_file, output_dir, opts, project=project)


@cli.command(
    name="extract",
    help="Processes raw input depth recordings to output a cropped and oriented "
    "video of the mouse and saves the output+metadata to h5 files in the given output directory. "
    "With --config-file, the settings the arena tool saved for this recording in the project's "
    "session_config.yaml are used over config.yaml's; options typed on the command line take "
    "precedence over both.",
)
@click_options(ExtractOptions, order=EXTRACT_ORDER, others={
    **_roi_others({"input_file": click.argument("input-file", type=click.Path(exists=True, resolve_path=False))}),
    "output_dir": click.option("--output-dir", default=None, show_default=False,
                               help="Folder for the recording's outputs, relative to the recording's folder (default: its "
                                    "session's folder in the project's aggregate_results, or proc beside the "
                                    "recording outside a project)"),
    "output_file": _OUTPUT_FILE,
})
@click.pass_context
def extract(ctx, input_file, output_dir, config_file, output_file, **_):
    config = command_settings(ctx, ExtractOptions, stage_classes=EXTRACT_STAGE)
    project = command_project(ctx)
    opts = ExtractOptions.resolve(config=config, entry=_entry(project, config_file, input_file),
                                  explicit=explicit_values(ctx, ExtractOptions))
    steps.extract(input_file, output_dir, opts, project=project)


#: ExtractOptions fields batch-extract hands the step in its options rather than as typed values.
_IN_BATCH_OPTIONS = ("num_frames", "skip_completed", "cluster_type")


def _typed_extract_options(ctx):
    """The extraction options typed on batch-extract's command line, to apply over
    the config file and each recording's session config: the ``ExtractOptions``
    fields but ``_IN_BATCH_OPTIONS``, and ``--output-dir`` and ``--output-file``,
    in the command's order."""
    typed = set(explicit_values(ctx, ExtractOptions)) - set(_IN_BATCH_OPTIONS)
    typed |= {name for name in ("output_dir", "output_file")
              if ctx.get_parameter_source(name) is ParameterSource.COMMANDLINE}
    return {name: value for name, value in ctx.params.items() if name in typed}


@cli.command(
    name="batch-extract",
    help="Batch processes all the raw depth recordings located in the input folder. With --cluster-type "
    "slurm, each recording's settings are written to extract-job.yaml in its session folder, which its "
    "job's command passes to extract --config-file, so a later edit to config.yaml does not change a "
    "queued job.",
)
@click_options(BatchExtractOptions, order=BATCH_EXTRACT_ORDER, others={
    **_roi_others({"input_folder": click.argument("input-folder",
                                                  type=click.Path(exists=True, resolve_path=False))}),
    "output_dir": click.option("--output-dir", default=None, show_default=False,
                               help="Folder holding each recording's session folder (default: the project's "
                                    "aggregate_results)"),
    "output_file": _OUTPUT_FILE,
})
@click.pass_context
def batch_extract(ctx, input_folder, config_file, **_):
    config_file = _project_config(config_file)
    if not config_file:
        raise click.ClickException(
            f"batch-extract needs a project's config.yaml, and there is none in {os.getcwd()}. Run it from the "
            "project folder, pass --config-file /path/to/project/config.yaml, or write a config.yaml here first "
            "with `moseq2-extract generate-config`."
        )
    config = command_settings(ctx, BatchExtractOptions, stage_classes=EXTRACT_STAGE)
    opts = BatchExtractOptions.resolve(config=config, explicit=explicit_values(ctx, BatchExtractOptions))
    typed = _typed_extract_options(ctx)
    # --output-file names no file in a batch: each recording's is named from its arena mask
    typed.pop("output_file", None)
    result = steps.batch_extract(
        input_folder, opts, config_file=config_file,
        # an --output-dir not typed is the project's aggregate_results
        output_dir=typed.pop("output_dir", None),
        explicit=typed,
        project=find_project(config_file),
    )
    # the summary the local extraction printed names each failure
    if result.failures:
        ctx.exit(1)


@cli.command(
    name="download-flip-file",
    help="Downloads Flip-correction model that helps with orienting the mouse during extraction.",
)
@click.argument(
    "config-file",
    type=click.Path(exists=True, resolve_path=False),
    default="config.yaml",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=os.getcwd(),
    help="Output directory for downloaded flip flie",
)
def download_flip_file(config_file, output_dir):

    setup.download_flip_file(config_file, output_dir, None)


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

    setup.generate_config(output_file, camera_type, overwrite=True)

    print("Successfully generated config file in base directory.")


@cli.command(
    name="aggregate-results",
    help="Lists every extracted recording in the project as one of its sessions in moseq2-project.yaml, for "
    "modeling and analysis. Each session's files are already in its folder in aggregate_results, where "
    "extraction wrote them. Run it in the project folder.",
)
@click_options(AggregateOptions, order=("input_dir", "output_dir", "mouse_threshold"), others={
    "input_dir": click.option("--input-dir", "-i", type=click.Path(), default=None, show_default=False,
                              help="Folder searched for the recordings (default: the project folder)"),
    "output_dir": click.option("--output-dir", "-o", type=click.Path(), default=None, show_default=False,
                               help="Folder holding the session folders, when extraction was given one with "
                                    "--output-dir (default: the project's aggregate_results)"),
})
@click.pass_context
def aggregate_extract_results(ctx, input_dir, output_dir, **_):
    project = find_project()
    if project is None:
        raise click.ClickException(
            f"There is no config.yaml in {os.getcwd()}, so this is not a project folder, and aggregate-results "
            "lists the sessions in a project's moseq2-project.yaml. Run it in the project folder, the one holding "
            "config.yaml and the recordings."
        )
    config = command_settings(ctx, AggregateOptions, stage_classes=EXTRACT_STAGE)
    steps.aggregate_results(project, AggregateOptions.resolve(config=config,
                                                              explicit=explicit_values(ctx, AggregateOptions)),
                            input_dir=input_dir, output_dir=output_dir)


@cli.command(
    name="convert-raw-to-avi",
    help="Loss less compresses a raw depth file (dat) into an avi file that is 8x smaller.",
)
@click_options(ConvertOptions, order=_AVI_ORDER, others={
    "input_file": click.argument("input-file", type=click.Path(exists=True, resolve_path=False)),
    "output_file": _OUTPUT_FILE,
})
@click.pass_context
def convert_raw_to_avi(ctx, input_file, output_file, **_):
    opts = ConvertOptions.resolve(explicit=explicit_values(ctx, ConvertOptions))
    convert_raw_to_avi_wrapper(
        input_file, output_file, opts.chunk_size, opts.fps, opts.delete, opts.threads, opts.mapping
    )


@cli.command(
    name="copy-slice",
    help="Copies a segment of an input depth recording into a new video file.",
)
@click_options(CopySliceOptions, order=(*_AVI_ORDER, "copy_slice"), others={
    "input_file": click.argument("input-file", type=click.Path(exists=True, resolve_path=False)),
    "output_file": _OUTPUT_FILE,
})
@click.pass_context
def copy_slice(ctx, input_file, output_file, **_):
    opts = CopySliceOptions.resolve(explicit=explicit_values(ctx, CopySliceOptions))
    copy_slice_wrapper(
        input_file, output_file, opts.copy_slice, opts.chunk_size, opts.fps, opts.delete, opts.threads,
        opts.mapping
    )


cli.add_command(print_record_command())


if __name__ == "__main__":
    cli()
