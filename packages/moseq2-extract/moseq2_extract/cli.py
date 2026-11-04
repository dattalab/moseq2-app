"""
CLI for extracting the depth data.

Each command's settings are generated from its options class in
``moseq2_extract.options``; the command reads ``--config-file``, resolves the
settings and calls its step in ``moseq2_extract.steps`` (or
``moseq2_extract.setup``).
"""

import os
from dataclasses import replace

import click

from moseq2_common.options import ConfigFileError, click_options, explicit_values, read_config
from moseq2_extract import setup, steps
from moseq2_extract.helpers.wrappers import (
    convert_raw_to_avi_wrapper,
    copy_slice_wrapper,
    generate_index_from_agg_res_wrapper,
    generate_index_wrapper,
)
from moseq2_extract.options import (
    AggregateOptions,
    BatchExtractOptions,
    ConvertOptions,
    CopySliceOptions,
    ExtractOptions,
    RoiOptions,
)


@click.group(context_settings={"show_default": True})
@click.version_option()
def cli():
    pass


def _roi_others(argument):
    """The options of find-roi, extract and batch-extract that are not settings, and their argument."""
    return {
        **argument,
        "config_file": click.option("--config-file", type=click.Path(), default=None),
        "output_dir": click.option("--output-dir", default="proc",
                                   help="Output directory to save the results h5 file"),
    }


_OUTPUT_FILE = click.option("-o", "--output-file", type=click.Path(), default=None, help="Path to output file")

# The help order of each command, as 1.5.0 lists its options.
_ROI_ORDER = (
    "progress_bar", "config_file", "recompute_bg", "use_plane_bground", "output_dir", "noise_tolerance",
    "erode_iterations", "bg_roi_erode", "dilate_iterations", "bg_sort_roi_by_position_max_rois",
    "bg_sort_roi_by_position", "bg_roi_fill_holes", "bg_roi_gradient_kernel", "bg_roi_gradient_threshold",
    "bg_roi_gradient_filter", "bg_roi_depth_range", "manual_set_depth_range", "camera_type", "bg_roi_weights",
    "bg_roi_index", "bg_roi_shape", "bg_roi_dilate",
)
_EXTRACTION_ORDER = (
    "mapping", "threads", "delete", "fps", "chunk_size", "output_file", "skip_completed", "compress_threads",
    "compress_chunk_size", "compress", "frame_trim", "model_smoothing_clips", "angle_hampel_sig",
    "angle_hampel_span", "centroid_hampel_sig", "centroid_hampel_span", "pixel_format", "movie_dtype",
    "frame_dtype", "write_movie", "chunk_overlap", "temporal_filter_size", "spatial_filter_size",
    "tail_filter_shape", "tail_filter_size", "tail_filter_iters", "cable_filter_size", "cable_filter_shape",
    "cable_filter_iters", "tracking_model_init", "tracking_model_segment", "tracking_model_ll_clip",
    "tracking_model_mask_threshold", "tracking_model_ll_threshold", "use_tracking_model", "use_cc",
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


def _read_config_file(config_file):
    """``config.yaml``'s settings, or None without ``--config-file``; a file that cannot be read stops the command."""
    if not config_file:
        return None
    try:
        return read_config(config_file)
    except ConfigFileError as error:
        raise click.ClickException(str(error)) from None


def shim_click_defaults(ctx, cls, config):
    """Resolve ``cls`` from click's defaults, ``config`` and the typed options (removing PR to be named).

    click converts each default through its option's type (``--fps``'s 30
    becomes 30.0, ``--bg-roi-weights``' (1, 0.1, 1) becomes (1.0, 0.1, 1.0)),
    and 1.5.0's command line ran on those values where the notebook ran on the
    declared ones. A null in ``config`` keeps the default.
    """
    defaults = {name: ctx.params[name] for name in cls().as_dict()}
    settings = {name: value for name, value in (config or {}).items() if value is not None}
    return cls.resolve(config={**defaults, **settings}, explicit=explicit_values(ctx, cls))


def _typed_depth_range_is_manual(ctx, opts):
    """Use a --bg-roi-depth-range typed on the command line as given, rather than
    replacing it with the detected range. A range from the config file still
    follows manual_set_depth_range."""
    if ctx.get_parameter_source("bg_roi_depth_range") is click.core.ParameterSource.COMMANDLINE:
        return replace(opts, manual_set_depth_range=True)
    return opts


@cli.command(
    name="find-roi",
    help="Finds the ROI (the arena) and background to subtract from frames when extracting.",
)
@click_options(RoiOptions, order=FIND_ROI_ORDER, others=_roi_others({
    "input_file": click.argument("input-file", type=click.Path(exists=True)),
}))
@click.pass_context
def find_roi(ctx, input_file, output_dir, config_file, **_):
    config = _read_config_file(config_file)
    opts = _typed_depth_range_is_manual(ctx, shim_click_defaults(ctx, RoiOptions, config))
    steps.find_roi(input_file, output_dir, opts)


@cli.command(
    name="extract",
    help="Processes raw input depth recordings to output a cropped and oriented"
    "video of the mouse and saves the output+metadata to h5 files in the given output directory.",
)
@click_options(ExtractOptions, order=EXTRACT_ORDER, others={
    **_roi_others({"input_file": click.argument("input-file", type=click.Path(exists=True, resolve_path=False))}),
    "output_file": _OUTPUT_FILE,
})
@click.pass_context
def extract(ctx, input_file, output_dir, config_file, output_file, **_):
    config = _read_config_file(config_file)
    opts = _typed_depth_range_is_manual(ctx, shim_click_defaults(ctx, ExtractOptions, config))
    # extract FILE reads config.yaml only, never the recording's session_config.yaml entry (L11, removed by PR 3.3).
    # The two paths 1.5.0 recorded among the extraction's parameters (R2, removed by PR 4.1):
    steps.extract(input_file, output_dir, opts,
                  shim_flat_config={"config_file": config_file, "output_file": output_file})


def _typed_extract_options(ctx):
    """The extraction options typed on batch-extract's command line, to apply over
    the config file and each recording's session config. Only options extract
    itself takes are kept; num_frames and skip_completed are passed separately."""
    passed_separately = {"input_file", "num_frames", "skip_completed", "config_file", "cluster_type"}
    names = {p.name for p in extract.params} - passed_separately
    typed = {
        k: v
        for k, v in ctx.params.items()
        if k in names and ctx.get_parameter_source(k) is click.core.ParameterSource.COMMANDLINE
    }
    if "bg_roi_depth_range" in typed:
        typed["manual_set_depth_range"] = True
    return typed


@cli.command(
    name="batch-extract",
    help="Batch processes " "all the raw depth recordings located in the input folder.",
)
@click_options(BatchExtractOptions, order=BATCH_EXTRACT_ORDER, others={
    **_roi_others({"input_folder": click.argument("input-folder",
                                                  type=click.Path(exists=True, resolve_path=False))}),
    "output_file": _OUTPUT_FILE,
})
@click.pass_context
def batch_extract(ctx, input_folder, config_file, **_):
    if not config_file:
        # Add message to tell the users to specify a config file
        print(
            "Command not run. Please specified a config file using --config-file flag."
        )
        return
    config = _read_config_file(config_file)
    opts = shim_click_defaults(ctx, BatchExtractOptions, config)
    typed = _typed_extract_options(ctx)
    result = steps.batch_extract(
        input_folder, opts, config_file=config_file,
        # an --output-dir not typed is config.yaml's for a local extraction, proc for the search and SLURM jobs
        output_dir=typed.pop("output_dir", None),
        explicit=typed,
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
    default="k2",
    type=str,
    help="specify the camera type (k2 or azure), default is k2",
)
def generate_config(output_file, camera_type):

    setup.generate_config(output_file, camera_type, overwrite=True)

    print("Successfully generated config file in base directory.")


@cli.command(
    name="generate-index",
    help="Generates an index file (moseq2-index.yaml) that contains all extracted session metadata.",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=os.getcwd(),
    help="Directory to find h5 files",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "moseq2-index.yaml"),
    help="Location for storing index",
)
def generate_index(input_dir, output_file):

    output_file = generate_index_wrapper(input_dir, output_file)

    if output_file is not None:
        print(f"Index file: {output_file} was successfully generated.")


@cli.command(
    name="aggregate-results",
    help="Copies all extracted results (h5, yaml, mp4) files from all extracted sessions to a new directory for modeling and analysis",
)
@click_options(AggregateOptions, order=("input_dir", "format", "output_dir", "mouse_threshold"), others={
    "input_dir": click.option("--input-dir", "-i", type=click.Path(), default=os.getcwd(),
                              help="Directory to find h5 files"),
    # the working directory's aggregate_results, where the notebook's is the project's (PR 4.5)
    "output_dir": click.option("--output-dir", "-o", type=click.Path(),
                               default=os.path.join(os.getcwd(), "aggregate_results/"),
                               help="Location for storing all results together"),
})
@click.pass_context
def aggregate_extract_results(ctx, input_dir, output_dir, **_):
    steps.aggregate_results(input_dir, output_dir, shim_click_defaults(ctx, AggregateOptions, None))


@cli.command(
    name="agg-to-index",
    help="Generate an index file from aggregated results with default as group names",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "aggregate_results"),
    help="Directory for aggregated results folder",
)
def agg_to_index(input_dir):

    generate_index_from_agg_res_wrapper(input_dir)


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
    opts = shim_click_defaults(ctx, ConvertOptions, None)
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
    opts = shim_click_defaults(ctx, CopySliceOptions, None)
    copy_slice_wrapper(
        input_file, output_file, opts.copy_slice, opts.chunk_size, opts.fps, opts.delete, opts.threads,
        opts.mapping
    )


if __name__ == "__main__":
    cli()
