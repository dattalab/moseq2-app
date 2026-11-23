"""``moseq2 extract``, ``moseq2 standalone extract`` and the video utilities.

``moseq2 extract [RECORDING ...]`` runs the Extract step in the project in the
working folder (``moseq2_extract.project.extract``): every recording that needs
it, or the recordings named whatever their plan. Settings typed as flags are
saved to the project before the run, as the notebook's Extract step saves its
own. ``moseq2 standalone extract RECORDING ...`` extracts the recordings named
(on the command line, or listed one per line in ``--files-from``) with one set
of settings (``--config-file``'s, then the flags), each into ``proc/`` beside
it, reading and writing no project file (``moseq2_extract.steps.extract``).
``moseq2 run-job`` is what an extraction job on a cluster runs.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import click

from moseq2_app.cli import BUG_LINE, EXIT_BUG, EXIT_EXPECTED, common_options, project_here, run_step, \
    settings_flags, typed_settings
from moseq2_common import capture
from moseq2_common.errors import ExpectedError
from moseq2_common.options import described_argument, read_settings
from moseq2_extract.options import EXTRACT_STAGE, ConvertOptions, CopySliceOptions, ExtractComputing, \
    ExtractOptions, RecordingsOptions

ORDER = 30

#: The extraction settings ``moseq2 extract`` takes, in its help's order: the arena, the extraction, the
#: recordings, then where it runs.
_ARENA = ("use_plane_bground", "noise_tolerance", "erode_iterations", "bg_roi_erode", "dilate_iterations",
          "bg_sort_roi_by_position_max_rois", "bg_sort_roi_by_position", "bg_roi_fill_holes",
          "bg_roi_gradient_kernel", "bg_roi_gradient_threshold", "bg_roi_gradient_filter", "bg_roi_depth_range",
          "manual_set_depth_range", "camera_type", "bg_roi_weights", "bg_roi_index", "bg_roi_shape",
          "bg_roi_dilate")
_EXTRACTION = ("mapping", "delete", "chunk_size", "skip_completed", "compress_chunk_size", "compress", "frame_trim",
               "model_smoothing_clips", "angle_hampel_sig", "angle_hampel_span", "centroid_hampel_sig",
               "centroid_hampel_span", "pixel_format", "movie_dtype", "frame_dtype", "write_movie", "chunk_overlap",
               "temporal_filter_size", "spatial_filter_size", "tail_filter_shape", "tail_filter_size",
               "tail_filter_iters", "cable_filter_size", "cable_filter_shape", "cable_filter_iters",
               "tracking_model_init", "tracking_model_segment", "tracking_model_ll_clip", "use_tracking_model",
               "use_cc", "widen_radius", "graduate_walls", "flip_classifier_smoothing", "flip_classifier",
               "compute_raw_scalars", "detected_true_depth", "max_height", "min_height", "num_frames", "crop_size")
_RECORDINGS = ("extensions", "max_depth")
_COMPUTING = ("cluster_type", "cores", "partition", "memory", "wall_time", "prefix")
#: The video utilities' settings, in their help's order.
_AVI_ORDER = ("mapping", "threads", "delete", "fps", "chunk_size")
#: The one-run switches ``moseq2 standalone extract`` adds.
_SWITCHES = ("progress_bar", "threads", "compress_threads")

#: What ``moseq2 extract`` says when Ctrl-C stops it (W9).
STOPPED_LINE = ("Stopped. Recordings that hadn't finished keep what they had before; run `moseq2 extract` to "
                "finish them.")


#: The classes whose settings ``moseq2 extract`` takes.
_CLASSES = (ExtractOptions, RecordingsOptions, ExtractComputing)


def _end(failed: dict) -> None:
    """Exit as a run with ``failed`` recordings ends: 3 when any failed with a bug (each traceback shown,
    then the bug line), 1 when every failure was an expected one; nothing when none failed."""
    if not failed:
        return
    bugs = {key: error for key, error in failed.items() if not isinstance(error, ExpectedError)}
    terminal = capture.terminal()
    for key, error in bugs.items():
        click.echo(f"\n{key}:", file=terminal)
        traceback.print_exception(type(error), error, error.__traceback__, file=terminal)
    if bugs:
        click.echo(BUG_LINE, file=terminal)
        raise click.exceptions.Exit(EXIT_BUG)
    raise click.exceptions.Exit(EXIT_EXPECTED)


@click.command("extract", short_help="Extract the recordings that need it: new ones, and ones whose settings changed.")
@click.argument("recordings", nargs=-1, type=click.Path(exists=True), metavar="[RECORDING]...")
@settings_flags(*_CLASSES, names=(*_ARENA, *_EXTRACTION, *_RECORDINGS, *_COMPUTING))
@common_options
@click.pass_context
def extract(ctx, recordings, quiet, debug, **_):
    """Extract the project's recordings: the mouse in every frame, centered and facing the same way.

    Run it in the project folder. Every included recording that is new, or whose settings or arena changed
    since it was extracted, is extracted; one named as RECORDING is extracted whatever its state. Settings
    typed as options are saved to the project first: arena settings typed with recordings named go to those
    recordings, everything else to the project's settings.
    """
    from moseq2_extract import project as extraction

    settings = typed_settings(ctx, _CLASSES)

    def work(reporter):
        project = project_here("extract")
        try:
            return extraction.extract(project, recordings=list(recordings) or None, settings=settings,
                                      reporter=reporter)
        except KeyboardInterrupt:
            click.echo(STOPPED_LINE, file=capture.terminal())
            raise click.exceptions.Exit(EXIT_EXPECTED) from None

    run = run_step(work, quiet=quiet, debug=debug)
    _end(run.failed)


W22_FIX = ("leave out --output-dir to write each recording's outputs into proc/ beside it, or extract one recording "
           "per command.")


def files_from(list_file: str | Path) -> list[str]:
    """The paths ``list_file`` names, one per line; ``#`` comments and blank lines are ignored, and a relative
    path is relative to the list's folder. A path that does not exist stops it (``ExpectedError``, W23)."""
    list_file = Path(list_file)
    paths = []
    for number, line in enumerate(list_file.read_text().splitlines(), start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        path = Path(text) if Path(text).is_absolute() else list_file.parent / text
        if not path.exists():
            raise ExpectedError(f"{list_file} names {text} on line {number}, which does not exist.",
                                fix="correct or delete that line.")
        paths.append(str(path))
    return paths


@click.command("extract", short_help="Extract the recordings named, each into proc/ beside it.")
@click.argument("recordings", nargs=-1, type=click.Path(exists=True), metavar="[RECORDING]...")
@click.option("--files-from", "list_file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="A text file listing recordings to extract, one path per line; lines starting with # and blank "
                   "lines are ignored, and a relative path is relative to the file's folder.")
@click.option("--config-file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="A settings file in config.yaml's shape; its extract section and camera_type are the settings, "
                   "under any typed as options.")
@click.option("--output-dir", type=click.Path(), default=None,
              help="The folder for the outputs of the one recording named, relative to the recording's folder.")
@settings_flags(ExtractOptions, names=(*_SWITCHES, *_ARENA, *_EXTRACTION))
@common_options
@click.pass_context
def standalone_extract(ctx, recordings, list_file, config_file, output_dir, quiet, debug, **_):
    """Extract each recording named, with one set of settings, into proc/ beside it.

    The settings are --config-file's, then the options typed. No project file is read or written.
    """
    from moseq2_extract import steps
    from moseq2_extract.helpers.extract import extraction_summary

    typed = typed_settings(ctx, (ExtractOptions,))
    named = list(recordings)

    def work(reporter):
        if list_file:
            named.extend(files_from(list_file))
        if not named:
            raise click.UsageError("Name at least one recording, or a list of them with --files-from.", ctx=ctx)
        if output_dir is not None and len(named) > 1:
            raise ExpectedError(f"`--output-dir` names one recording's output folder, and {len(named)} recordings "
                                "were given.", fix=W22_FIX)
        config = read_settings(config_file, ExtractOptions, stage_classes=EXTRACT_STAGE) if config_file else {}
        opts = ExtractOptions.resolve(config=config, explicit=typed)
        failed = {}
        for recording in named:
            try:
                steps.extract(recording, output_dir, opts, reporter=reporter)
            except Exception as error:  # noqa: BLE001 -- recorded, and the rest are extracted
                reporter.status(f"Unexpected error: {error}")
                reporter.status(f"could not extract {recording}")
                failed[recording] = error
        reporter.status(extraction_summary(len(named), 0, list(failed.items())))
        return failed

    _end(run_step(work, quiet=quiet, debug=debug))


@click.command("run-job")
@click.argument("job_file", type=click.Path(exists=True, dir_okay=False))
@common_options
def run_job(job_file, quiet, debug):
    """Extract the recording a cluster job's settings file names, as the job does."""
    from moseq2_extract import project as extraction

    run_step(lambda reporter: extraction.run_job(job_file, reporter=reporter), quiet=quiet, debug=debug)


@click.command("convert-raw-to-avi", short_help="Compress a raw .dat recording into an .avi about 8 times smaller.")
@described_argument(
    "input-file", type=click.Path(exists=True, resolve_path=False),
    missing="the raw depth recording to compress, a depth.dat file, which in a project sits in its session's "
            "folder. Give its path, for example `moseq2 standalone convert-raw-to-avi session_1/depth.dat` from the "
            "project folder.")
@settings_flags(ConvertOptions, names=_AVI_ORDER)
@click.option("-o", "--output-file", type=click.Path(), default=None,
              help="The .avi to write (default: the recording's name with .avi, beside it).")
@common_options
@click.pass_context
def convert_raw_to_avi(ctx, input_file, output_file, quiet, debug, **_):
    """Compress a raw depth recording (.dat) into an .avi about 8 times smaller, checked frame by frame."""
    from moseq2_extract.helpers.wrappers import convert_raw_to_avi_wrapper

    opts = ConvertOptions.resolve(explicit=typed_settings(ctx, (ConvertOptions,)))
    run_step(lambda reporter: convert_raw_to_avi_wrapper(input_file, output_file, opts.chunk_size, opts.fps,
                                                         opts.delete, opts.threads, opts.mapping),
             quiet=quiet, debug=debug)


@click.command("copy-slice", short_help="Copy a range of frames of a recording into a new video file.")
@described_argument(
    "input-file", type=click.Path(exists=True, resolve_path=False),
    missing="the depth recording to copy a segment of: a .dat, .avi, .mkv or .tar.gz file, which in a project sits "
            "in its session's folder. Give its path, for example `moseq2 standalone copy-slice session_1/depth.avi` "
            "from the project folder.")
@settings_flags(CopySliceOptions, names=(*_AVI_ORDER, "copy_slice"))
@click.option("-o", "--output-file", type=click.Path(), default=None,
              help="The file to write (default: the recording's name with .avi, beside it).")
@common_options
@click.pass_context
def copy_slice(ctx, input_file, output_file, quiet, debug, **_):
    """Copy a range of frames of a depth recording into a new video file."""
    from moseq2_extract.helpers.wrappers import copy_slice_wrapper

    opts = CopySliceOptions.resolve(explicit=typed_settings(ctx, (CopySliceOptions,)))
    run_step(lambda reporter: copy_slice_wrapper(input_file, output_file, opts.copy_slice, opts.chunk_size,
                                                 opts.fps, opts.delete, opts.threads, opts.mapping),
             quiet=quiet, debug=debug)


STEPS = (extract,)
STANDALONE = (standalone_extract, convert_raw_to_avi, copy_slice)
HIDDEN = (run_job,)
