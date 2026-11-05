"""The settings of each extraction step, declared once.

One class per command. Each field carries its default, its command-line flags,
its click type and its help text; ``cli.py`` generates the commands' options
from these classes, and ``steps.py`` takes an instance. The paths a step works
on (the recording or folder, ``--output-dir``, ``--output-file``) and
``--config-file`` are not settings: they are the steps' arguments and plain
options in ``cli.py``.

``RoiOptions`` holds ``find-roi``'s settings, which ``extract`` and
``batch-extract`` share; ``ExtractOptions`` adds the frame cleaning
(``moseq2_common.frames.CleaningOptions``, which flip-classifier training
shares) and the rest of ``extract``'s, and
``BatchExtractOptions`` adds the folder search and the SLURM job settings. The
fields marked ``arena=True`` are the settings the arena tool sets for each
recording (``session_config.ARENA_TOOL_SETTINGS``).

Every class here but the utilities' reads ``config.yaml``'s ``extract``
section (``EXTRACT_STAGE``), and ``camera_type`` from its top level. A range
passed explicitly in ``bg_roi_depth_range`` (typed on the command line, or in a
script's ``explicit``) is used as given: it sets ``manual_set_depth_range``
unless that is passed too.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import click

from moseq2_common.frames import CleaningOptions
from moseq2_common.options import Options, option
from moseq2_common.slurm import SlurmOptions
from moseq2_extract.recordings import DEFAULT_MAX_DEPTH, RECORDING_EXTENSIONS


@dataclass(frozen=True, kw_only=True)
class RoiOptions(Options):
    """``find-roi``: finding the arena and the background."""

    SECTION = "extract"

    progress_bar: bool = option(False, "--progress-bar", "-p", is_flag=True, setting=False,
                                help="Show verbose progress bars.")
    recompute_bg: bool = option(False, "--recompute-bg", help="Overwrite previously computed background image")
    use_plane_bground: bool = option(False, "--use-plane-bground", is_flag=True,
                                     help="Use a plane fit for the background. Useful when mice don't move much")
    noise_tolerance: int = option(30, "--noise-tolerance", type=int, arena=True,
                                  help="Extent of noise to accept during RANSAC Plane ROI computation.")
    erode_iterations: int = option(0, "--erode-iterations", type=int,
                                   help="Number of erosion iterations to decrease bucket floor size.")
    bg_roi_erode: tuple = option((1, 1), "--bg-roi-erode", type=(int, int),
                                 help="Size of cv2 Structure Element to erode roi.")
    dilate_iterations: int = option(1, "--dilate-iterations", type=int, arena=True,
                                    help="Number of dilation iterations to increase bucket floor size.")
    bg_sort_roi_by_position_max_rois: int = option(2, "--bg-sort-roi-by-position-max-rois", type=int,
                                                   help="The number of maximum ROIs sorted by area")
    bg_sort_roi_by_position: bool = option(False, "--bg-sort-roi-by-position", type=bool,
                                           help="Sort ROIs by position")
    bg_roi_fill_holes: bool = option(True, "--bg-roi-fill-holes", type=bool, help="Fill holes in ROI")
    bg_roi_gradient_kernel: int = option(7, "--bg-roi-gradient-kernel", type=int,
                                         help="Kernel size for Sobel gradient filtering")
    bg_roi_gradient_threshold: float = option(3000, "--bg-roi-gradient-threshold", type=float,
                                              help="Gradient must be less than threshold to include points")
    bg_roi_gradient_filter: bool = option(False, "--bg-roi-gradient-filter", type=bool,
                                          help="Use graident filter to exclude walls for detected ROI")
    bg_roi_depth_range: tuple = option((650, 750), "--bg-roi-depth-range", type=(float, float), arena=True,
                                       help="Range to search for floor of arena (in mm)")
    manual_set_depth_range: bool = option(False, "--manual-set-depth-range", is_flag=True, arena=True,
                                          help="Flag to deactivate auto depth range setting.")
    camera_type: str = option(
        "auto", "--camera-type", type=click.Choice(["auto", "kinect", "azure", "realsense"]),
        help="Camera the recordings were made with. It sets bg_roi_weights, pixel_format and "
        "movie_dtype for that camera where they are missing or still at the Kinect v2 values; "
        "a value you changed is kept. auto tells the camera from the file (.dat is a Kinect v2, "
        ".mkv an Azure Kinect or Orbbec Femto, .avi by its frame size). An .mkv recording gets "
        "the Azure settings under kinect too.",
    )
    bg_roi_weights: tuple = option((1, 0.1, 1), "--bg-roi-weights", type=(float, float, float), arena=True,
                                   help="ROI feature weighting (area, extent, dist to center)")
    bg_roi_index: int = option(0, "--bg-roi-index", type=int, arena=True,
                               help="Index of which detected ROI mask to use")
    bg_roi_shape: str = option("ellipse", "--bg-roi-shape", type=str, arena=True,
                               help="Shape to use to detect roi (ellipse or rect)")
    bg_roi_dilate: tuple = option((10, 10), "--bg-roi-dilate", type=(int, int), arena=True,
                                  help="Size of StructuringElement to dilate roi")

    @classmethod
    def resolve(cls, config: Mapping | None = None, entry: Mapping | None = None,
                explicit: Mapping | None = None):
        """``Options.resolve``, with an explicit ``bg_roi_depth_range`` used as given (module docstring)."""
        explicit = dict(explicit or {})
        if "bg_roi_depth_range" in explicit:
            explicit.setdefault("manual_set_depth_range", True)
        return super().resolve(config=config, entry=entry, explicit=explicit)


@dataclass(frozen=True, kw_only=True)
class ExtractOptions(RoiOptions, CleaningOptions):
    """``extract``: everything ``find-roi`` takes, the frame cleaning, and the extraction itself."""

    # as SlurmOptions.cluster_type, which batch-extract takes it from
    cluster_type: str = option("local", "--cluster-type", type=click.Choice(["local", "slurm"]),
                               help="Platform to train models on")
    # reading the recording (moseq2-extract's -t is --temporal-filter-size, so --threads has no short flag)
    mapping: str = option("DEPTH", "-m", "--mapping", type=str,
                          help="Ffprobe stream selection variable. Default: DEPTH")
    threads: int = option(8, "--threads", type=int, help="Number of threads for decoding .avi and .mkv recordings")
    delete: bool = option(False, "--delete", is_flag=True, help="Delete raw file if encoding is sucessful")
    fps: float = option(30, "--fps", type=float, help="Video FPS")
    chunk_size: int = option(3000, "-b", "--chunk-size", type=int, help="Chunk size")
    skip_completed: bool = option(False, "--skip-completed", is_flag=True,
                                  help="Will skip the extraction if it is already completed.")
    compress_threads: int = option(3, "--compress-threads", type=int, help="Number of threads for encoding")
    compress_chunk_size: int = option(3000, "--compress-chunk-size", type=int,
                                      help="Chunk size for .avi compression")
    compress: bool = option(False, "--compress", type=bool, arena=True,
                            help="Convert .dat to .avi after successful extraction")
    frame_trim: tuple = option((0, 0), "--frame-trim", type=(int, int),
                               help="Frames to trim from beginning and end of data")
    model_smoothing_clips: tuple = option((0, 0), "--model-smoothing-clips", type=(float, float),
                                          help="Model smoothing clips")
    angle_hampel_sig: float = option(3, "--angle-hampel-sig", type=float, help="Angle filter sig")
    angle_hampel_span: int = option(0, "--angle-hampel-span", type=int, help="Angle filter span")
    centroid_hampel_sig: float = option(3, "--centroid-hampel-sig", type=float, help="Hampel filter sig")
    centroid_hampel_span: int = option(0, "--centroid-hampel-span", type=int, help="Hampel filter span")
    pixel_format: str = option("gray16le", "--pixel-format", type=str, arena=True,
                               help="Pixel format for reading in .avi and .mkv videos")
    movie_dtype: str = option("<u2", "--movie-dtype", arena=True,
                              help="Data type for raw frames read in for extraction")
    frame_dtype: str = option("uint8", "--frame-dtype", type=click.Choice(["uint8", "uint16"]), arena=True,
                              help="Data type for processed frames")
    write_movie: bool = option(True, "--write-movie", type=bool,
                               help="Write a results output movie including an extracted mouse")
    chunk_overlap: int = option(0, "--chunk-overlap", type=int, arena=True,
                                help="Frames overlapped in each chunk. Useful for cable tracking")
    tracking_model_init: str = option("raw", "--tracking-model-init", type=str,
                                      help="Method for tracking model initialization")
    tracking_model_segment: bool = option(True, "--tracking-model-segment", type=bool,
                                          help="Segment likelihood mask from tracking model")
    tracking_model_ll_clip: float = option(-100, "--tracking-model-ll-clip", type=float,
                                           help="Clip log-likelihoods below this value")
    use_tracking_model: bool = option(
        False, "--use-tracking-model", type=bool, arena=True,
        help="Use an expectation-maximization style model to aid mouse tracking. Useful for data with cables",
    )
    use_cc: bool = option(True, "--use-cc", type=bool, help="Extract features using largest connected components.")
    widen_radius: int = option(0, "--widen-radius", type=int,
                               help="Number of pixels to increase/decrease radius by when graduating bucket walls.")
    graduate_walls: bool = option(
        False, "--graduate-walls", type=bool,
        help="Graduates and dilates the background image to compensate for slanted bucket walls. \\_/",
    )
    flip_classifier_smoothing: int = option(51, "--flip-classifier-smoothing", type=int, arena=True,
                                            help="Number of frames to smooth flip classifier probabilities")
    flip_classifier: str | None = option(
        None, "--flip-classifier",
        help="path to the flip classifier used to properly orient the mouse (.pkl file)",
    )
    compute_raw_scalars: bool = option(False, "--compute-raw-scalars", is_flag=True,
                                       help="Compute scalar values from raw cropped frames.")
    detected_true_depth: str = option(
        "auto", "--detected-true-depth", type=str,
        help='Option to override automatic depth estimation during extraction. \
This is only a debugging parameter, for cases where dilate_iterations > 1, otherwise has no effect. Either "auto" or an int value.',
    )
    max_height: int = option(120, "--max-height", type=int, arena=True,
                             help="Max mouse height threshold from floor (mm)")
    num_frames: int | None = option(None, "--num-frames", "-n", type=int,
                                    help="Number of frames to extract. Will extract full session if set to None.")
    crop_size: tuple = option((80, 80), "--crop-size", "-c", type=(int, int), arena=True,
                              help="Width and height of cropped mouse image")


@dataclass(frozen=True, kw_only=True)
class BatchExtractOptions(ExtractOptions, SlurmOptions):
    """``batch-extract``: ``extract``'s settings, the folder search and the SLURM jobs."""

    extensions: tuple = option(RECORDING_EXTENSIONS, "--extensions", type=str, multiple=True,
                               help="File extension of raw data")
    skip_checks: bool = option(False, "--skip-checks", is_flag=True,
                               help="Flag: include recordings whose extraction already finished")
    max_depth: int = option(DEFAULT_MAX_DEPTH, "--max-depth", type=int,
                            help="How many folders below the input folder to search for recordings")
    extract_out_script: str = option("extract_out.sh", "--extract-out-script", type=click.Path(),
                                     help="Name of bash script file to save extract commands.")
    ncpus: int = option(1, "--ncpus", type=int, help="CPUs each extraction job requests (slurm only)")


@dataclass(frozen=True, kw_only=True)
class AggregateOptions(Options):
    """``aggregate-results``."""

    SECTION = "extract"

    format: str = option("{start_time}_{session_name}_{subject_name}", "--format", "-f", type=str,
                         help="New file name formats from resepective metadata")
    mouse_threshold: float = option(0, "--mouse-threshold", type=float,
                                    help="Threshold value for mean depth to include frames in aggregated results")


@dataclass(frozen=True, kw_only=True)
class ConvertOptions(Options):
    """``convert-raw-to-avi``."""

    mapping: str = option("DEPTH", "-m", "--mapping", type=str,
                          help="Ffprobe stream selection variable. Default: DEPTH")
    threads: int = option(8, "-t", "--threads", type=int, help="Number of threads for encoding")
    delete: bool = option(False, "--delete", is_flag=True, help="Delete raw file if encoding is sucessful")
    fps: float = option(30, "--fps", type=float, help="Video FPS")
    chunk_size: int = option(3000, "-b", "--chunk-size", type=int, help="Chunk size")


@dataclass(frozen=True, kw_only=True)
class CopySliceOptions(ConvertOptions):
    """``copy-slice``."""

    copy_slice: tuple = option((0, 1000), "-c", "--copy-slice", type=(int, int),
                               help="Slice indices used for copy")


#: The classes of the extraction steps, whose fields config.yaml's ``extract`` section may set.
EXTRACT_STAGE = (RoiOptions, ExtractOptions, BatchExtractOptions, AggregateOptions, CleaningOptions)
