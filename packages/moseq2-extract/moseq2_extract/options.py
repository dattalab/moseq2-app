"""The settings of each extraction step, declared once.

One class per step. Each field carries its default, its command-line flags, its
click type, its one plain-language help text and, for a setting a step's GUI
shows, its control (``moseq2_common.options.Control``); the command line, the
GUI and the settings docs are all made from these. The paths a step works on
(the recordings, ``--output-dir``, ``--output-file``) and ``--config-file`` are
not settings: they are the steps' arguments.

- ``CameraOptions``: the camera, ``camera_type``, read from ``config.yaml``'s top
  level by every class that inherits it.
- ``ProjectOptions``: the project's own settings, at ``config.yaml``'s top level
  (``SECTION`` ``""``): the camera, which ``animals`` the recordings show (it
  picks head/tail detection, ``moseq2_extract.project.head_tail_detection``) and
  which recordings every step after extraction leaves out (``exclude``).
- ``RoiOptions``: finding the arena and the background (``find-roi``).
- ``ExtractOptions``: everything ``find-roi`` takes, the frame cleaning
  (``moseq2_common.frames.CleaningOptions``, which flip-classifier training
  shares) and the extraction itself. ``threads`` and ``compress_threads`` are
  one-run switches: a project's extraction sets both from where it runs
  (``ExtractComputing``), and ``moseq2 standalone extract`` takes them as flags.
- ``RecordingsOptions``: which files in the project folder are recordings.
- ``DiagnosticsOptions``: the mouse height under which extraction diagnostics
  takes a recording's first frames to hold no mouse.
- ``ExtractComputing``: where extraction runs, ``config.yaml``'s
  ``computing.extract`` block (``moseq2_common.computing``).
- ``ConvertOptions``, ``CopySliceOptions``: the video utilities, which read no
  settings file.

The fields marked ``arena=True`` are the settings the arena tool sets for each
recording (``session_config.ARENA_TOOL_SETTINGS``). Every class here but the
utilities' and the computing block reads ``config.yaml``'s ``extract`` section
(``EXTRACT_STAGE``), and ``camera_type`` from its top level. A range passed
explicitly in ``bg_roi_depth_range`` (typed on the command line, or in a
script's ``explicit``) is used as given: it sets ``manual_set_depth_range``
unless that is passed too.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import click

from moseq2_common.computing import ComputingOptions
from moseq2_common.frames import CleaningOptions
from moseq2_common.options import Control, Options, option
from moseq2_extract.recordings import DEFAULT_MAX_DEPTH, RECORDING_EXTENSIONS

#: The animals a project's recordings can show, as Setup offers them.
ANIMALS = (("c57", "Adult C57 mice"), ("fiber", "Large mice with fiber implants"),
           ("inscopix", "Mice with Inscopix cables"), ("other", "Other"))


@dataclass(frozen=True, kw_only=True)
class CameraOptions(Options):
    """The camera the recordings were made with, at ``config.yaml``'s top level."""

    SECTION = ""

    camera_type: str = option(
        "auto", "--camera-type", type=click.Choice(["auto", "kinect", "azure", "realsense"]),
        help="The camera the recordings were made with: kinect (a Kinect v2), azure (an Azure Kinect or Orbbec "
             "Femto), realsense, or auto to tell it from each recording (.dat is a Kinect v2, .mkv an Azure "
             "Kinect or Orbbec Femto, .avi by its frame size). It sets how the depth frames are read wherever "
             "those settings are still at the Kinect v2 values; a value you changed is kept. An .mkv recording "
             "is read as an Azure one under kinect too.",
    )


@dataclass(frozen=True, kw_only=True)
class ProjectOptions(CameraOptions):
    """The project's own settings, at ``config.yaml``'s top level."""

    SECTION = ""

    animals: str = option(
        "c57", "--animals", type=click.Choice([value for value, _ in ANIMALS]),
        help="The animals in the recordings, which decides how extraction tells the mouse's head from its "
             "tail: c57 (adult C57 mice), fiber (large mice with fiber implants), inscopix (mice with Inscopix "
             "cables) or other (detected as adult C57 mice). A head/tail classifier you trained, kept in the "
             "project folder, is used in its place.",
        control=Control("choice", "Animals", choices=ANIMALS,
                        help="The animals in your recordings, which decides how the mouse's head is told from its "
                             "tail."))
    exclude: tuple = option(
        (), "--exclude", type=str, multiple=True,
        help="Recordings to leave out of every step after extraction, each by its path in the project folder "
             "(such as cohort1/mouse_1/depth.avi).")


@dataclass(frozen=True, kw_only=True)
class RoiOptions(CameraOptions):
    """``find-roi``: finding the arena and the background."""

    SECTION = "extract"

    progress_bar: bool = option(False, "--progress-bar", "-p", is_flag=True, setting=False,
                                help="Show a progress bar for each stage of finding the arena and extracting.")
    use_plane_bground: bool = option(
        False, "--use-plane-bground", is_flag=True,
        help="Take the arena floor to be a flat plane fitted to the depth, instead of the depth the recording "
             "shows when the mouse is elsewhere. Helps when the mouse barely moves.")
    noise_tolerance: int = option(30, "--noise-tolerance", type=int, arena=True,
                                  help="How far, in mm, a point may lie from the fitted floor plane and still count "
                                       "as floor.")
    erode_iterations: int = option(0, "--erode-iterations", type=int,
                                   help="How many times to shrink the arena from its edge, to keep the walls out.")
    bg_roi_erode: tuple = option((1, 1), "--bg-roi-erode", type=(int, int),
                                 help="The width and height, in pixels, of each step that shrinks the arena.")
    dilate_iterations: int = option(1, "--dilate-iterations", type=int, arena=True,
                                    help="How many times to grow the arena from its edge, to take in more floor.")
    bg_sort_roi_by_position_max_rois: int = option(
        2, "--bg-sort-roi-by-position-max-rois", type=int,
        help="With the candidate arenas numbered by position, how many of the largest are numbered.")
    bg_sort_roi_by_position: bool = option(
        False, "--bg-sort-roi-by-position", type=bool,
        help="Number the candidate arenas by where they are in the frame, instead of by how well they match.")
    bg_roi_fill_holes: bool = option(True, "--bg-roi-fill-holes", type=bool,
                                     help="Fill any holes inside the arena.")
    bg_roi_gradient_kernel: int = option(7, "--bg-roi-gradient-kernel", type=int,
                                         help="The size, in pixels, of the edge detector that finds the walls.")
    bg_roi_gradient_threshold: float = option(
        3000, "--bg-roi-gradient-threshold", type=float,
        help="How steeply the depth may change at a point that still counts as floor; steeper points are wall.")
    bg_roi_gradient_filter: bool = option(False, "--bg-roi-gradient-filter", type=bool,
                                          help="Leave the walls out of the arena by how steeply the depth changes.")
    bg_roi_depth_range: tuple = option((650, 750), "--bg-roi-depth-range", type=(float, float), arena=True,
                                       help="The nearest and farthest the arena floor can be from the camera, in mm.")
    manual_set_depth_range: bool = option(
        False, "--manual-set-depth-range", is_flag=True, arena=True,
        help="Search the floor depth range as given, instead of the range found from each recording.")
    bg_roi_weights: tuple = option(
        (1, 0.1, 1), "--bg-roi-weights", type=(float, float, float), arena=True,
        help="How much a candidate arena's area, its shape and its distance from the frame's center count when "
             "the arena is picked.")
    bg_roi_index: int = option(0, "--bg-roi-index", type=int, arena=True,
                               help="Which candidate arena to use, from 0 for the best match.")
    bg_roi_shape: str = option("ellipse", "--bg-roi-shape", type=str, arena=True,
                               help="The shape fitted to the arena: ellipse or rect.")
    bg_roi_dilate: tuple = option((10, 10), "--bg-roi-dilate", type=(int, int), arena=True,
                                  help="The width and height, in pixels, of each step that grows the arena.")

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

    # reading the recording (-t is --temporal-filter-size, so --threads has no short flag)
    mapping: str = option("DEPTH", "-m", "--mapping", type=str,
                          help="Which of the recording's video streams holds the depth: DEPTH, or the stream's "
                               "number.")
    threads: int = option(8, "--threads", type=int, setting=False,
                          help="How many threads read an .avi or .mkv recording.")
    delete: bool = option(False, "--delete", is_flag=True,
                          help="Delete a .dat recording once its compressed .avi has been checked.")
    chunk_size: int = option(3000, "-b", "--chunk-size", type=int,
                             help="How many frames are extracted at a time; fewer use less memory.")
    skip_completed: bool = option(False, "--skip-completed", is_flag=True,
                                  help="Leave a recording whose extraction has finished as it is.")
    compress_threads: int = option(3, "--compress-threads", type=int, setting=False,
                                   help="How many threads compress a .dat recording into an .avi.")
    compress_chunk_size: int = option(3000, "--compress-chunk-size", type=int,
                                      help="How many frames are compressed at a time.")
    compress: bool = option(False, "--compress", type=bool, arena=True,
                            help="After extracting a .dat recording, save it as an .avi about 8 times smaller.")
    frame_trim: tuple = option((0, 0), "--frame-trim", type=(int, int),
                               help="How many frames to leave out at the start and at the end of each recording.")
    model_smoothing_clips: tuple = option(
        (0, 0), "--model-smoothing-clips", type=(float, float),
        help="The range the tracking model's log-likelihoods are clipped to when it smooths the mouse's shape.")
    angle_hampel_sig: float = option(
        3, "--angle-hampel-sig", type=float,
        help="How many standard deviations the heading may jump before the heading filter replaces it.")
    angle_hampel_span: int = option(0, "--angle-hampel-span", type=int,
                                    help="The window, in frames, of the filter that replaces outlying headings; "
                                         "0 turns it off.")
    centroid_hampel_sig: float = option(
        3, "--centroid-hampel-sig", type=float,
        help="How many standard deviations the position may jump before the position filter replaces it.")
    centroid_hampel_span: int = option(0, "--centroid-hampel-span", type=int,
                                       help="The window, in frames, of the filter that replaces outlying positions; "
                                            "0 turns it off.")
    pixel_format: str = option("gray16le", "--pixel-format", type=str, arena=True,
                               help="How an .avi or .mkv recording stores its depth values.")
    movie_dtype: str = option("<u2", "--movie-dtype", arena=True,
                              help="The data type of the depth values read from the recording.")
    frame_dtype: str = option("uint8", "--frame-dtype", type=click.Choice(["uint8", "uint16"]), arena=True,
                              help="The data type the extracted frames are saved as.")
    write_movie: bool = option(True, "--write-movie", type=bool,
                               help="Save a short video of the extracted mouse beside each extraction, to check.")
    chunk_overlap: int = option(0, "--chunk-overlap", type=int, arena=True,
                                help="How many frames each chunk shares with the next; helps the tracking model "
                                     "follow the mouse from one chunk to the next.")
    tracking_model_init: str = option("raw", "--tracking-model-init", type=str,
                                      help="How the tracking model starts.")
    tracking_model_segment: bool = option(True, "--tracking-model-segment", type=bool,
                                          help="Keep only the pixels the tracking model takes to be the mouse.")
    tracking_model_ll_clip: float = option(-100, "--tracking-model-ll-clip", type=float,
                                           help="The lowest log-likelihood the tracking model gives a pixel.")
    use_tracking_model: bool = option(
        False, "--use-tracking-model", type=bool, arena=True,
        help="Turn on if a cable or implant hides part of the mouse in many frames: the mouse is then followed "
             "with a model of its shape.",
        control=Control("toggle", "Follow the mouse under cables"))
    use_cc: bool = option(True, "--use-cc", type=bool,
                          help="Take the mouse to be the largest connected shape in each frame.")
    widen_radius: int = option(0, "--widen-radius", type=int,
                               help="How many pixels to widen (or, below 0, narrow) the arena's edge when its walls "
                                    "are graduated.")
    graduate_walls: bool = option(False, "--graduate-walls", type=bool,
                                  help="Make up for walls that slope outward, by graduating the background at the "
                                       "arena's edge.")
    flip_classifier_smoothing: int = option(
        51, "--flip-classifier-smoothing", type=int, arena=True,
        help="Over how many frames head/tail detection's answers are smoothed.")
    flip_classifier: str | None = option(
        None, "--flip-classifier",
        help="A head/tail classifier file to use in place of the one for the project's animals.")
    compute_raw_scalars: bool = option(False, "--compute-raw-scalars", is_flag=True,
                                       help="Also measure the mouse in the raw frames, before they are cleaned.")
    detected_true_depth: str = option(
        "auto", "--detected-true-depth", type=str,
        help="The floor's depth, in mm, in place of the one found from each recording; auto finds it. It "
             "matters only when dilate_iterations is above 1.")
    max_height: int = option(
        120, "--max-height", type=int, arena=True,
        help="Anything taller than this above the floor isn't counted as the mouse, in mm.",
        control=Control("number", "Tallest the mouse can be", minimum=20, maximum=300, unit="mm"))
    num_frames: int | None = option(None, "--num-frames", "-n", type=int,
                                    help="Extract only this many frames of each recording; left empty, every frame.")
    crop_size: tuple = option(
        (80, 80), "--crop-size", "-c", type=(int, int), arena=True,
        help="The width and height, in pixels, of the box cut out around the mouse in every frame.",
        control=Control("number", "Crop around the mouse", minimum=40, maximum=200, unit="pixels",
                        help="The size of the square cut out around the mouse in every frame.",
                        shown_as="square"))


@dataclass(frozen=True, kw_only=True)
class RecordingsOptions(Options):
    """Which files in the project folder are recordings."""

    SECTION = "extract"

    extensions: tuple = option(RECORDING_EXTENSIONS, "--extensions", type=str, multiple=True,
                               help="The kinds of recording file to look for, by extension; repeat it for each.")
    max_depth: int = option(DEFAULT_MAX_DEPTH, "--max-depth", type=int,
                            help="How many folders below the project folder to look in for recordings.")


@dataclass(frozen=True, kw_only=True)
class DiagnosticsOptions(Options):
    """Extraction diagnostics' settings."""

    SECTION = "extract"

    mouse_threshold: float = option(
        0, "--mouse-threshold", type=float,
        help="A recording whose first frames average no more than this height, in mm, holds no mouse there.")


#: Where extraction runs, as the Extract step offers it.
_WALL_TIMES = (("auto", "From each recording's length"), (2, "2 hours"), (4, "4 hours"), (12, "12 hours"))


@dataclass(frozen=True, kw_only=True)
class ExtractComputing(ComputingOptions):
    """Where extraction runs: ``config.yaml``'s ``computing.extract`` block."""

    STEP = "extract"

    wall_time: str = option(
        "auto", "--wall-time", type=str,
        help="How long each cluster job may run before the cluster stops it, as hours:minutes:seconds (such as "
             "4:00:00), or auto for three times its recording's length, at least an hour.",
        control=Control("choice", "Time per job", choices=_WALL_TIMES, shown_as="hours",
                        help="How long each cluster job may run before the cluster stops it.",
                        show_if=("cluster_type", "slurm")))


@dataclass(frozen=True, kw_only=True)
class ConvertOptions(Options):
    """``convert-raw-to-avi``."""

    mapping: str = option("DEPTH", "-m", "--mapping", type=str,
                          help="Which of the recording's video streams holds the depth: DEPTH, or the stream's "
                               "number.")
    threads: int = option(8, "-t", "--threads", type=int, help="How many threads compress the video.")
    delete: bool = option(False, "--delete", is_flag=True,
                          help="Delete the .dat recording once its compressed copy has been checked.")
    fps: float = option(30, "--fps", type=float, help="The video's frame rate, in frames per second.")
    chunk_size: int = option(3000, "-b", "--chunk-size", type=int, help="How many frames are copied at a time.")


@dataclass(frozen=True, kw_only=True)
class CopySliceOptions(ConvertOptions):
    """``copy-slice``."""

    copy_slice: tuple = option((0, 1000), "-c", "--copy-slice", type=(int, int),
                               help="The first frame to copy and the frame to stop before.")


#: The classes of the extraction steps, whose fields config.yaml's ``extract`` section may set.
EXTRACT_STAGE = (RoiOptions, ExtractOptions, RecordingsOptions, DiagnosticsOptions, CleaningOptions)

#: The computing blocks of the extraction steps, which ``generate-config`` writes under ``computing``.
COMPUTING = (ExtractComputing,)
