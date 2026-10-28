"""
CLI for extracting the depth data.
"""

import os
import click
from moseq2_extract._yaml import write_yaml
from tqdm.auto import tqdm
from copy import deepcopy
from moseq2_extract.util import (
    command_with_config,
    read_yaml,
    recursive_find_unextracted_dirs,
)
from moseq2_extract.helpers.wrappers import (
    get_roi_wrapper,
    extract_wrapper,
    flip_file_wrapper,
    generate_index_wrapper,
    aggregate_extract_results_wrapper,
    generate_index_from_agg_res_wrapper,
    convert_raw_to_avi_wrapper,
    copy_slice_wrapper,
)
from moseq2_extract.helpers.extract import run_slurm_extract, run_local_extract

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
@click.version_option()
def cli():
    pass



def common_roi_options(function):
    """
    Decorator function for grouping shared ROI related parameters.

    Args:
    function: Function to add enclosed parameters to as click options.

    Returns:
    function: Updated function including shared parameters.
    """

    function = click.option(
        "--bg-roi-dilate",
        default=(10, 10),
        type=(int, int),
        help="Size of StructuringElement to dilate roi",
    )(function)
    function = click.option(
        "--bg-roi-shape",
        default="ellipse",
        type=str,
        help="Shape to use to detect roi (ellipse or rect)",
    )(function)
    function = click.option(
        "--bg-roi-index",
        default=0,
        type=int,
        help="Index of which detected ROI mask to use",
    )(function)
    function = click.option(
        "--bg-roi-weights",
        default=(1, 0.1, 1),
        type=(float, float, float),
        help="ROI feature weighting (area, extent, dist to center)",
    )(function)
    function = click.option(
        "--camera-type",
        default="auto",
        type=click.Choice(["auto", "kinect", "azure", "realsense"]),
        help='Camera type used for recording for auto-sets bg-roi-weights to precomputed values for different camera types. \
                             Possible types: ["kinect", "azure", "realsense"]',
    )(function)
    function = click.option(
        "--manual-set-depth-range",
        is_flag=True, default=False,
        help="Flag to deactivate auto depth range setting.",
    )(function)
    function = click.option(
        "--bg-roi-depth-range",
        default=(650, 750),
        type=(float, float),
        help="Range to search for floor of arena (in mm)",
    )(function)
    function = click.option(
        "--bg-roi-gradient-filter",
        default=False,
        type=bool,
        help="Use graident filter to exclude walls for detected ROI",
    )(function)
    function = click.option(
        "--bg-roi-gradient-threshold",
        default=3000,
        type=float,
        help="Gradient must be less than threshold to include points",
    )(function)
    function = click.option(
        "--bg-roi-gradient-kernel",
        default=7,
        type=int,
        help="Kernel size for Sobel gradient filtering",
    )(function)
    function = click.option(
        "--bg-roi-fill-holes", default=True, type=bool, help="Fill holes in ROI"
    )(function)
    function = click.option(
        "--bg-sort-roi-by-position",
        default=False,
        type=bool,
        help="Sort ROIs by position",
    )(function)
    function = click.option(
        "--bg-sort-roi-by-position-max-rois",
        default=2,
        type=int,
        help="The number of maximum ROIs sorted by area",
    )(function)
    function = click.option(
        "--dilate-iterations",
        default=1,
        type=int,
        help="Number of dilation iterations to increase bucket floor size.",
    )(function)
    function = click.option(
        "--bg-roi-erode",
        default=(1, 1),
        type=(int, int),
        help="Size of cv2 Structure Element to erode roi.",
    )(function)
    function = click.option(
        "--erode-iterations",
        default=0,
        type=int,
        help="Number of erosion iterations to decrease bucket floor size.",
    )(function)
    function = click.option(
        "--noise-tolerance",
        default=30,
        type=int,
        help="Extent of noise to accept during RANSAC Plane ROI computation.",
    )(function)
    function = click.option(
        "--output-dir",
        default="proc",
        help="Output directory to save the results h5 file",
    )(function)
    function = click.option(
        "--use-plane-bground",
        is_flag=True, default=False,
        help="Use a plane fit for the background. Useful when mice don't move much",
    )(function)
    function = click.option(
        "--recompute-bg",
        default=False,
        help="Overwrite previously computed background image",
    )(function)
    function = click.option(
        "--config-file", type=click.Path(), default=None
    )(function)
    function = click.option(
        "--progress-bar", "-p", is_flag=True, default=False, help="Show verbose progress bars."
    )(function)
    return function


def common_avi_options(function):
    """
    Decorator function for grouping shared video processing parameters.

    Args:
    function: Function to add enclosed parameters to as click options.

    Returns:
    function: Updated function including shared parameters.
    """

    function = click.option(
        "-o",
        "--output-file",
        type=click.Path(),
        default=None,
        help="Path to output file",
    )(function)
    function = click.option(
        "-b", "--chunk-size", type=int, default=3000, help="Chunk size"
    )(function)
    function = click.option("--fps", type=float, default=30, help="Video FPS")(function)
    function = click.option(
        "--delete", is_flag=True, default=False, help="Delete raw file if encoding is sucessful"
    )(function)
    function = click.option(
        "-t", "--threads", type=int, default=8, help="Number of threads for encoding"
    )(function)
    function = click.option(
        "-m",
        "--mapping",
        type=str,
        default="DEPTH",
        help="Ffprobe stream selection variable. Default: DEPTH",
    )(function)

    return function


def extract_options(function):
    """
    Decorator function for grouping shared extraction prameters.

    Args:
    function : Function to add enclosed parameters to as click options.

    Returns:
    function: Updated function including shared parameters.
    """

    function = click.option(
        "--crop-size",
        "-c",
        default=(80, 80),
        type=(int, int),
        help="Width and height of cropped mouse image",
    )(function)
    function = click.option(
        "--num-frames",
        "-n",
        default=None,
        type=int,
        help="Number of frames to extract. Will extract full session if set to None.",
    )(function)
    function = click.option(
        "--min-height",
        default=10,
        type=int,
        help="Min mouse height threshold from floor (mm)",
    )(function)
    function = click.option(
        "--max-height",
        default=120,
        type=int,
        help="Max mouse height threshold from floor (mm)",
    )(function)
    function = click.option(
        "--detected-true-depth",
        default="auto",
        type=str,
        help='Option to override automatic depth estimation during extraction. \
This is only a debugging parameter, for cases where dilate_iterations > 1, otherwise has no effect. Either "auto" or an int value.',
    )(function)
    function = click.option(
        "--compute-raw-scalars",
        is_flag=True, default=False,
        help="Compute scalar values from raw cropped frames.",
    )(function)
    function = click.option(
        "--flip-classifier",
        default=None,
        help="path to the flip classifier used to properly orient the mouse (.pkl file)",
    )(function)
    function = click.option(
        "--flip-classifier-smoothing",
        default=51,
        type=int,
        help="Number of frames to smooth flip classifier probabilities",
    )(function)
    function = click.option(
        "--graduate-walls",
        default=False,
        type=bool,
        help="Graduates and dilates the background image to compensate for slanted bucket walls. \\_/",
    )(function)
    function = click.option(
        "--widen-radius",
        default=0,
        type=int,
        help="Number of pixels to increase/decrease radius by when graduating bucket walls.",
    )(function)
    function = click.option(
        "--use-cc",
        default=True,
        type=bool,
        help="Extract features using largest connected components.",
    )(function)
    function = click.option(
        "--use-tracking-model",
        default=False,
        type=bool,
        help="Use an expectation-maximization style model to aid mouse tracking. Useful for data with cables",
    )(function)
    function = click.option(
        "--tracking-model-ll-threshold",
        default=-100,
        type=float,
        help="Threshold on log-likelihood for pixels to use for update during tracking",
    )(function)
    function = click.option(
        "--tracking-model-mask-threshold",
        default=-16,
        type=float,
        help="Threshold on log-likelihood to include pixels for centroid and angle calculation",
    )(function)
    function = click.option(
        "--tracking-model-ll-clip",
        default=-100,
        type=float,
        help="Clip log-likelihoods below this value",
    )(function)
    function = click.option(
        "--tracking-model-segment",
        default=True,
        type=bool,
        help="Segment likelihood mask from tracking model",
    )(function)
    function = click.option(
        "--tracking-model-init",
        default="raw",
        type=str,
        help="Method for tracking model initialization",
    )(function)
    function = click.option(
        "--cable-filter-iters",
        default=0,
        type=int,
        help="Number of cable filter iterations",
    )(function)
    function = click.option(
        "--cable-filter-shape",
        default="rectangle",
        type=str,
        help="Cable filter shape (rectangle or ellipse)",
    )(function)
    function = click.option(
        "--cable-filter-size",
        default=(5, 5),
        type=(int, int),
        help="Cable filter size (in pixels)",
    )(function)
    function = click.option(
        "--tail-filter-iters",
        default=1,
        type=int,
        help="Number of tail filter iterations",
    )(function)
    function = click.option(
        "--tail-filter-size", default=(9, 9), type=(int, int), help="Tail filter size"
    )(function)
    function = click.option(
        "--tail-filter-shape", default="ellipse", type=str, help="Tail filter shape"
    )(function)
    function = click.option(
        "--spatial-filter-size",
        "-s",
        default=[3],
        type=int,
        help="Space prefilter kernel (median filter, must be odd)",
        multiple=True,
    )(function)
    function = click.option(
        "--temporal-filter-size",
        "-t",
        default=[0],
        type=int,
        help="Time prefilter kernel (median filter, must be odd)",
        multiple=True,
    )(function)
    function = click.option(
        "--chunk-overlap",
        default=0,
        type=int,
        help="Frames overlapped in each chunk. Useful for cable tracking",
    )(function)
    function = click.option(
        "--write-movie",
        default=True,
        type=bool,
        help="Write a results output movie including an extracted mouse",
    )(function)
    function = click.option(
        "--frame-dtype",
        default="uint8",
        type=click.Choice(["uint8", "uint16"]),
        help="Data type for processed frames",
    )(function)
    function = click.option(
        "--movie-dtype",
        default="<i2",
        help="Data type for raw frames read in for extraction",
    )(function)
    function = click.option(
        "--pixel-format",
        default="gray16le",
        type=str,
        help="Pixel format for reading in .avi and .mkv videos",
    )(function)
    function = click.option(
        "--centroid-hampel-span", default=0, type=int, help="Hampel filter span"
    )(function)
    function = click.option(
        "--centroid-hampel-sig", default=3, type=float, help="Hampel filter sig"
    )(function)
    function = click.option(
        "--angle-hampel-span", default=0, type=int, help="Angle filter span"
    )(function)
    function = click.option(
        "--angle-hampel-sig", default=3, type=float, help="Angle filter sig"
    )(function)
    function = click.option(
        "--model-smoothing-clips",
        default=(0, 0),
        type=(float, float),
        help="Model smoothing clips",
    )(function)
    function = click.option(
        "--frame-trim",
        default=(0, 0),
        type=(int, int),
        help="Frames to trim from beginning and end of data",
    )(function)
    function = click.option(
        "--compress",
        default=False,
        type=bool,
        help="Convert .dat to .avi after successful extraction",
    )(function)
    function = click.option(
        "--compress-chunk-size",
        type=int,
        default=3000,
        help="Chunk size for .avi compression",
    )(function)
    function = click.option(
        "--compress-threads", type=int, default=3, help="Number of threads for encoding"
    )(function)
    function = click.option(
        "--skip-completed",
        is_flag=True, default=False,
        help="Will skip the extraction if it is already completed.",
    )(function)

    return function


@cli.command(
    name="find-roi",
    cls=command_with_config("config_file"),
    help="Finds the ROI (the arena) and background to subtract from frames when extracting.",
)
@click.argument("input-file", type=click.Path(exists=True))
@common_roi_options
def find_roi(input_file, output_dir, **config_data):

    get_roi_wrapper(input_file, config_data, output_dir)


@cli.command(
    name="extract",
    cls=command_with_config("config_file"),
    help="Processes raw input depth recordings to output a cropped and oriented"
    "video of the mouse and saves the output+metadata to h5 files in the given output directory.",
)
@click.argument("input-file", type=click.Path(exists=True, resolve_path=False))
@click.option(
    "--cluster-type",
    type=click.Choice(["local", "slurm"]),
    default="local",
    help="Platform to train models on",
)
@common_roi_options
@common_avi_options
@extract_options
def extract(input_file, output_dir, num_frames, skip_completed, **config_data):

    extract_wrapper(
        input_file, output_dir, config_data, num_frames=num_frames, skip=skip_completed
    )


@cli.command(
    name="batch-extract",
    cls=command_with_config("config_file"),
    help="Batch processes " "all the raw depth recordings located in the input folder.",
)
@click.argument("input-folder", type=click.Path(exists=True, resolve_path=False))
@common_roi_options
@common_avi_options
@extract_options
@click.option(
    "--extensions",
    default=[".dat"],
    type=str,
    help="File extension of raw data",
    multiple=True,
)
@click.option(
    "--skip-checks",
    is_flag=True, default=False,
    help="Flag: skip checks for the existance of a metadata file",
)
@click.option(
    "--extract-out-script",
    type=click.Path(),
    default="extract_out.sh",
    help="Name of bash script file to save extract commands.",
)
@click.option(
    "--cluster-type",
    type=click.Choice(["local", "slurm"]),
    default="local",
    help="Platform to train models on",
)
@click.option(
    "--prefix",
    type=str,
    default="",
    help="Batch command string to prefix model training command (slurm only).",
)
@click.option(
    "--ncpus", "-c", type=int, default=1, help="Number of cores to use in extraction"
)
@click.option("--memory", type=str, default="5GB", help="RAM (slurm only)")
@click.option("--wall-time", type=str, default="3:00:00", help="Wall time (slurm only)")
@click.option(
    "--partition", type=str, default="short", help="Partition name (slurm only)"
)
@click.option(
    "--get-cmd", is_flag=True, default=True, help="Print scan command strings."
)
@click.option("--run-cmd", is_flag=True, default=False, help="Run scan command strings.")
def batch_extract(
    input_folder,
    output_dir,
    skip_completed,
    num_frames,
    extensions,
    skip_checks,
    **config_data,
):

    # check if there is a config file
    config_file = config_data.get("config_file")
    if not config_file:
        # Add message to tell the users to specify a config file
        print(
            "Command not run. Please specified a config file using --config-file flag."
        )
        return

    # Add message to tell the users to specify a config file
    to_extract = []
    for ex in extensions:
        to_extract.extend(
            recursive_find_unextracted_dirs(
                input_folder,
                extension=ex,
                skip_checks=True if ex in (".tgz", ".tar.gz") else skip_checks,
                yaml_path=os.path.join(output_dir, "results_00.yaml"),
            )
        )

    # Add message when all sessions are extracted
    if len(to_extract) == 0:
        print(
            'No session to be extracted. If you want to re-extract the data, please add "--skip-checks"'
        )
        return

    if config_data["cluster_type"] == "local":
        # the session specific config doesn't get generated in session proc file
        # session specific config direct used in config_data dictionary in extraction from extract_command function
        run_local_extract(to_extract, config_file, skip_completed)
    else:
        # add paramters to config
        config_data["session_config_path"] = (
            read_yaml(config_file).get("session_config_path", "")
            if config_file is not None
            else ""
        )
        config_data["config_file"] = os.path.abspath(config_file)
        config_data["output_dir"] = output_dir
        config_data["skip_completed"] = skip_completed
        config_data["num_frames"] = num_frames
        config_data["extensions"] = extensions
        config_data["skip_checks"] = skip_checks
        # run slurm extract will generate a config.yaml in session proc file for slurm
        run_slurm_extract(input_folder, to_extract, config_data, skip_completed)


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

    flip_file_wrapper(config_file, output_dir)


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

    objs = extract.params
    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}
    if camera_type == "azure":
        params["bg_roi_depth_range"] = [550, 650]
        params["spatial_filter_size"] = [5]
        params["tail_filter_size"] = [15, 15]
        params["crop_size"] = [120, 120]
        params["camera_type"] = "azure"

    with open(output_file, "w") as f:
        write_yaml(params, f)

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
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=os.getcwd(),
    help="Directory to find h5 files",
)
@click.option(
    "--format",
    "-f",
    type=str,
    default="{start_time}_{session_name}_{subject_name}",
    help="New file name formats from resepective metadata",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "aggregate_results/"),
    help="Location for storing all results together",
)
@click.option(
    "--mouse-threshold",
    default=0,
    type=float,
    help="Threshold value for mean depth to include frames in aggregated results",
)
def aggregate_extract_results(input_dir, format, output_dir, mouse_threshold):

    aggregate_extract_results_wrapper(input_dir, format, output_dir, mouse_threshold)


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
@click.argument("input-file", type=click.Path(exists=True, resolve_path=False))
@common_avi_options
def convert_raw_to_avi(
    input_file, output_file, chunk_size, fps, delete, threads, mapping
):

    convert_raw_to_avi_wrapper(
        input_file, output_file, chunk_size, fps, delete, threads, mapping
    )


@cli.command(
    name="copy-slice",
    help="Copies a segment of an input depth recording into a new video file.",
)
@click.argument("input-file", type=click.Path(exists=True, resolve_path=False))
@common_avi_options
@click.option(
    "-c",
    "--copy-slice",
    type=(int, int),
    default=(0, 1000),
    help="Slice indices used for copy",
)
def copy_slice(
    input_file, output_file, copy_slice, chunk_size, fps, delete, threads, mapping
):

    copy_slice_wrapper(
        input_file, output_file, copy_slice, chunk_size, fps, delete, threads, mapping
    )


if __name__ == "__main__":
    cli()
