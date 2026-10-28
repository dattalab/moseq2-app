"""
Helper functions for handling crowd movie file writing and video metadata maintenance.
"""

import av
from fractions import Fraction
import cv2
import warnings
import numpy as np
from os.path import join
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from moseq2_viz.viz import make_crowd_matrix
from cytoolz.itertoolz import peek, pluck, first
from cytoolz.dicttoolz import valfilter, merge_with
from moseq2_viz.model.util import get_syllable_slices
from cytoolz.curried import get_in, keyfilter, valmap
from moseq2_viz._yaml import write_yaml


def check_video_parameters(index: dict) -> dict:
    """
    Verify extraction parameters that should be consistent across sessions. If not, the function raises a RuntimeError.

    Args:
    index (dict): dictionary of extraction parameters.

    Returns:
    vid_parameters (dict): a dictionary with a subset of the used extraction parameters.
    """
    from moseq2_viz.util import read_yaml

    # define constants
    check_parameters = ["crop_size", "fps"]

    get_yaml = get_in(["path", 1])
    ymls = list(map(get_yaml, index["files"].values()))

    # load yaml config files when needed
    dicts = map(read_yaml, ymls)
    # get the parameters key within each dict
    params = pluck("parameters", dicts)

    first_entry, params = peek(params)
    if "resolution" in first_entry:
        check_parameters += ["resolution"]

    # filter for only keys in check_parameters
    params = map(keyfilter(lambda k: k in check_parameters), params)
    # turn lists (in the dict values) into tuples
    params = map(valmap(lambda x: tuple(x) if isinstance(x, list) else x), params)

    # get unique parameter values
    vid_parameters = merge_with(set, params)

    incorrect_parameters = valfilter(lambda x: len(x) > 1, vid_parameters)

    # if there are multiple values for a parameter, raise error
    if incorrect_parameters:
        raise RuntimeError(
            "The following parameters are not equal "
            + f"across extractions: {list(incorrect_parameters)}"
        )

    # grab the first value in the set
    vid_parameters = valmap(first, vid_parameters)

    # update resolution
    if "resolution" in vid_parameters:
        vid_parameters["resolution"] = tuple(
            x + 100 for x in vid_parameters["resolution"]
        )
    else:
        vid_parameters["resolution"] = None

    return vid_parameters


def write_crowd_movie_info_file(model_path, model_fit, index_file, output_dir):
    """
    Create an info.yaml file in the crowd movie directory that holds model training parameters.

    Args:
    model_path (str): path to model used to generate movies
    model_fit (dict): loaded ARHMM mdoel results
    index_file (str): path to index file used with model
    output_dir (str): path to crowd movies directory to store file in.
    """

    # Crowd movie info file contents; used to indicate the modeling state the crowd_movies were generated from
    info_parameters = ["model_class", "kappa", "gamma", "alpha"]

    # Loading parameters to dict to save to file in output directory
    info_file = join(output_dir, "info.yaml")
    info_dict = {k: model_fit["model_parameters"][k] for k in info_parameters}

    # Adding model and index file paths
    info_dict["model_path"] = model_path
    info_dict["index_path"] = index_file

    # Convert numpy dtypes to their corresponding primitives
    info_dict = valmap(
        lambda v: v.item() if isinstance(v, np.generic) else v, info_dict
    )

    # Write metadata info file
    with open(info_file, "w") as f:
        write_yaml(info_dict, f)


def write_crowd_movies(
    sorted_index, config_data, ordering, labels, label_uuids, output_dir
):
    """
    Create crowd movies for each syllable.

    Args:
    sorted_index (dict): dictionary of sorted index data.
    config_data (dict): dictionary of visualization parameters.
    filename_format (str): string format that denotes the saved crowd movie file names.
    ordering (list): ordering for the new mapping of the relabeled syllable usages.
    labels (numpy ndarray): list of syllable usages
    label_uuids (list): list of session uuids each series of labels belongs to.
    output_dir (str): path directory where all the movies are written.
    """
    progress_bar = config_data.get("progress_bar", False)

    # Filtering parameters
    clean_params = {
        "gaussfilter_space": config_data["gaussfilter_space"],
        "medfilter_space": config_data["medfilter_space"],
    }

    # Set crowd movie filename format based on whether syllables were relabeled
    if config_data.get("sort", True):
        filename_format = "syllable_sorted-id-{:02d}_({})_original-id-{:02d}.mp4"
    else:
        filename_format = "syllable_{:02d}.mp4"

    # Ensure all video metadata parameters are consistent
    vid_parameters = check_video_parameters(sorted_index)
    if vid_parameters["resolution"] is not None:
        config_data["raw_size"] = vid_parameters["resolution"]
    # writing function
    config_data["fps"] = vid_parameters["fps"]

    # Get frame slices from all included sessions for each syllable label
    slice_fun = partial(
        get_syllable_slices, labels=labels, label_uuids=label_uuids, index=sorted_index
    )

    # create crowd movie matrix to put the examples in the same syllable into a movie
    matrix_fun = partial(
        make_crowd_matrix,
        nexamples=config_data.get("max_examples", 20),
        max_dur=config_data.get("max_dur", 60),
        min_dur=config_data.get("min_dur", 0),
        min_height=config_data.get("min_height", 10),
        crop_size=vid_parameters.get("crop_size", (80, 80)),
        raw_size=config_data.get("raw_size", (512, 424)),
        select_median_duration_instances=config_data.get(
            "select_median_duration_instances", False
        ),
        scale=config_data.get("scale", 1),
        pad=config_data.get("pad", 30),
        frame_path=config_data.get("frame_path", "frames"),
        legacy_jitter_fix=config_data.get("legacy_jitter_fix", False),
        seed=config_data.get("seed", 0),
        **clean_params,
    )

    # write the crowd movies
    write_fun = partial(
        write_frames_preview,
        fps=vid_parameters["fps"],
        depth_min=config_data["min_height"],
        depth_max=config_data["max_height"],
        cmap=config_data["cmap"],
        progress_bar=progress_bar,
    )

    namer = partial(
        _fname_formatter,
        format=filename_format,
        output_dir=output_dir,
        ordering=ordering,
        count=config_data["count"],
    )

    make_matrix = partial(
        _matrix_writer_helper,
        matrix_fun=matrix_fun,
        slice_fun=slice_fun,
        write_fun=write_fun,
        namer=namer,
    )

    # parallel process the crowd movies for all syllables
    with mp.Pool(config_data.get("processes")) as pool:
        # Compute crowd matrices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return list(
                tqdm(
                    filter(
                        lambda x: x is not None,
                        pool.imap(make_matrix, config_data["crowd_syllables"]),
                    ),
                    desc="Writing crowd movies",
                    total=len(config_data["crowd_syllables"]),
                )
            )


def _fname_formatter(syll, format, output_dir, ordering, count):
    """
    Create filename strings for the syllable crowd movies to generate.

    Args:
    syll (int): syllable number.
    format (str): format string for outputted file. E.g. "syllable_sorted-id-{:02d}_({})_original-id-{:02d}.mp4"
    output_dir (str): path to output directory containing the crowd movie.
    ordering (dict): dictionary that holds the original id of the syllable crowd movie (pre-reordering).
    count (str): method to compute syllable mean usage, either 'usage' or 'frames'.

    Returns:
    (str): path to syllable crowd movie.
    """
    return join(output_dir, format.format(syll, count, ordering[syll]))


def _matrix_writer_helper(syll, matrix_fun, slice_fun, write_fun, namer):
    """
    Generate crowd movies using multiprocessing.

    Args:
    syll (int): syllable number.
    matrix_fun (function): helper function to create stacked video matrices given syllable slices.
    slice_fun (function): helper function to generate syllable slices given syllable number.
    write_fun (function): helper function to write the crowd movies to their respective files.
    namer (function): helper function to create filename strings for the syllable crowd movies to generate.

    Returns:
    (function): helper function to write the crowd movies to their respective files, if the video matrices were created.
    """

    mtx = matrix_fun(slice_fun(syll))
    if mtx is not None:
        return write_fun(namer(syll), mtx)
    return None


class _VideoWriter:
    """An open output file, written one frame at a time.

    A crowd movie is composited in chunks and the file stays open between
    them. ``communicate`` flushes whatever the encoder is still holding,
    writes the container's trailer and closes the file; calling it twice is
    harmless.
    """

    def __init__(self, filename, codec, pix_fmt, frame_size, fps, threads):
        self.container = av.open(str(filename), "w")
        # A rate is a rational, and callers hand this one in as whatever
        # their config or CLI made of it: an int, a float from a `type=float`
        # option, or a string from a yaml file. 1000 is enough denominator
        # for the broadcast rates (30000/1001 and friends) a camera reports.
        rate = Fraction(str(fps)).limit_denominator(1000)
        self.stream = self.container.add_stream(codec, rate=rate)
        self.stream.width, self.stream.height = frame_size
        self.stream.pix_fmt = pix_fmt
        self.stream.thread_type = "AUTO"
        self.stream.codec_context.thread_count = threads

    def write(self, frame, format):
        """Encode one ``(height, width, 3)`` array."""
        self.container.mux(self.stream.encode(av.VideoFrame.from_ndarray(frame, format=format)))

    def communicate(self, *args, **kwargs):
        """Flush the encoder, write the trailer and close the file."""
        if self.container is not None:
            self.container.mux(self.stream.encode())
            self.container.close()
            self.container = None
        return None, None


def write_frames_preview(
    filename,
    frames=np.empty((0,)),
    threads=6,
    fps=30,
    pixel_format="rgb24",
    codec="h264",
    slices=24,
    slicecrc=1,
    frame_size=None,
    depth_min=0,
    depth_max=80,
    cmap="jet",
    text=None,
    text_scale=1,
    text_thickness=2,
    pipe=None,
    close_pipe=True,
    progress_bar=True,
    **kwargs,
):
    """
    Writes out a false-colored mp4 video.

    Args:
    filename (str): path to write output crowd movie file
    frames (numpy.ndarray): num_frames x rows x columns
    threads (int): number of threads to write file
    fps (int): frames per second
    pixel_format (str): format of the frames handed to the encoder
    codec (str): encoder to write with
    slices (int): unused, kept so a config splatted in as keywords still fits
    slicecrc (int): unused, kept so a config splatted in as keywords still fits
    frame_size (tuple): image dimensions
    depth_min (int): minimum mouse distance from bucket floor
    depth_max (int): maximum mouse distance from bucket floor
    cmap (str): color map selection.
    text (range(num_frames): display frame number in output video.
    text_scale (int): text size.
    text_thickness (int): text thickness.
    pipe (_VideoWriter): if not None, indicates that there are more frames to be written.
    close_pipe (bool): indicates whether video is done writing, and to close pipe to file-stream.
    progress_bar (bool): display progress bar.
    kwargs (dict): extra keyword arguments

    Returns:
    pipe (_VideoWriter): if there are more slices/chunks to write to, otherwise the path to the movie.
    """

    # pad frames so that dimensions are divisible by 2
    if not np.mod(frames.shape[1], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), "constant", constant_values=0)

    if not np.mod(frames.shape[2], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=0)

    # Get frame dimensions
    if not frame_size and type(frames) is np.ndarray:
        frame_size = (frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = (int(frames[0]), int(frames[1]))
    elif isinstance(frame_size, str):
        width, height = frame_size.split("x")
        frame_size = (int(width), int(height))

    # Set text metadata to write frame numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    txt_pos = (5, frames.shape[-1] - 40)

    if not pipe:
        pipe = _VideoWriter(
            filename,
            codec=codec,
            # The output pixel format, which is what decides whether a browser
            # can play the result. Left to the encoder's own preference, an
            # rgb24 input lands on yuv444p, giving a High 4:4:4 Predictive
            # stream; browser H.264 is 8-bit 4:2:0 only, so the movie is
            # undecodable in the widget that displays it.
            pix_fmt="yuv420p",
            frame_size=frame_size,
            fps=fps,
            threads=threads,
        )

    # Get color map
    use_cmap = plt.get_cmap(cmap)

    # Write movie
    for i in tqdm(
        range(frames.shape[0]), desc="Writing frames", disable=not progress_bar
    ):
        disp_img = frames[i, :].copy().astype("float32")
        disp_img = (disp_img - depth_min) / (depth_max - depth_min)
        disp_img[disp_img < 0] = 0
        disp_img[disp_img > 1] = 1
        disp_img = np.delete(use_cmap(disp_img), 3, 2) * 255
        # OpenCV 5 asserts img.depth() == CV_8U in putText, where 4.x drew
        # onto a float image without complaint. The colormap hands back
        # float64 in 0-255, so the cast happens before the text is drawn
        # rather than at the write below.
        disp_img = disp_img.astype("uint8")
        if text is not None:
            disp_img = cv2.putText(
                disp_img,
                text,
                txt_pos,
                font,
                text_scale,
                white,
                text_thickness,
                cv2.LINE_AA,
            )
        pipe.write(disp_img, format=pixel_format)

    if close_pipe:
        pipe.communicate()
        return filename
    else:
        return pipe
