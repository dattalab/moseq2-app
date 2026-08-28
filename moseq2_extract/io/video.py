"""
Video and video-metadata read/write functions.
"""

import os
import cv2
import tarfile
import datetime
import subprocess
import numpy as np
from os.path import exists
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def get_raw_info(filename, bit_depth=16, frame_size=(512, 424)):
    """
    Get info from a raw data file with specified frame dimensions and bit depth.

    Args:
    filename (str): name of raw data file
    bit_depth (int): bits per pixel (default: 16)
    frame_dims (tuple): wxh or hxw of each frame

    Returns:
    file_info (dict): dictionary containing depth file metadata
    """

    bytes_per_frame = (frame_size[0] * frame_size[1] * bit_depth) / 8

    if type(filename) is not tarfile.TarFile:
        file_info = {
            "bytes": os.stat(filename).st_size,
            "nframes": int(os.stat(filename).st_size / bytes_per_frame),
            "dims": frame_size,
            "bytes_per_frame": bytes_per_frame,
        }
    else:
        tar_members = filename.getmembers()
        tar_names = [_.name for _ in tar_members]
        input_file = tar_members[tar_names.index("depth.dat")]
        file_info = {
            "bytes": input_file.size,
            "nframes": int(input_file.size / bytes_per_frame),
            "dims": frame_size,
            "bytes_per_frame": bytes_per_frame,
        }
    return file_info


def read_frames_raw(
    filename,
    frames=None,
    frame_size=(512, 424),
    bit_depth=16,
    movie_dtype="<u2",
    **kwargs,
):
    """
    Reads in data from raw binary file.

    Args:
    filename (string): name of raw data file
    frames (list or range): frames to extract
    frame_dims (tuple): wxh of frames in pixels
    bit_depth (int): bits per pixel (default: 16)
    movie_dtype (str): An indicator for numpy to store the piped ffmpeg-read video in memory for processing.

    Returns:
    chunk (numpy ndarray): nframes x h x w
    """

    vid_info = get_raw_info(filename, frame_size=frame_size, bit_depth=bit_depth)

    if vid_info["dims"] != frame_size:
        frame_size = vid_info["dims"]

    if type(frames) is int:
        frames = [frames]
    elif not frames or (type(frames) is range) and len(frames) == 0:
        frames = range(0, vid_info["nframes"])

    seek_point = np.maximum(0, frames[0] * vid_info["bytes_per_frame"])
    read_points = len(frames) * frame_size[0] * frame_size[1]

    dims = (len(frames), frame_size[1], frame_size[0])

    if type(filename) is tarfile.TarFile:
        tar_members = filename.getmembers()
        tar_names = [_.name for _ in tar_members]
        input_file = tar_members[tar_names.index("depth.dat")]
        with filename.extractfile(input_file) as f:
            f.seek(int(seek_point))
            chunk = f.read(int(len(frames) * vid_info["bytes_per_frame"]))
            chunk = np.frombuffer(chunk, dtype=np.dtype(movie_dtype)).reshape(dims)
    else:
        with open(filename, "rb") as f:
            f.seek(int(seek_point))
            chunk = np.fromfile(
                file=f, dtype=np.dtype(movie_dtype), count=read_points
            ).reshape(dims)

    return chunk


# https://gist.github.com/hiwonjoon/035a1ead72a767add4b87afe03d0dd7b
def get_video_info(filename, mapping="DEPTH", threads=8, count_frames=False, **kwargs):
    """
    Get file metadata from videos.

    Args:
    filename (str): name of file to read video metadata from.
    mapping (str): chooses the stream to read from files.
    threads (int): number of threads to simultanoues run the ffprobe command
    count_frames (bool): indicates whether to count the frames individually.

    Returns:
    out_dict (dict): dictionary containing video file metadata
    """

    mapping_dict = get_stream_names(filename)
    if isinstance(mapping, str):
        mapping = mapping_dict.get(mapping, 0)

    stream_str = "stream=width,height,r_frame_rate,"
    if count_frames:
        stream_str += "nb_read_frames"
    else:
        stream_str += "nb_frames"

    command = [
        "ffprobe",
        "-v",
        "fatal",
        "-select_streams",
        f"v:{mapping}",
        "-show_entries",
        stream_str,
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        "-threads",
        str(threads),
        filename,
        "-sexagesimal",
    ]

    if count_frames:
        command += ["-count_frames"]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()

    if err:
        print(err)

    out = out.decode().split("\n")
    out_dict = {
        "file": filename,
        "dims": (int(float(out[0])), int(float(out[1]))),
        "fps": float(out[2].split("/")[0]) / float(out[2].split("/")[1]),
    }

    try:
        out_dict["nframes"] = int(out[3])
    except ValueError:
        out_dict["nframes"] = None

    return out_dict


# simple command to pipe frames to an ffv1 file
def write_frames(
    filename,
    frames,
    threads=6,
    fps=30,
    pixel_format="gray16le",
    codec="ffv1",
    close_pipe=True,
    pipe=None,
    frame_dtype="uint16",
    slices=24,
    slicecrc=1,
    frame_size=None,
    get_cmd=False,
):
    """
    Write frames to avi file using the ffv1 lossless encoder

    Args:
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format video color scheme
    codec (str): ffmpeg encoding-writer method to use
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    pipe (subProcess.Pipe): pipe to currently open video file.
    frame_dtype (str): indicates the data type to use when writing the videos
    slices (int): number of frame slices to write at a time.
    slicecrc (int): check integrity of slices
    frame_size (tuple): shape/dimensions of image.
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing)

    Returns:
    pipe (subProcess.Pipe): indicates whether video writing is complete.
    """

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)
    if not frame_size and type(frames) is np.ndarray:
        frame_size = "{0:d}x{1:d}".format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = "{0:d}x{1:d}".format(frames[0], frames[1])

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "fatal",
        "-framerate",
        str(fps),
        "-f",
        "rawvideo",
        "-s",
        frame_size,
        "-pix_fmt",
        pixel_format,
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-threads",
        str(threads),
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-r",
        str(fps),
        filename,
    ]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in tqdm(
        range(frames.shape[0]), disable=True, desc=f"Writing frames to {filename}"
    ):
        pipe.stdin.write(frames[i].astype(frame_dtype).tostring())

    if close_pipe:
        pipe.communicate()
        return None
    else:
        return pipe


def get_stream_names(filename, stream_tag="title"):
    """
    Run an FFProbe command to determine whether an input video file contains multiple streams, and
    returns a stream_name to paired int values to extract the desired stream.

    Args:
    filename (str): path to video file to get streams from.
    stream_tag (str): value of the stream tags for ffprobe command to return

    Returns:
    out (dict): Dictionary of string to int pairs for the included streams in the mkv file.
    Dict will be used to choose the correct mapping number to choose which stream to read in read_frames().
    """

    command = [
        "ffprobe",
        "-v",
        "fatal",
        "-show_entries",
        "stream_tags={}".format(stream_tag),
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filename,
    ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()

    if err or len(out) == 0:
        return {"DEPTH": 0}

    out = out.decode("utf-8").rstrip("\n").split("\n")

    return {o: i for i, o in enumerate(out)}


def read_frames(
    filename,
    frames=range(
        0,
    ),
    threads=6,
    fps=30,
    frames_is_timestamp=False,
    pixel_format="gray16le",
    movie_dtype="uint16",
    frame_size=None,
    slices=24,
    slicecrc=1,
    mapping="DEPTH",
    get_cmd=False,
    finfo=None,
    **kwargs,
):
    """
    Read in frames from the .mp4/.avi file using a pipe from ffmpeg.

    Args:
    filename (str): filename to get frames from
    frames (list or numpy.ndarray): list of frames to grab
    threads (int): number of threads to use for decode
    fps (int): frame rate of camera
    frames_is_timestamp (bool): if False, indicates timestamps represent kinect v2 absolute machine timestamps,
    pixel_format (str): ffmpeg pixel format of data
    movie_dtype (str): An indicator for numpy to store the piped ffmpeg-read video in memory for processing.
    frame_size (str): wxh frame size in pixels
    slices (int): number of slices to use for decode
    slicecrc (int): check integrity of slices
    mapping (str): the stream to read from mkv files.
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing).
    finfo (dict): dictionary containing video file metadata

    Returns:
    video (numpy.ndarray):  frames x rows x columns
    """

    if finfo is None:
        finfo = get_video_info(filename, threads=threads, **kwargs)

    if frames is None or len(frames) == 0:
        frames = np.arange(finfo["nframes"], dtype="int64")

    if not frame_size:
        frame_size = finfo["dims"]

    # Compute starting time point to retrieve frames from
    if frames_is_timestamp:
        start_time = str(datetime.timedelta(seconds=frames[0]))
    else:
        start_time = str(datetime.timedelta(seconds=frames[0] / fps))

    command = [
        "ffmpeg",
        "-loglevel",
        "fatal",
        "-ss",
        start_time,
        "-i",
        filename,
        "-vframes",
        str(len(frames)),
        "-f",
        "image2pipe",
        "-s",
        "{:d}x{:d}".format(frame_size[0], frame_size[1]),
        "-pix_fmt",
        pixel_format,
        "-threads",
        str(threads),
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-vcodec",
        "rawvideo",
    ]

    if isinstance(mapping, str):
        mapping_dict = get_stream_names(filename)
        mapping = mapping_dict.get(mapping, 0)

    if filename.endswith((".mkv", ".avi")):
        command += ["-map", f"0:{mapping}"]
        command += ["-vsync", "0"]

    command += ["-"]

    if get_cmd:
        return command

    pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = pipe.communicate()

    if err:
        print("Error:", err)
        return None

    video = np.frombuffer(out, dtype=movie_dtype).reshape(
        (len(frames), frame_size[1], frame_size[0])
    )

    return video.astype("uint16")


def read_mkv(
    filename,
    frames=range(
        0,
    ),
    pixel_format="gray16be",
    movie_dtype="uint16",
    frames_is_timestamp=True,
    timestamps=None,
    **kwargs,
):
    """
    Read in frames from a .mkv file using a pipe from ffmpeg.

    Args:
    filename (str): filename to get frames from
    frames (list or numpy.ndarray): list of frame indices to read
    pixel_format (str): ffmpeg pixel format of data
    movie_dtype (str): An indicator for numpy to store the piped ffmpeg-read video in memory for processing.
    frames_is_timestamp (bool): if False, use machine timestamp if True, use frame as timestamp
    timestamps (list): array of timestamps to slice into using the frame indices
    threads (int): number of threads to use for decode
    fps (int): frame rate of camera in Hz
    frame_size (str): wxh frame size in pixels
    frame_dtype (str): indicates the data type to use when reading the videos
    slices (int): number of slices to use for decode
    slicecrc (int): check integrity of slices
    mapping (int): ffmpeg channel mapping; "o:mapping"; chooses the stream to read from mkv files.
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing).

    Returns:
    video (numpy.ndarray):  frames x rows x columns
    """

    if timestamps is None and exists(filename):
        timestamps = load_timestamps_from_movie(
            filename, mapping=kwargs.get("mapping", "DEPTH")
        )

    if timestamps is not None:
        if isinstance(frames, range):
            frames = timestamps[slice(frames.start, frames.stop, frames.step)]
        else:
            frames = [timestamps[frames[0]]]

    return read_frames(
        filename,
        frames,
        pixel_format=pixel_format,
        movie_dtype=movie_dtype,
        frames_is_timestamp=frames_is_timestamp,
        **kwargs,
    )


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
    get_cmd=False,
    cmap="jet",
    pipe=None,
    close_pipe=True,
    frame_range=None,
    progress_bar=False,
):
    """
    Simple command to pipe frames to an ffv1 file. Writes out a false-colored mp4 video.

    Args:
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format video color scheme
    codec (str): ffmpeg encoding-writer method to use
    slices (int): number of frame slices to write at a time.
    slicecrc (int): check integrity of slices
    frame_size (tuple): shape/dimensions of image.
    depth_min (int): minimum mouse depth from floor in (mm)
    depth_max (int): maximum mouse depth from floor in (mm)
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing)
    cmap (str): color map to use.
    pipe (subProcess.Pipe): pipe to currently open video file.
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    frame_range (range()): frame indices to write on video
    progress_bar (bool): If True, displays a TQDM progress bar for the video writing progress.

    Returns:
    pipe (subProcess.Pipe): indicates whether video writing is complete.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    txt_pos = (5, frames.shape[-1] - 40)

    if not np.mod(frames.shape[1], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), "constant", constant_values=0)

    if not np.mod(frames.shape[2], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=0)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = "{0:d}x{1:d}".format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = "{0:d}x{1:d}".format(frames[0], frames[1])

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "fatal",
        "-threads",
        str(threads),
        "-framerate",
        str(fps),
        "-f",
        "rawvideo",
        "-s",
        frame_size,
        "-pix_fmt",
        pixel_format,
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-r",
        str(fps),
        filename,
    ]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # scale frames to appropriate depth ranges
    use_cmap = plt.get_cmap(cmap)
    for i in tqdm(
        range(frames.shape[0]),
        disable=not progress_bar,
        desc=f"Writing frames to {filename}",
    ):
        disp_img = frames[i, :].copy().astype("float32")
        disp_img = (disp_img - depth_min) / (depth_max - depth_min)
        disp_img[disp_img < 0] = 0
        disp_img[disp_img > 1] = 1
        disp_img = np.delete(use_cmap(disp_img), 3, 2) * 255
        # OpenCV 5 asserts img.depth() == CV_8U in putText, where 4.x drew onto
        # a float image without complaint. The colormap hands back float64 in
        # 0-255, so the cast to uint8 that used to happen at the pipe write
        # below now happens here, before the frame number is drawn.
        #
        # This is not pixel-identical, and the difference is confined to the
        # text. It is drawn with LINE_AA, so its antialiased edges blend against
        # whatever precision the image carries, and blending in uint8 rather
        # than in float64 moves the boundary pixels of the digits. Every other
        # pixel is unchanged: the cast is the same truncation wherever it runs.
        # The artifact is the preview movie's frame-number overlay, which is for
        # looking at rather than measuring.
        disp_img = disp_img.astype("uint8")
        if frame_range is not None:
            try:
                cv2.putText(
                    disp_img,
                    str(frame_range[i]),
                    txt_pos,
                    font,
                    1,
                    white,
                    2,
                    cv2.LINE_AA,
                )
            except (IndexError, ValueError):
                # len(frame_range) M < len(frames) or
                # txt_pos is outside of the frame dimensions
                print("Could not overlay frame number on preview on video.")

        pipe.stdin.write(disp_img.tostring())

    if close_pipe:
        pipe.communicate()
        return None
    else:
        return pipe


def load_movie_data(
    filename, frames=None, frame_size=(512, 424), bit_depth=16, **kwargs
):
    """
    Parse file extension and load the movie data into numpy array.

    Args:
    filename (str): Path to video.
    frames (int or list): Frame indices to read in to output array.
    frame_size (tuple): Video dimensions (nrows, ncols)
    bit_depth (int): Number of bits per pixel, corresponds to image resolution.
    kwargs (dict): Any additional parameters that could be required in read_frames_raw().

    Returns:
    frame_data (numpy.ndarray): Read video as numpy array. (nframes, nrows, ncols)
    """

    if type(frames) is int:
        frames = [frames]
    try:
        if type(filename) is tarfile.TarFile:
            frame_data = read_frames_raw(
                filename,
                frames=frames,
                frame_size=frame_size,
                bit_depth=bit_depth,
                **kwargs,
            )
        elif filename.lower().endswith(".dat"):
            frame_data = read_frames_raw(
                filename,
                frames=frames,
                frame_size=frame_size,
                bit_depth=bit_depth,
                **kwargs,
            )
        elif filename.lower().endswith(".mkv"):
            frame_data = read_mkv(filename, frames, frame_size=frame_size, **kwargs)
        elif filename.lower().endswith(".avi"):
            frame_data = read_frames(filename, frames, frame_size=frame_size, **kwargs)

    except AttributeError as e:
        print("Error reading movie:", e)
        frame_data = read_frames_raw(
            filename,
            frames=frames,
            frame_size=frame_size,
            bit_depth=bit_depth,
            **kwargs,
        )

    return frame_data


def get_movie_info(
    filename, frame_size=(512, 424), bit_depth=16, mapping="DEPTH", threads=8, **kwargs
):
    """
    Return dict of movie metadata.

    Args:
    filename (str): path to video file
    frame_dims (tuple): video dimensions
    bit_depth (int): integer indicating data type encoding
    mapping (str): the stream to read from mkv files
    threads (int): number of threads to simultaneously read timestamps stored within the raw data file.

    Returns:
    metadata (dict): dictionary containing video file metadata
    """

    try:
        if type(filename) is tarfile.TarFile:
            metadata = get_raw_info(
                filename, frame_size=frame_size, bit_depth=bit_depth
            )
        elif filename.lower().endswith(".dat"):
            metadata = get_raw_info(
                filename, frame_size=frame_size, bit_depth=bit_depth
            )
        elif filename.lower().endswith((".avi", ".mkv")):
            metadata = get_video_info(
                filename, mapping=mapping, threads=threads, **kwargs
            )
    except AttributeError as e:
        print("Error reading movie metadata:", e)
        metadata = {}

    return metadata


def load_timestamps_from_movie(input_file, threads=8, mapping="DEPTH"):
    """
    Run a ffprobe command to extract the timestamps from the .mkv file, and pipes the
    output data to a csv file.

    Args:
    filename (str): path to input file to extract timestamps from.
    threads (int): number of threads to simultaneously read timestamps
    mapping (str): chooses the stream to read from mkv files. (Will default to if video is not an mkv format)

    Returns:
    timestamps (list): list of float values representing timestamps for each frame.
    """

    print("Loading movie timestamps")

    if isinstance(mapping, str):
        mapping_dict = get_stream_names(input_file)
        mapping = mapping_dict.get(mapping, 0)

    command = [
        "ffprobe",
        "-select_streams",
        f"v:{mapping}",
        "-threads",
        str(threads),
        "-show_entries",
        "frame=pkt_pts_time",
        "-v",
        "quiet",
        input_file,
        "-of",
        "csv=p=0",
    ]

    ffprobe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffprobe.communicate()

    if err:
        print("Error:", err)
        return None

    timestamps = [float(t) for t in out.split()]

    if len(timestamps) == 0:
        return None

    return timestamps
