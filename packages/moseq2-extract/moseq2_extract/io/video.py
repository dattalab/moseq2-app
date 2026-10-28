"""
Video and video-metadata read/write functions.
"""

import os
import av
import functools
from fractions import Fraction
import cv2
import tarfile
import numpy as np
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
    movie_dtype (str): numpy dtype to read the raw frames as.

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



def _open(filename):
    """Open a video for reading, or say why it cannot be read.

    Every probe and reader below starts here, so a path that is missing,
    truncated past its header or not a video at all fails in one place and
    with one exception rather than part way through a decode.
    """
    try:
        return av.open(str(filename))
    except (OSError, av.FFmpegError) as error:
        raise ValueError(f"could not read {filename}: {error}") from error


def _video_stream(container, mapping):
    """The stream a ``mapping`` names, as an index or as a track title."""
    if isinstance(mapping, str):
        mapping = get_stream_names(container.name).get(mapping, 0)
    return container.streams.video[mapping]


@functools.lru_cache(maxsize=8)
def _frame_timestamps(filename, index, fingerprint):
    """Every frame's presentation timestamp, in the stream's own time base.

    A container is not indexed by frame number, so the only thing that says
    where frame *n* is, is the timestamp of the *n*th frame. Working one out
    from the rate the stream declares instead would be at the mercy of what
    that rate says: Matroska often carries a guessed or millisecond-quantised
    rate, and a recording that dropped frames did not run at its nominal one.

    Reading them costs a demux of the file, without decoding any of it, so
    the list is kept for as long as the file it came from is unchanged.
    """
    with _open(filename) as container:
        stream = container.streams.video[index]
        return tuple(
            packet.pts
            for packet in container.demux(stream)
            if packet.pts is not None and packet.size
        )


def _fingerprint(filename):
    """What makes a cached timestamp list stale."""
    stat = os.stat(filename)
    return stat.st_size, stat.st_mtime_ns


def get_video_info(filename, mapping="DEPTH", threads=8, count_frames=False, **kwargs):
    """
    Get file metadata from videos.

    Args:
    filename (str): name of file to read video metadata from.
    mapping (str): chooses the stream to read from files.
    threads (int): number of threads to use for decode
    count_frames (bool): indicates whether to count the frames individually.

    Returns:
    out_dict (dict): dictionary containing video file metadata
    """

    with _open(filename) as container:
        stream = _video_stream(container, mapping)

        # The rate the stream declares, not the average over the frames that
        # turned up. A recording that dropped frames still ran at the rate
        # its camera was set to, and that is what the readers seek by.
        rate = stream.base_rate

        # Matroska stores a duration rather than a frame count, so an Azure
        # Kinect recording has none to report. None rather than 0: a caller
        # that asks for every frame tests this and falls back to the
        # timestamps, where 0 would read nothing and report success.
        if count_frames:
            # Demuxing ends with a zero-size flush packet, which is not a
            # frame and must not be counted as one: a frame count one past
            # the end reaches gen_batch_sequence and the last batch asks
            # for a frame the file does not hold.
            nframes = sum(1 for packet in container.demux(stream) if packet.size)
        else:
            nframes = stream.frames or None

        return {
            "file": filename,
            "dims": (stream.width, stream.height),
            "fps": float(rate),
            "nframes": nframes,
        }



def _frame_dimensions(frames, frame_size):
    """The width and height to size an output stream with.

    ``frame_size`` wins when it is given, as either a ``(width, height)``
    pair or a ``"WxH"`` string. Otherwise the frames say: an ndarray by its
    shape, or a bare ``(width, height)`` pair passed in place of one by a
    caller opening the file before its first chunk has arrived.
    """
    if frame_size:
        if isinstance(frame_size, str):
            width, height = frame_size.split("x")
            return int(width), int(height)
        return int(frame_size[0]), int(frame_size[1])
    if isinstance(frames, np.ndarray):
        return int(frames.shape[2]), int(frames.shape[1])
    return int(frames[0]), int(frames[1])


class _VideoWriter:
    """An open output file, written one frame at a time.

    Extraction writes its output in chunks and keeps the file open between
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
        """Encode one ``(height, width)`` or ``(height, width, 3)`` array."""
        self.container.mux(self.stream.encode(av.VideoFrame.from_ndarray(frame, format=format)))

    def communicate(self, *args, **kwargs):
        """Flush the encoder, write the trailer and close the file."""
        if self.container is not None:
            self.container.mux(self.stream.encode())
            self.container.close()
            self.container = None
        return None, None


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
):
    """
    Write frames to avi file using the ffv1 lossless encoder

    Args:
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write, or a (width, height) pair to size
    the file with when no frames have arrived yet.
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format video color scheme
    codec (str): encoder to write with
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    pipe (_VideoWriter): the already open file to keep writing to.
    frame_dtype (str): indicates the data type to use when writing the videos
    slices (int): unused, kept so a config splatted in as keywords still fits
    slicecrc (int): unused, kept so a config splatted in as keywords still fits
    frame_size (tuple): shape/dimensions of image.

    Returns:
    pipe (_VideoWriter): indicates whether video writing is complete.
    """

    if not pipe:
        pipe = _VideoWriter(
            filename,
            codec=codec,
            pix_fmt=pixel_format,
            frame_size=_frame_dimensions(frames, frame_size),
            fps=fps,
            threads=threads,
        )

    if isinstance(frames, np.ndarray):
        for i in tqdm(
            range(frames.shape[0]), disable=True, desc=f"Writing frames to {filename}"
        ):
            pipe.write(frames[i].astype(frame_dtype), format=pixel_format)

    if close_pipe:
        pipe.communicate()
        return None
    else:
        return pipe


def get_stream_names(filename, stream_tag="title"):
    """
    Determine whether an input video file contains multiple streams, and
    returns a stream_name to paired int values to extract the desired stream.

    Args:
    filename (str): path to video file to get streams from.
    stream_tag (str): the stream metadata tag naming each track

    Returns:
    out (dict): Dictionary of string to int pairs for the included streams in the mkv file.
    Dict will be used to choose the correct mapping number to choose which stream to read in read_frames().
    """

    with _open(filename) as container:
        named = {
            stream.metadata[stream_tag]: i
            for i, stream in enumerate(container.streams.video)
            if stream_tag in stream.metadata
        }

    # A file from anything but an Azure Kinect names nothing. Its one track
    # is the one the caller is about to read depth out of.
    return named or {"DEPTH": 0}


def read_frames(
    filename,
    frames=range(
        0,
    ),
    threads=6,
    fps=30,
    pixel_format="gray16le",
    movie_dtype="uint16",
    frame_size=None,
    slices=24,
    slicecrc=1,
    mapping="DEPTH",
    finfo=None,
    **kwargs,
):
    """
    Read in frames from a video file.

    Frames are addressed by index. ``fps`` describes the recording and is not
    used to locate anything, so a recording at any rate returns the frames
    that were asked for.

    Args:
    filename (str): filename to get frames from
    frames (list or numpy.ndarray): list of frames to grab
    threads (int): number of threads to use for decode
    fps (int): frame rate of camera
    pixel_format (str): pixel format to return the frames in
    movie_dtype (str): numpy dtype of the returned array
    frame_size (tuple): wxh frame size in pixels, if the frames are to be resized
    slices (int): unused, kept so a config splatted in as keywords still fits
    slicecrc (int): unused, kept so a config splatted in as keywords still fits
    mapping (str or int): the stream to read from mkv files.
    finfo (dict): dictionary containing video file metadata

    Returns:
    video (numpy.ndarray):  frames x rows x columns
    """

    with _open(filename) as container:
        stream = _video_stream(container, mapping)
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = threads

        if frames is None or len(frames) == 0:
            if finfo is None:
                finfo = get_video_info(filename, threads=threads, **kwargs)
            frames = np.arange(finfo["nframes"], dtype="int64")

        width, height = frame_size or (stream.width, stream.height)

        # Where the first requested frame actually sits, read from the file
        # rather than worked out from the rate it declares. The seek lands on
        # or before it, because a decoder has to start from a keyframe, so
        # the frames in between are decoded and dropped.
        timestamps = _frame_timestamps(str(filename), stream.index, _fingerprint(filename))
        if frames[0] >= len(timestamps):
            raise ValueError(
                f"{filename} holds {len(timestamps)} frames and frame "
                f"{frames[0]} was requested"
            )
        first = timestamps[frames[0]]
        container.seek(first, stream=stream)

        video = []
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts < first:
                continue
            video.append(
                frame.reformat(width=width, height=height, format=pixel_format).to_ndarray()
            )
            if len(video) == len(frames):
                break

    if len(video) != len(frames):
        raise ValueError(
            f"{filename} has {len(video)} frames from index {frames[0]}, "
            f"and {len(frames)} were requested"
        )

    return np.asarray(video, dtype=movie_dtype)


def read_mkv(
    filename,
    frames=range(
        0,
    ),
    pixel_format="gray16le",
    movie_dtype="uint16",
    timestamps=None,
    **kwargs,
):
    """
    Read in frames from an Azure Kinect .mkv file.

    An Azure Kinect names its tracks, and the depth one is not the first, so
    the stream is chosen by title rather than by position. The track holds
    16-bit samples big-endian; the decoder converts them, so what comes back
    is in the machine's own byte order whatever the recording stored.

    Args:
    filename (str): filename to get frames from
    frames (list or numpy.ndarray): list of frame indices to read
    pixel_format (str): pixel format to return the frames in
    movie_dtype (str): numpy dtype of the returned array
    timestamps (list): unused, kept so an existing caller still fits
    mapping (str or int): the stream to read from mkv files.

    Returns:
    video (numpy.ndarray):  frames x rows x columns
    """

    kwargs.setdefault("mapping", "DEPTH")
    return read_frames(
        filename,
        frames,
        pixel_format=pixel_format,
        movie_dtype=movie_dtype,
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
    cmap="jet",
    pipe=None,
    close_pipe=True,
    frame_range=None,
    progress_bar=False,
):
    """
    Write out a false-colored mp4 a person can watch in a browser.

    Args:
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write, or a (width, height) pair to size
    the file with when no frames have arrived yet.
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format of the frames handed to the encoder
    codec (str): encoder to write with
    slices (int): unused, kept so a config splatted in as keywords still fits
    slicecrc (int): unused, kept so a config splatted in as keywords still fits
    frame_size (tuple): shape/dimensions of image.
    depth_min (int): minimum mouse depth from floor in (mm)
    depth_max (int): maximum mouse depth from floor in (mm)
    cmap (str): color map to use.
    pipe (_VideoWriter): the already open file to keep writing to.
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    frame_range (range()): frame indices to write on video
    progress_bar (bool): If True, displays a TQDM progress bar for the video writing progress.

    Returns:
    pipe (_VideoWriter): indicates whether video writing is complete.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)

    if isinstance(frames, np.ndarray):
        txt_pos = (5, frames.shape[-1] - 40)

        # H.264 cannot encode an odd width or height.
        if not np.mod(frames.shape[1], 2) == 0:
            frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), "constant", constant_values=0)

        if not np.mod(frames.shape[2], 2) == 0:
            frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=0)

    if not pipe:
        pipe = _VideoWriter(
            filename,
            codec=codec,
            # The output pixel format, which is what decides whether a browser
            # can play the result. Left to the encoder's own preference, an
            # rgb24 input lands on yuv444p, giving a High 4:4:4 Predictive
            # stream; browser H.264 is 8-bit 4:2:0 only, so the movie is
            # undecodable in the preview widget that displays it. The crowd
            # movie writer in moseq2-viz sets the same pair.
            pix_fmt="yuv420p",
            frame_size=_frame_dimensions(frames, frame_size),
            fps=fps,
            threads=threads,
        )

    if isinstance(frames, np.ndarray):
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
            # OpenCV 5 asserts img.depth() == CV_8U in putText, where 4.x drew
            # onto a float image without complaint. The colormap hands back
            # float64 in 0-255, so the cast to uint8 that used to happen at the
            # pipe write below now happens here, before the frame number is
            # drawn.
            #
            # This is not pixel-identical, and the difference is confined to
            # the text. It is drawn with LINE_AA, so its antialiased edges
            # blend against whatever precision the image carries, and blending
            # in uint8 rather than in float64 moves the boundary pixels of the
            # digits. Every other pixel is unchanged: the cast is the same
            # truncation wherever it runs. The artifact is the preview movie's
            # frame-number overlay, which is for looking at rather than
            # measuring.
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

            pipe.write(disp_img, format=pixel_format)

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
    Read the presentation timestamp of every frame in a movie.

    An Azure Kinect recording carries no frame count, so the length of this
    list is the only frame count there is for one.

    Args:
    input_file (str): path to input file to extract timestamps from.
    threads (int): number of threads to use for decode
    mapping (str): chooses the stream to read from mkv files.

    Returns:
    timestamps (list): list of float values representing timestamps for each frame.
    """

    print("Loading movie timestamps")

    with _open(input_file) as container:
        stream = _video_stream(container, mapping)
        time_base = stream.time_base
        index = stream.index

    return [
        float(pts * time_base)
        for pts in _frame_timestamps(str(input_file), index, _fingerprint(input_file))
    ]
