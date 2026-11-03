"""Writing movies: the open-file writer, and the false-coloured preview a person watches.

Extraction writes its depth output and its preview movie through these, the
flip widget its corrected preview, and moseq2-viz its crowd movies.

This module imports PyAV, OpenCV and matplotlib, which come with the
``video`` extra: ``python -m pip install "moseq2-common[video]"``.
moseq2-extract, moseq2-pca and moseq2-viz require that extra.
"""

from fractions import Fraction

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

__all__ = ["VideoWriter", "frame_dimensions", "write_frames_preview"]


def frame_dimensions(frames, frame_size):
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


class VideoWriter:
    """An open output file, written one frame at a time.

    Extraction and crowd movies write in chunks and keep the file open
    between them. ``communicate`` flushes whatever the encoder is still
    holding, writes the container's trailer and closes the file; calling it
    twice is harmless.
    """

    def __init__(self, filename, codec, pix_fmt, frame_size, fps, threads):
        # The mp4 index goes ahead of the frames, so a browser can show the
        # first frame and seek without fetching the whole file. Other
        # containers ignore the flag.
        self.container = av.open(str(filename), "w", container_options={"movflags": "+faststart"})
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


def write_frames_preview(
    filename,
    frames=np.empty((0,)),
    *,
    threads=6,
    fps=30,
    pixel_format="rgb24",
    codec="h264",
    frame_size=None,
    depth_min=0,
    depth_max=80,
    cmap="jet",
    frame_numbers=None,
    text=None,
    text_scale=1,
    text_thickness=2,
    pipe=None,
    close_pipe=True,
    progress_bar=False,
    **unused,
):
    """
    Write out a false-coloured mp4 a person can watch in a browser.

    Args:
    filename (str): path to file to write to.
    frames (np.ndarray): depth frames (frames x rows x columns) to write, or a
        (width, height) pair to size the file with when no frames have arrived yet.
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format of the frames handed to the encoder
    codec (str): encoder to write with
    frame_size (tuple or str): (width, height) or "WxH" to size the file
        with; by default the (padded) frames decide.
    depth_min (int): height from the floor (mm) at the bottom of the colormap
    depth_max (int): height from the floor (mm) at the top of the colormap
    cmap (str): matplotlib colormap name.
    frame_numbers (sequence): a number to draw on each frame, as extraction
        does; a frame it cannot be drawn on is written without one.
    text (str): a caption to draw on every frame, as crowd movies do.
    text_scale (int): text size.
    text_thickness (int): text thickness.
    pipe (VideoWriter): the already open file to keep writing to.
    close_pipe (bool): close the file once these frames are written.
    progress_bar (bool): show a progress bar while writing.
    unused (dict): ignored, so a config splatted in as keywords still fits;
        ``frame_range``, the name ``frame_numbers`` had before 1.6.0, is refused.

    Returns:
    pipe (VideoWriter or str): the open file when ``close_pipe`` is False,
        otherwise ``filename``.
    """

    if "frame_range" in unused:
        raise TypeError(
            f"write_frames_preview() was given frame_range= for {filename}; since moseq2 1.6.0 "
            "the frame numbers to draw are passed as frame_numbers=. Rename the keyword."
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)

    if isinstance(frames, np.ndarray):
        # Frame numbers sit by the frames' own width and a caption by the
        # padded width, as extraction and crowd movies have always placed them.
        number_pos = (5, frames.shape[-1] - 40)

        # H.264 cannot encode an odd width or height.
        if not np.mod(frames.shape[1], 2) == 0:
            frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), "constant", constant_values=0)

        if not np.mod(frames.shape[2], 2) == 0:
            frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=0)

        text_pos = (5, frames.shape[-1] - 40)

    if not pipe:
        pipe = VideoWriter(
            filename,
            codec=codec,
            # The output pixel format, which is what decides whether a browser
            # can play the result. Left to the encoder's own preference, an
            # rgb24 input lands on yuv444p, giving a High 4:4:4 Predictive
            # stream; browser H.264 is 8-bit 4:2:0 only, so the movie is
            # undecodable in the widgets that display it.
            pix_fmt="yuv420p",
            frame_size=frame_dimensions(frames, frame_size),
            fps=fps,
            threads=threads,
        )

    if isinstance(frames, np.ndarray):
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
            # float64 in 0-255, so the cast to uint8 happens here, before any
            # text is drawn, rather than at the write below.
            #
            # This is not pixel-identical to drawing on the float image, and
            # the difference is confined to the text. It is drawn with
            # LINE_AA, so its antialiased edges blend against whatever
            # precision the image carries, and blending in uint8 rather than
            # in float64 moves the boundary pixels of the characters. Every
            # other pixel is unchanged: the cast is the same truncation
            # wherever it runs.
            disp_img = disp_img.astype("uint8")
            if frame_numbers is not None:
                try:
                    cv2.putText(
                        disp_img,
                        str(frame_numbers[i]),
                        number_pos,
                        font,
                        text_scale,
                        white,
                        text_thickness,
                        cv2.LINE_AA,
                    )
                except (IndexError, ValueError):
                    # fewer numbers than frames, or the position is outside the frame
                    print("Could not overlay frame number on preview on video.")
            elif text is not None:
                cv2.putText(
                    disp_img,
                    text,
                    text_pos,
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
    return pipe
