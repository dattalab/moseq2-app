"""Write a synthetic Azure Kinect recording with an arena and a mouse in it.

The ``azure-mkv`` fixture's depth is a ramp, ``frame * 1000 + row * 100 +
column``, which says exactly which sample the reader returned and is what the
video tests want. It is not a recording anything can be extracted from: the
floor depth read at its centre brackets a single row of pixels, and a row has
no plane through it.

This one is laid out the way a real recording is, a ``COLOR`` track and a
16-bit ``DEPTH`` track named by Matroska title tags, and holds what
an Azure Kinect mounted 596 mm above a 17 inch arena sees: a floor tilted by a
few millimetres across the arena, the wall's rim closer to the camera, the
room beyond it farther away, and a mouse walking a circle on the floor. It is
written in well under a second, so a test builds it rather than fetching it.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

WIDTH, HEIGHT = 128, 112
NFRAMES = 60
FPS = 30

#: Camera face to arena floor, mm, at the centre of the frame.
FLOOR_DEPTH = 596
#: The floor's tilt, mm per pixel, along x and y.
FLOOR_TILT = (0.04, -0.03)
#: Arena radius and wall thickness, pixels.
ARENA_RADIUS, WALL = 48, 5
RIM_DEPTH, ROOM_DEPTH = 420, 1000
#: The mouse's body: half length and half width in pixels, height in mm.
MOUSE_HALF_LENGTH, MOUSE_HALF_WIDTH, MOUSE_HEIGHT = 14.0, 7.0, 40.0
#: The circle the mouse walks, pixels from the centre, and its turn per frame.
PATH_RADIUS, TURN_PER_FRAME = 24.0, 2 * np.pi / 90


def floor():
    """Depth of the bare arena: floor, wall rim and room, mm."""
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH].astype("float64")
    cy, cx = (HEIGHT - 1) / 2, (WIDTH - 1) / 2
    depth = FLOOR_DEPTH + FLOOR_TILT[0] * (cols - cx) + FLOOR_TILT[1] * (rows - cy)
    radius = np.hypot(rows - cy, cols - cx)
    depth[(radius >= ARENA_RADIUS) & (radius < ARENA_RADIUS + WALL)] = RIM_DEPTH
    depth[radius >= ARENA_RADIUS + WALL] = ROOM_DEPTH
    return depth


def mouse(i):
    """Height of the mouse above the floor in frame ``i``, mm, and its centre (x, y) in pixels."""
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH].astype("float64")
    cy, cx = (HEIGHT - 1) / 2, (WIDTH - 1) / 2
    angle = TURN_PER_FRAME * i
    x, y = cx + PATH_RADIUS * np.cos(angle), cy + PATH_RADIUS * np.sin(angle)
    # The body points along the direction of travel, tangent to the circle.
    heading = angle + np.pi / 2
    along = (cols - x) * np.cos(heading) + (rows - y) * np.sin(heading)
    across = -(cols - x) * np.sin(heading) + (rows - y) * np.cos(heading)
    r2 = (along / MOUSE_HALF_LENGTH) ** 2 + (across / MOUSE_HALF_WIDTH) ** 2
    height = MOUSE_HEIGHT * np.sqrt(np.clip(1 - r2, 0, None))
    return height, (x, y)


def depth_frames(seed=0):
    """The recording's depth frames, mm, as native ``uint16``, with 1 mm of sensor noise."""
    rng = np.random.default_rng(seed)
    bare = floor()
    frames = np.empty((NFRAMES, HEIGHT, WIDTH), np.uint16)
    for i in range(NFRAMES):
        height, _ = mouse(i)
        frames[i] = np.round(bare - height + rng.normal(0, 1, bare.shape))
    return frames


def write_azure_arena_mkv(path, seed=0):
    """Write the recording to ``path``, and return its depth frames as written."""
    import av

    frames = depth_frames(seed)
    with av.open(str(path), "w", format="matroska") as out:
        # An Azure Kinect stores depth as raw big-endian samples, which Matroska
        # here does not take; FFV1 is lossless, and the reader converts either.
        color = out.add_stream("ffv1", rate=FPS)
        color.width, color.height, color.pix_fmt = WIDTH, HEIGHT, "gray"
        color.metadata["title"] = "COLOR"
        depth = out.add_stream("ffv1", rate=FPS)
        depth.width, depth.height, depth.pix_fmt = WIDTH, HEIGHT, "gray16le"
        depth.metadata["title"] = "DEPTH"
        for stream in (color, depth):
            stream.time_base = Fraction(1, 1000)
        for i, sample in enumerate(frames):
            pts = round(i * 1000 / FPS)
            colour = av.VideoFrame.from_ndarray(np.full((HEIGHT, WIDTH), 128, np.uint8), format="gray")
            depth_frame = av.VideoFrame.from_ndarray(sample, format="gray16le")
            for frame, stream in ((colour, color), (depth_frame, depth)):
                frame.pts, frame.time_base = pts, stream.time_base
                out.mux(stream.encode(frame))
        out.mux(color.encode())
        out.mux(depth.encode())
    return frames
